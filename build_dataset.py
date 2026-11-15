"""Stage 2: build the grounding dataset with the FROZEN world model.

For every scene we store:
  text        scene description only (no actions, no question)
  actions     M action sequences of length T            (N, M, T)
  latents     frozen-encoder latents E(o_t) for each seq  (N, M, T+1, D)
  labels      outcome labels per (scene, sequence)        {task: (N, M)}
  heldout     whether a sequence uses a held-out action   (N, M) bool

Held-out actions (default: up-right, down-left) never appear in writer or probe
TRAINING sequences. Test sequences that contain them measure generalisation to
unseen actions (H2).

Usage:  python build_dataset.py --wm runs/wm.pt --out runs/data.pt [--scenes 12000]
"""
from __future__ import annotations

import argparse
import random

import torch

import sim
from world_model import D_LATENT, Encoder

HELDOUT_ACTIONS = (1, 5)   # "up-right", "down-left"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="runs/wm.pt")
    ap.add_argument("--out", default="runs/data.pt")
    ap.add_argument("--scenes", type=int, default=12000)
    ap.add_argument("--M", type=int, default=4, help="action sequences per scene")
    ap.add_argument("--T", type=int, default=8, help="episode length (writer trains on a shorter prefix)")
    ap.add_argument("--text-style", choices=["exact", "qualitative"], default="exact")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    ck = torch.load(args.wm, map_location=dev, weights_only=False)
    enc = Encoder().to(dev)
    enc.load_state_dict(ck["encoder"])
    enc.eval().requires_grad_(False)

    rng, gen = random.Random(args.seed), torch.Generator().manual_seed(args.seed)
    N, M, T = args.scenes, args.M, args.T
    texts, init_pos = [], []
    all_actions = torch.empty(N, M, T, dtype=torch.long)
    latents = torch.empty(N, M, T + 1, D_LATENT, dtype=torch.float16)
    labels = {k: torch.empty(N, M, dtype=torch.long) for k in sim.TASKS}

    # train/val/test split by SCENE (70/10/20); test scenes get some held-out-action sequences
    n_train, n_val = int(0.7 * N), int(0.1 * N)
    split = torch.zeros(N, dtype=torch.long)
    split[n_train:n_train + n_val] = 1
    split[n_train + n_val:] = 2

    for s0 in range(0, N, args.chunk):
        s1 = min(N, s0 + args.chunk)
        B = s1 - s0
        scene = sim.sample_scenes(B, rng, "cpu")          # physics on CPU (many tiny ops)
        scene_dev = scene.to(dev)                          # rendering + encoding on GPU
        for b in range(B):
            texts.append(sim.describe(scene, b, rng, args.text_style))
        init_pos.append(scene.pos.clone())
        for m in range(M):
            is_test = split[s0:s1] == 2
            # train/val scenes: never use held-out actions. Test scenes: sequences m>=M/2 are
            # forced to contain at least one held-out action.
            acts = sim.directed_actions(scene, T, rng, "cpu", forbid=HELDOUT_ACTIONS) if m % 2 == 0 \
                else sim.random_actions(B, T, gen, "cpu", forbid=HELDOUT_ACTIONS)
            if m >= M // 2:
                pos_t = torch.randint(T, (B,), generator=gen)
                choice = torch.tensor(HELDOUT_ACTIONS)[torch.randint(len(HELDOUT_ACTIONS), (B,), generator=gen)]
                forced = acts.clone()
                forced[torch.arange(B), pos_t] = choice
                acts = torch.where(is_test.unsqueeze(1), forced, acts)
            states, contact = sim.simulate(scene, acts)
            frames = sim.frame_pairs(sim.render(scene_dev, states.to(dev)))
            with torch.no_grad():
                z = enc(frames)
            all_actions[s0:s1, m] = acts.cpu()
            latents[s0:s1, m] = z.half().cpu()
            for k, v in sim.outcome_labels(states, contact).items():
                labels[k][s0:s1, m] = v.cpu()
        print(f"scenes {s1}/{N}", flush=True)

    heldout = torch.zeros(N, M, dtype=torch.bool)
    for a in HELDOUT_ACTIONS:
        heldout |= (all_actions == a).any(-1)

    torch.save({"texts": texts, "init_pos": torch.cat(init_pos), "actions": all_actions,
                "latents": latents, "labels": labels, "split": split, "heldout": heldout,
                "args": vars(args)}, args.out)
    print("saved", args.out)
    print("example text:", texts[0])
    for k, v in labels.items():
        print(f"label {k}: class counts {torch.bincount(v.flatten(), minlength=sim.TASKS[k]).tolist()}")
    print("held-out-action sequences in test:", int(heldout[split == 2].sum()), "/", int((split == 2).sum() * M))


if __name__ == "__main__":
    main()
