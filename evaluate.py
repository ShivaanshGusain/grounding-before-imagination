"""Stage 5: evaluate grounding (H1, H2) and downstream usable information (H3).

Part 1 - grounding. On TEST scenes, roll each written state forward under the real action
         sequences and compare with the true future latents, per horizon, separately for
         sequences whose actions were SEEN vs UNSEEN by the writer/probes. (The frozen world
         model itself was pretrained on all 9 actions; "unseen" means unseen by writer + probe.)
Part 2 - usable information. Probes (linear and small MLP) predict outcome labels from
         different feature sets; we report held-out log loss (nats) and accuracy.

Train scenes are split: the first 60% fit the writers, the other 40% fit the probes, so every
probe feature is out-of-sample for the writer. Probes early-stop on val; results are on test.

Feature sets (all include scene-text features h and the actions as per-step (dx, dy) directions,
so a text-only probe can extrapolate to unseen diagonal actions):
  a_text              [h, A]                                      text only
  b_writer_state      [h, A, z0_hat_pred]                         written state, no rollout
  c_rollout_pred      [h, A, P-rollout(z0_hat_pred)]              the method
  c_rollout_stat      [h, A, P-rollout(z0_hat_static)]            static-grounding writer
  e1_state_randP      [h, A, P_rand-rollout(z0_hat_pred)]         same start state, random dynamics
  e2_state_shufA      [h, A, P-rollout(z0_hat_pred, shuffled A)]  same start state, wrong actions
  e3_randwriter       [h, A, P_rand-rollout(z0_hat_random)]       writer trained through P_rand
  d_priv_start        [h, A, P-rollout(E(o_0))]                   privileged: real start state
  f_priv_future       [h, A, true future latents]                 privileged: reader ceiling

Usage:  python evaluate.py --text runs/text_hf.pt --tag hf --seeds 0,1,2 --outdir results/hf
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import sim
from writer import Writer, load_predictor, random_predictor_seed, writer_train_mask

MODES = ["static", "predictive", "random"]
GROUPS = ["test_seen", "test_unseen"]


def load_writer(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    w = Writer(ck["d_in"]).to(device)
    w.load_state_dict(ck["writer"])
    return w.eval(), ck


def rollout_feats(roll):
    """roll (R,T,D) -> [last state, mean over time] (R, 2D)."""
    return torch.cat([roll[:, -1], roll.mean(1)], -1)


# --------------------------------------------------------------------------- probes
def train_probe(Xtr, ytr, Xva, yva, n_cls, kind, seed, device, max_epochs=200):
    """Columns that are (near-)constant in the probe-train rows are DROPPED, so features never seen
    varying in training (e.g. held-out actions, rare BOW bins) cannot be blown up by a tiny std."""
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    keep = sd > 1e-6
    mu, sd = mu[keep], sd[keep]

    def norm(X):
        return ((X[:, keep] - mu) / sd).clamp(-50, 50)

    Xtr_n, Xva_n = norm(Xtr), norm(Xva)
    d = Xtr_n.shape[1]
    if kind == "linear":
        model, wd = nn.Linear(d, n_cls), 1e-3
    else:
        model, wd = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_cls)), 1e-2
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=wd)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr_n), device=device)
        for i in range(0, len(perm), 1024):
            idx = perm[i:i + 1024]
            loss = F.cross_entropy(model(Xtr_n[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(Xva_n), yva).item()
        if best_state is None or vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(best_state)
    model.eval()
    return lambda X: model(norm(X))


@torch.no_grad()
def score(predict, X, y):
    logits = predict(X)
    return {"logloss": F.cross_entropy(logits, y).item(),
            "acc": (logits.argmax(-1) == y).float().mean().item()}


def mean_std(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    return m, (sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5


@torch.no_grad()
def grounding_metrics(roll, Z_flat, rows, P, A_flat, T):
    """Per-horizon rollout error / persistence error, plus privileged error, on `rows`."""
    per_t = {}
    for t in range(1, T + 1):
        true_t = Z_flat[rows, t]
        err = ((roll[rows, t - 1] - true_t) ** 2).mean().item()
        persist = ((Z_flat[rows, 0] - true_t) ** 2).mean().item()
        priv = ((P.rollout(Z_flat[rows, 0], A_flat[rows, :t])[:, -1] - true_t) ** 2).mean().item()
        per_t[t] = {"err": err, "persist": persist, "err_over_persistence": err / max(persist, 1e-12),
                    "privileged_over_persistence": priv / max(persist, 1e-12)}
    return per_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/data.pt")
    ap.add_argument("--text", default="runs/text_hf.pt")
    ap.add_argument("--wm", default="runs/wm.pt")
    ap.add_argument("--writer-pattern", default="runs/writer_{mode}_{tag}_s{seed}.pt")
    ap.add_argument("--tag", default="hf")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--outdir", default="results/hf")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    data = torch.load(args.data, weights_only=False)
    H_raw = torch.load(args.text, weights_only=False)["features"].float()
    Z = data["latents"].float().to(dev)                 # (N,M,T+1,D)
    A = data["actions"].to(dev)                          # (N,M,T)
    split_cpu = data["split"]
    split, heldout = split_cpu.to(dev), data["heldout"].to(dev)
    probe_scene = ((split_cpu == 0) & ~writer_train_mask(split_cpu)).to(dev)
    N, M, T = A.shape
    P = load_predictor(args.wm, "predictive", dev)

    # flattened (scene, sequence) rows
    scene_of = torch.arange(N, device=dev).repeat_interleave(M)
    A_flat = A.reshape(N * M, T)
    Z_flat = Z.reshape(N * M, T + 1, -1)
    A_dir = sim.ACTION_DIRS.to(dev)[A_flat].reshape(N * M, -1)      # (R, 2T) action directions
    row_split = split[scene_of]
    row_held = heldout.reshape(-1)
    groups = {"train": probe_scene[scene_of], "val": row_split == 1,
              "test_seen": (row_split == 2) & ~row_held, "test_unseen": (row_split == 2) & row_held}
    labels = {k: v.reshape(-1).to(dev) for k, v in data["labels"].items()}
    print({k: int(v.sum()) for k, v in groups.items()}, "rows per group", flush=True)

    with torch.no_grad():
        true_start = rollout_feats(P.rollout(Z_flat[:, 0], A_flat))
        true_future = rollout_feats(Z_flat[:, 1:])
    base = torch.cat([H_raw.to(dev)[scene_of], A_dir], -1)   # probes standardise internally
    gperm = torch.Generator().manual_seed(999)
    shuffled_A = A_flat[torch.randperm(N * M, generator=gperm).to(dev)]

    grounding = {}        # variant -> seed -> {z0 metrics, group -> per_t}
    probe_res = {}        # cond -> kind -> task -> group -> {"logloss": [...], "acc": [...]}
    for seed in seeds:
        P_rand = load_predictor(args.wm, "random", dev, seed=random_predictor_seed(seed))
        z0_hat = {}
        for mode in MODES:
            w, ck = load_writer(args.writer_pattern.format(mode=mode, tag=args.tag, seed=seed), dev)
            with torch.no_grad():
                z0_hat[mode] = w((H_raw.to(dev) - ck["mu"].to(dev)) / ck["sd"].to(dev))   # (N,D)
        with torch.no_grad():
            zp = z0_hat["predictive"][scene_of]
            rolls = {
                "static": P.rollout(z0_hat["static"][scene_of], A_flat),
                "predictive": P.rollout(zp, A_flat),
                "random_writer": P_rand.rollout(z0_hat["random"][scene_of], A_flat),
                "pred_state_randP": P_rand.rollout(zp, A_flat),
                "pred_state_shufA": P.rollout(zp, shuffled_A),
            }
        start_of = {"static": "static", "predictive": "predictive", "random_writer": "random",
                    "pred_state_randP": "predictive", "pred_state_shufA": "predictive"}

        # ---- Part 1: grounding
        test_scenes = split == 2
        zt0 = Z[test_scenes, 0, 0]
        for variant, roll in rolls.items():
            zh0 = z0_hat[start_of[variant]][test_scenes]
            sim_mat = F.normalize(zh0, dim=-1) @ F.normalize(zt0, dim=-1).T
            g = {"z0_mse": F.mse_loss(zh0, zt0).item(),
                 "z0_retrieval_top1": (sim_mat.argmax(1) == torch.arange(len(zt0), device=dev)).float().mean().item()}
            for gname in GROUPS:
                if groups[gname].sum() > 0:
                    g[gname] = grounding_metrics(roll, Z_flat, groups[gname], P, A_flat, T)
            grounding.setdefault(variant, {})[seed] = g
            parts = [f"{gk.split('_')[1]}: " + " ".join(f"{g[gk][t]['err_over_persistence']:.2f}" for t in range(1, T + 1))
                     for gk in GROUPS if gk in g]
            print(f"[seed {seed}] {variant:17s} z0 retrieval@1 {g['z0_retrieval_top1']:.3f}  err/persist "
                  + " | ".join(parts), flush=True)

        # ---- Part 2: usable information (features built one at a time to bound GPU memory)
        builders = {
            "a_text": lambda: base,
            "b_writer_state": lambda: torch.cat([base, zp], -1),
            "c_rollout_pred": lambda: torch.cat([base, rollout_feats(rolls["predictive"])], -1),
            "c_rollout_stat": lambda: torch.cat([base, rollout_feats(rolls["static"])], -1),
            "e1_state_randP": lambda: torch.cat([base, rollout_feats(rolls["pred_state_randP"])], -1),
            "e2_state_shufA": lambda: torch.cat([base, rollout_feats(rolls["pred_state_shufA"])], -1),
            "e3_randwriter": lambda: torch.cat([base, rollout_feats(rolls["random_writer"])], -1),
            "d_priv_start": lambda: torch.cat([base, true_start], -1),
            "f_priv_future": lambda: torch.cat([base, true_future], -1),
        }
        for cname, make in builders.items():
            X = make()
            for kind in ["linear", "mlp"]:
                for task, n_cls in sim.TASKS.items():
                    y = labels[task]
                    predict = train_probe(X[groups["train"]], y[groups["train"]], X[groups["val"]], y[groups["val"]],
                                          n_cls, kind, seed, dev)
                    for gname in GROUPS:
                        if groups[gname].sum() == 0:
                            continue
                        r = score(predict, X[groups[gname]], y[groups[gname]])
                        slot = (probe_res.setdefault(cname, {}).setdefault(kind, {}).setdefault(task, {})
                                .setdefault(gname, {"logloss": [], "acc": []}))
                        slot["logloss"].append(r["logloss"])
                        slot["acc"].append(r["acc"])
                    del predict
            del X
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[seed {seed}] probes done: {cname}", flush=True)
        del z0_hat, rolls, zp

    # label priors: predict train class frequencies (Laplace-smoothed), log loss on test
    priors = {}
    for task, n_cls in sim.TASKS.items():
        y = labels[task]
        p = torch.bincount(y[groups["train"]], minlength=n_cls).float() + 1
        p = p / p.sum()
        priors[task] = {g: (-torch.log(p[y[groups[g]]]).mean().item() if groups[g].sum() else float("nan"))
                        for g in GROUPS}

    json.dump(grounding, open(os.path.join(args.outdir, "grounding.json"), "w"), indent=1)
    json.dump({"probes": probe_res, "prior_logloss": priors}, open(os.path.join(args.outdir, "probes.json"), "w"), indent=1)
    write_report(args, grounding, probe_res, priors, T)


def write_report(args, grounding, probe_res, priors, T):
    L = [f"# Results ({args.tag} text features)", "",
         f"Seeds {args.seeds} vary writer init/order, probe init and the random-predictor draw. The world "
         "model and the dataset/split are FIXED across seeds, so ± understates the full variance.", ""]
    if args.tag == "bow":
        L += ["> **BOW caveat.** Bag-of-words features only weakly bind coordinates to the object they describe. "
              "Treat BOW as a lower bound / pipeline check; draw H1-H3 conclusions from the `hf` report.", ""]
    L += ["## Part 1: grounding (H1, H2)", "",
          "Rollout error of each start state ÷ error of *copying the true start latent* (persistence). "
          "Below 1 = better than standing still at the real start. Rows:",
          "- `predictive` / `static`: written start state rolled through the frozen predictor P (the comparison for H2)",
          "- `pred_state_randP`: the SAME predictive written state rolled through a random predictor (dynamics control)",
          "- `pred_state_shufA`: the SAME state rolled through P with another sequence's actions (action control)",
          "- `random_writer`: writer trained through a random predictor and rolled through it (training-signal control)",
          "- `privileged`: real start latent E(o_0) rolled through P (ceiling for the writer)", ""]
    for gname in GROUPS:
        L += [f"### {gname.replace('_', ' ')} actions", "",
              "| start state | z0 retrieval@1 | " + " | ".join(f"t={t}" for t in range(1, T + 1)) + " |",
              "|---|---|" + "---|" * T]
        priv_row = None
        for variant, per_seed in grounding.items():
            seeds = list(per_seed)
            if gname not in per_seed[seeds[0]]:
                continue
            ret = mean_std([per_seed[s]["z0_retrieval_top1"] for s in seeds])
            cells = []
            for t in range(1, T + 1):
                m, s = mean_std([per_seed[sd][gname][t]["err_over_persistence"] for sd in seeds])
                cells.append(f"{m:.2f}±{s:.2f}")
            L.append(f"| {variant} | {ret[0]:.3f} | " + " | ".join(cells) + " |")
            if priv_row is None:
                priv_row = [f"{per_seed[seeds[0]][gname][t]['privileged_over_persistence']:.2f}" for t in range(1, T + 1)]
        if priv_row:
            L.append("| privileged (real E(o_0)) | 1.000 | " + " | ".join(priv_row) + " |")
        L.append("")
    L += ["Writers were trained on horizons t ≤ --T-train (default 4); larger t tests horizon generalisation. "
          "H1: `predictive` well below 1 and below `pred_state_randP` / `pred_state_shufA`, approaching `privileged`. "
          "H2: `predictive` < `static`, especially for t > 4 and on unseen actions.", ""]

    L += ["## Part 2: usable information (H3)", "",
          "Held-out log loss in nats (lower is better), mean±std over seeds, accuracy in brackets. `prior` = "
          "predicting train class frequencies. Usable-information gain of the method = `a_text − c_rollout_pred`. "
          "Read the MLP tables as the stricter test.", ""]
    for kind in ["linear", "mlp"]:
        for gname in GROUPS:
            L += [f"### {kind} probe, {gname.replace('_', ' ')} actions", "",
                  "| condition | " + " | ".join(sim.TASKS) + " |", "|---|" + "---|" * len(sim.TASKS)]
            L.append("| prior | " + " | ".join(f"{priors[t][gname]:.3f}" for t in sim.TASKS) + " |")
            for cname, kinds in probe_res.items():
                cells = []
                for task in sim.TASKS:
                    slot = kinds[kind][task].get(gname)
                    if not slot:
                        cells.append("–")
                        continue
                    m, s = mean_std(slot["logloss"])
                    a, _ = mean_std(slot["acc"])
                    cells.append(f"{m:.3f}±{s:.3f} ({a * 100:.0f}%)")
                L.append(f"| {cname} | " + " | ".join(cells) + " |")
            L.append("")
    path = os.path.join(args.outdir, "report.md")
    open(path, "w").write("\n".join(L))
    print("wrote", path)


if __name__ == "__main__":
    main()
