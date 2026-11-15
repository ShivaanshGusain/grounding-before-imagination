"""Stage 5: evaluate grounding (H1, H2) and downstream usable information (H3).

Part 1 - grounding. On TEST scenes, roll each writer's state forward under the real
         action sequences and compare with the true future latents, per horizon,
         separately for sequences with SEEN vs UNSEEN (held-out) actions.
Part 2 - usable information. Train probes (linear and small MLP) to predict outcome
         labels from different feature sets; report held-out log loss (nats) and accuracy.

Feature sets (all include the scene-text features h and one-hot actions A):
  a_text          [h, A]                                     text only
  b_writer_state  [h, A, z0_hat]                             written state, no rollout
  c_rollout_pred  [h, A, rollout(z0_hat_predictive)]         full method
  c_rollout_stat  [h, A, rollout(z0_hat_static)]             static-grounding writer
  e_random_pred   [h, A, rollout_randomP(z0_hat_random)]     no learned dynamics (control)
  d_priv_start    [h, A, rollout(E(o_0))]                    privileged: real start state
  f_priv_future   [h, A, true future latents]                privileged: reader ceiling

Usage:  python evaluate.py --text runs/text_hf.pt --seeds 0,1,2 --outdir results/hf
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
from writer import Writer, load_predictor

MODES = ["static", "predictive", "random"]


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
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    norm = lambda X: (X - mu) / sd
    Xtr, Xva = norm(Xtr), norm(Xva)
    d = Xtr.shape[1]
    if kind == "linear":
        model = nn.Linear(d, n_cls)
        wd = 1e-3
    else:
        model = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_cls))
        wd = 1e-2
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=wd)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(perm), 1024):
            idx = perm[i:i + 1024]
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(Xva), yva).item()
        if vl < best - 1e-4:
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
    split, heldout = data["split"].to(dev), data["heldout"].to(dev)
    N, M, T = A.shape
    P = load_predictor(args.wm, "predictive", dev)
    P_rand = load_predictor(args.wm, "random", dev)

    # flattened (scene, sequence) rows
    scene_of = torch.arange(N, device=dev).repeat_interleave(M)
    A_flat = A.reshape(N * M, T)
    Z_flat = Z.reshape(N * M, T + 1, -1)
    A_onehot = F.one_hot(A_flat, sim.N_ACTIONS).float().reshape(N * M, -1)
    row_split = split[scene_of]
    row_held = heldout.reshape(-1)
    groups = {"train": row_split == 0, "val": row_split == 1,
              "test_seen": (row_split == 2) & ~row_held, "test_unseen": (row_split == 2) & row_held}
    labels = {k: v.reshape(-1).to(dev) for k, v in data["labels"].items()}

    true_start = rollout_feats(P.rollout(Z_flat[:, 0], A_flat))
    true_future = rollout_feats(Z_flat[:, 1:])

    grounding = {}        # mode -> seed -> group -> metrics
    probe_res = {}        # cond -> kind -> task -> group -> list over seeds
    for seed in seeds:
        z0_hat, rolls = {}, {}
        for mode in MODES:
            w, ck = load_writer(args.writer_pattern.format(mode=mode, tag=args.tag, seed=seed), dev)
            H = ((H_raw - ck["mu"]) / ck["sd"]).to(dev)
            with torch.no_grad():
                zh = w(H)                                  # (N,D)
                pred = P_rand if mode == "random" else P
                roll = pred.rollout(zh[scene_of], A_flat)  # (N*M,T,D)
            z0_hat[mode], rolls[mode] = zh, roll

            # ---- Part 1: grounding
            g = {}
            test_scenes = (split == 2)
            zt0 = Z[test_scenes, 0, 0]
            zh0 = zh[test_scenes]
            sim_mat = F.normalize(zh0, dim=-1) @ F.normalize(zt0, dim=-1).T
            g["z0_mse"] = F.mse_loss(zh0, zt0).item()
            g["z0_retrieval_top1"] = (sim_mat.argmax(1) == torch.arange(len(zt0), device=dev)).float().mean().item()
            for gname in ["test_seen", "test_unseen"]:
                rows = groups[gname]
                if rows.sum() == 0:
                    continue
                per_t = {}
                for t in range(1, T + 1):
                    true_t = Z_flat[rows, t]
                    err = ((roll[rows, t - 1] - true_t) ** 2).mean().item()
                    persist = ((Z_flat[rows, 0] - true_t) ** 2).mean().item()
                    priv = ((P.rollout(Z_flat[rows, 0], A_flat[rows, :t])[:, -1] - true_t) ** 2).mean().item()
                    per_t[t] = {"err": err, "err_over_persistence": err / persist, "privileged_err": priv}
                g[gname] = per_t
            grounding.setdefault(mode, {})[seed] = g
            print(f"[seed {seed}] {mode:10s} z0 retrieval@1 {g['z0_retrieval_top1']:.3f}  "
                  f"err/persist t=1..{T} seen: " +
                  " ".join(f"{g['test_seen'][t]['err_over_persistence']:.2f}" for t in range(1, T + 1)), flush=True)

        # ---- Part 2: usable information
        Hrows = ((H_raw - H_raw[data["split"] == 0].mean(0)) / (H_raw[data["split"] == 0].std(0) + 1e-5)).to(dev)[scene_of]
        base = torch.cat([Hrows, A_onehot], -1)
        conds = {
            "a_text": base,
            "b_writer_state": torch.cat([base, z0_hat["predictive"][scene_of]], -1),
            "c_rollout_pred": torch.cat([base, rollout_feats(rolls["predictive"])], -1),
            "c_rollout_stat": torch.cat([base, rollout_feats(rolls["static"])], -1),
            "e_random_pred": torch.cat([base, rollout_feats(rolls["random"])], -1),
            "d_priv_start": torch.cat([base, true_start], -1),
            "f_priv_future": torch.cat([base, true_future], -1),
        }
        for cname, X in conds.items():
            for kind in ["linear", "mlp"]:
                for task, n_cls in sim.TASKS.items():
                    y = labels[task]
                    predict = train_probe(X[groups["train"]], y[groups["train"]], X[groups["val"]], y[groups["val"]],
                                          n_cls, kind, seed, dev)
                    for gname in ["test_seen", "test_unseen"]:
                        if groups[gname].sum() == 0:
                            continue
                        r = score(predict, X[groups[gname]], y[groups[gname]])
                        slot = probe_res.setdefault(cname, {}).setdefault(kind, {}).setdefault(task, {}).setdefault(gname, {"logloss": [], "acc": []})
                        slot["logloss"].append(r["logloss"]); slot["acc"].append(r["acc"])
            print(f"[seed {seed}] probes done: {cname}", flush=True)

    # label priors (entropy of train class frequencies, evaluated on test) for reference
    priors = {}
    for task, n_cls in sim.TASKS.items():
        y = labels[task]
        p = torch.bincount(y[groups["train"]], minlength=n_cls).float() + 1
        p = p / p.sum()
        priors[task] = {g: (-torch.log(p[y[groups[g]]]).mean().item() if groups[g].sum() else float("nan"))
                        for g in ["test_seen", "test_unseen"]}

    json.dump(grounding, open(os.path.join(args.outdir, "grounding.json"), "w"), indent=1)
    json.dump({"probes": probe_res, "prior_logloss": priors}, open(os.path.join(args.outdir, "probes.json"), "w"), indent=1)
    write_report(args, grounding, probe_res, priors, T)


def write_report(args, grounding, probe_res, priors, T):
    L = [f"# Results ({args.tag} text features, seeds {args.seeds})", ""]
    L += ["## Part 1: grounding (H1, H2)", "",
          "Rollout error of the text-written state divided by the error of *copying the true start state* "
          "(persistence). Below 1 = the rolled-out written state predicts the future better than "
          "standing still at the real start. `privileged` = rollout from the real start latent E(o_0).", ""]
    for gname in ["test_seen", "test_unseen"]:
        L += [f"### {gname.replace('_', ' ')} actions", "",
              "| writer | z0 retrieval@1 | " + " | ".join(f"t={t}" for t in range(1, T + 1)) + " |",
              "|---|---|" + "---|" * T]
        for mode, per_seed in grounding.items():
            seeds = list(per_seed)
            if gname not in per_seed[seeds[0]]:
                continue
            ret = mean_std([per_seed[s]["z0_retrieval_top1"] for s in seeds])
            cells = []
            for t in range(1, T + 1):
                m, s = mean_std([per_seed[sd][gname][t]["err_over_persistence"] for sd in seeds])
                cells.append(f"{m:.2f}±{s:.2f}")
            L.append(f"| {mode} | {ret[0]:.3f} | " + " | ".join(cells) + " |")
        first = list(grounding)[0]
        s0 = list(grounding[first])[0]
        if gname in grounding[first][s0]:
            cells = []
            for t in range(1, T + 1):
                g = grounding[first][s0][gname][t]
                persist = g["err"] / g["err_over_persistence"]
                cells.append(f"{g['privileged_err'] / persist:.2f}")
            L.append("| privileged (real E(o_0)) | 1.000 | " + " | ".join(cells) + " |")
        L.append("")
    L += ["Writers were trained on horizons t ≤ --T-train (default 4); larger t tests horizon generalisation.", ""]

    L += ["## Part 2: usable information (H3)", "",
          "Held-out log loss in nats (lower is better), mean±std over seeds. `prior` = predicting train class "
          "frequencies. The usable-information gain of the method is `a_text − c_rollout_pred`.", ""]
    for kind in ["linear", "mlp"]:
        for gname in ["test_seen", "test_unseen"]:
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
