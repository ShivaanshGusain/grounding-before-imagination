"""Stage 1: pretrain a small action-conditioned JEPA-style latent world model, then freeze it.

This stands in for V-JEPA 2-AC / JEPA-WMs so the whole pipeline can be debugged
end to end on a domain we fully control.

  encoder E: (frame_t, frame_{t-1}) -> z_t in R^D
  predictor P: (z_t, a_t) -> z_{t+1}       (residual MLP, rolled out recursively)
Loss: multi-step latent prediction against an EMA target encoder
      + VICReg variance/covariance terms to prevent collapse.

Usage:  python world_model.py --out runs/wm.pt [--steps 15000]
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import sim

D_LATENT = 128


class Encoder(nn.Module):
    def __init__(self, d=D_LATENT, width=64):
        super().__init__()
        c = width
        self.net = nn.Sequential(
            nn.Conv2d(6, c, 4, 2, 1), nn.GELU(),            # 32
            nn.Conv2d(c, 2 * c, 4, 2, 1), nn.GELU(),        # 16
            nn.Conv2d(2 * c, 4 * c, 4, 2, 1), nn.GELU(),    # 8
            nn.Conv2d(4 * c, 4 * c, 3, 1, 1), nn.GELU(),
            nn.Flatten(),
            nn.Linear(4 * c * 8 * 8, 512), nn.GELU(),
            nn.Linear(512, d),
        )

    def forward(self, x):  # x (..., 6, 64, 64)
        lead = x.shape[:-3]
        z = self.net(x.reshape(-1, *x.shape[-3:]))
        return z.reshape(*lead, -1)


class Predictor(nn.Module):
    def __init__(self, d=D_LATENT, hidden=512, n_actions=sim.N_ACTIONS):
        super().__init__()
        self.act = nn.Embedding(n_actions, 64)
        self.net = nn.Sequential(
            nn.Linear(d + 64, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, z, a):
        return z + self.net(torch.cat([z, self.act(a)], -1))

    def rollout(self, z0, actions):
        """z0 (B,D), actions (B,T) -> (B,T,D): states after each action."""
        zs, z = [], z0
        for t in range(actions.shape[1]):
            z = self(z, actions[:, t])
            zs.append(z)
        return torch.stack(zs, 1)


def vicreg_terms(z):
    z = z.reshape(-1, z.shape[-1])
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()
    cov = (z.T @ z) / (z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off ** 2).sum() / z.shape[1]
    return var_loss, cov_loss


def make_batch(B, T, rng, gen, device, size=64):
    scene = sim.sample_scenes(B, rng, "cpu")      # physics on CPU (many tiny ops), render on device
    # half random actions, half directed towards a puck (so contacts happen)
    a_rand = sim.random_actions(B // 2, T, gen, "cpu")
    sub = sim.Scene(scene.pos[B // 2:], scene.mass[B // 2:], scene.friction[B // 2:],
                    scene.rgb[B // 2:], scene.bg[B // 2:], scene.meta[B // 2:])
    a_dir = sim.directed_actions(sub, T, rng, "cpu")
    actions = torch.cat([a_rand, a_dir], 0)
    states, _ = sim.simulate(scene, actions)
    states = states.to(device)
    frames = sim.render(scene.to(device), states, size)
    return sim.frame_pairs(frames), actions.to(device), states


def ridge_r2(X, Y, Xt, Yt, lam=1e-2):
    """Closed-form ridge probe; returns mean R^2 over target dims."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X, Xt = (X - mu) / sd, (Xt - mu) / sd
    X = torch.cat([X, torch.ones(len(X), 1, device=X.device)], 1)
    Xt = torch.cat([Xt, torch.ones(len(Xt), 1, device=Xt.device)], 1)
    W = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device=X.device), X.T @ Y)
    pred = Xt @ W
    ss_res = ((Yt - pred) ** 2).sum(0)
    ss_tot = ((Yt - Yt.mean(0)) ** 2).sum(0) + 1e-9
    return (1 - ss_res / ss_tot).mean().item()


@torch.no_grad()
def sanity_check(enc, pred, rng, gen, device, T, n=2048):
    """How good is the frozen world model? (state decodability + rollout quality)."""
    enc.eval(); pred.eval()
    xs, acts, sts = [], [], []
    for _ in range(n // 256):
        x, a, s = make_batch(256, T, rng, gen, device)
        xs.append(x); acts.append(a); sts.append(s)
    x, a, s = torch.cat(xs), torch.cat(acts), torch.cat(sts)
    z = enc(x)                                   # (n, T+1, D)
    zr = pred.rollout(z[:, 0], a)                # (n, T, D)
    target = s[..., :2].reshape(len(s), T + 1, -1)  # positions of all bodies
    half = len(s) // 2
    out = {"state_probe_R2_encoder_t0": ridge_r2(z[:half, 0], target[:half, 0], z[half:, 0], target[half:, 0])}
    for t in range(1, T + 1):
        out[f"state_probe_R2_rollout_t{t}"] = ridge_r2(z[:half, t], target[:half, t], zr[half:, t - 1], target[half:, t])
        err = ((zr[:, t - 1] - z[:, t]) ** 2).mean().item()
        persist = ((z[:, 0] - z[:, t]) ** 2).mean().item()
        out[f"rollout_err_over_persistence_t{t}"] = err / max(persist, 1e-9)
    enc.train(); pred.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/wm.pt")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--T", type=int, default=6, help="rollout length during pretraining")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng, gen = random.Random(args.seed), torch.Generator().manual_seed(args.seed)
    dev = torch.device(args.device)
    enc, pred = Encoder().to(dev), Predictor().to(dev)
    tgt = copy.deepcopy(enc).requires_grad_(False)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        x, a, _ = make_batch(args.batch, args.T, rng, gen, dev)
        z = enc(x)                                      # (B, T+1, D)
        with torch.no_grad():
            zt = tgt(x)
        one_step = pred(z[:, :-1].reshape(-1, D_LATENT), a.reshape(-1)).reshape_as(z[:, 1:])
        multi = pred.rollout(z[:, 0], a)
        loss_pred = F.mse_loss(one_step, zt[:, 1:]) + F.mse_loss(multi, zt[:, 1:])
        loss_align = F.mse_loss(z, zt)                  # keep online close to target space
        var_l, cov_l = vicreg_terms(z)
        loss = loss_pred + 0.1 * loss_align + 1.0 * var_l + 0.04 * cov_l
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(pred.parameters()), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for p, q in zip(tgt.parameters(), enc.parameters()):
                p.mul_(args.ema).add_(q.detach(), alpha=1 - args.ema)
        if step % 200 == 0 or step == 1:
            print(f"step {step:6d} loss {loss.item():.4f} pred {loss_pred.item():.4f} "
                  f"var {var_l.item():.3f} cov {cov_l.item():.3f} ({time.time() - t0:.0f}s)", flush=True)

    # The frozen world model = EMA target encoder + predictor (predictor was trained to hit its outputs).
    report = sanity_check(tgt, pred, rng, gen, dev, args.T, n=512 if args.steps < 1000 else 2048)
    print(json.dumps(report, indent=2))
    torch.save({"encoder": tgt.state_dict(), "predictor": pred.state_dict(),
                "args": vars(args), "sanity": report}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
