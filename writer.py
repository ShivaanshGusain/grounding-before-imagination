"""Stage 4: train a text -> world-model-state WRITER.

  --mode static      match the initial latent only:       || W(h) - E(o_0) ||^2
  --mode predictive  train THROUGH the frozen predictor:  sum_t || P^t(W(h), a) - E(o_t) ||^2  (+0.1 static)
  --mode random      same as predictive, but through a RANDOMLY INITIALISED frozen predictor
                     (control: same architecture and compute, no learned dynamics)

The predictor's weights are frozen; gradients flow through it into the writer.
Training uses only train-split sequences (which never contain held-out actions)
and only the first --T-train steps, so longer horizons are a generalisation test.

Usage:  python writer.py --mode predictive --text runs/text_hf.pt --seed 0 --out runs/writer_predictive_s0.pt
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from world_model import D_LATENT, Predictor


class Writer(nn.Module):
    def __init__(self, d_in, d_out=D_LATENT, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, d_out))

    def forward(self, h):
        return self.net(h)


def load_predictor(wm_path, mode, device, seed=1234):
    if mode == "random":
        # fixed random init shared by training and evaluation; fork so the caller's RNG is untouched
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            pred = Predictor().to(device)
    else:
        pred = Predictor().to(device)
        pred.load_state_dict(torch.load(wm_path, map_location=device, weights_only=False)["predictor"])
    return pred.eval().requires_grad_(False)


def losses(writer, pred, h, z, a, mode, T_train):
    """h (B,F); z (B,M,T+1,D); a (B,M,T). Returns (loss, rollout_mse, static_mse)."""
    B, M = a.shape[:2]
    z0_hat = writer(h)
    static = F.mse_loss(z0_hat, z[:, 0, 0])
    if mode == "static":
        return static, torch.tensor(float("nan")), static
    start = z0_hat.unsqueeze(1).expand(B, M, -1).reshape(B * M, -1)
    roll = pred.rollout(start, a[:, :, :T_train].reshape(B * M, T_train))
    roll_mse = F.mse_loss(roll, z[:, :, 1:T_train + 1].reshape(B * M, T_train, -1))
    return roll_mse + 0.1 * static, roll_mse, static


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "predictive", "random"], required=True)
    ap.add_argument("--data", default="runs/data.pt")
    ap.add_argument("--text", default="runs/text_hf.pt")
    ap.add_argument("--wm", default="runs/wm.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--T-train", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    data = torch.load(args.data, weights_only=False)
    H = torch.load(args.text, weights_only=False)["features"].float()
    split = data["split"]
    tr, va = split == 0, split == 1
    mu, sd = H[tr].mean(0), H[tr].std(0) + 1e-5
    H = ((H - mu) / sd).to(dev)
    Z = data["latents"].float().to(dev)
    A = data["actions"].to(dev)

    pred = load_predictor(args.wm, args.mode, dev)
    writer = Writer(H.shape[1]).to(dev)
    opt = torch.optim.AdamW(writer.parameters(), lr=args.lr, weight_decay=0.05)

    tr_idx, va_idx = tr.nonzero().squeeze(1).to(dev), va.nonzero().squeeze(1).to(dev)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(args.epochs):
        writer.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=dev)]
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            loss, _, _ = losses(writer, pred, H[idx], Z[idx], A[idx], args.mode, args.T_train)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        writer.eval()
        with torch.no_grad():
            vl, vroll, vstat = losses(writer, pred, H[va_idx], Z[va_idx], A[va_idx], args.mode, args.T_train)
        print(f"[{args.mode} s{args.seed}] epoch {ep:3d} val loss {vl.item():.4f} "
              f"rollout {vroll.item():.4f} static {vstat.item():.4f}", flush=True)
        if vl.item() < best - 1e-5:
            best, bad = vl.item(), 0
            best_state = {k: v.detach().clone() for k, v in writer.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break

    torch.save({"writer": best_state, "d_in": H.shape[1], "mu": mu, "sd": sd,
                "mode": args.mode, "args": vars(args), "best_val": best}, args.out)
    print("saved", args.out, "best val", best)


if __name__ == "__main__":
    main()
