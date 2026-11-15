"""Stage 3: frozen text features for every scene description.

Two encoders:
  --encoder hf   a frozen HuggingFace causal LM; mean-pooled hidden states from a few layers
  --encoder bow  hashed word + character n-gram counts (no download; also a useful baseline)

Usage:
  python text_features.py --data runs/data.pt --out runs/text_hf.pt --encoder hf --model Qwen/Qwen2.5-0.5B
  python text_features.py --data runs/data.pt --out runs/text_bow.pt --encoder bow
"""
from __future__ import annotations

import argparse
import hashlib
import re

import torch


_ENTITIES = ["red", "blue", "green", "yellow", "purple", "pusher", "table", "surface"]


def _bound_grams(t):
    """Entity-bound features: in each sentence, tie every word/number to the sentence's entity
    (colour / pusher / table), plus coarse bins of the k-th number, so BOW knows WHICH object is where."""
    out = []
    for sent in t.split(". "):
        words = re.findall(r"[a-z]+", sent)
        ent = next((w for w in words if w in _ENTITIES), None)
        if ent is None:
            continue
        ent = "table" if ent == "surface" else ent
        out += [f"{ent}|{w}" for w in words]
        for k, num in enumerate(re.findall(r"\d+\.\d+", sent)):
            v = float(num)
            out += [f"{ent}|n{k}|b10_{int(v * 10)}", f"{ent}|n{k}|b20_{int(v * 20)}"]
    return out


def bow_features(texts, dim=4096):
    X = torch.zeros(len(texts), dim)
    for i, t in enumerate(texts):
        t = t.lower()
        grams = re.findall(r"[a-z]+|\d+\.\d+|\d+", t)
        s = f" {t} "
        grams += [s[j:j + n] for n in (3, 4, 5) for j in range(len(s) - n + 1)]
        grams += _bound_grams(t)
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16)
            X[i, h % dim] += 1.0 if (h >> 31) & 1 else -1.0
    return X / (X.norm(dim=1, keepdim=True) + 1e-6)


@torch.no_grad()
def hf_features(texts, model_name, layers, batch, device):
    from transformers import AutoModel, AutoTokenizer  # imported lazily: only needed for --encoder hf
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device).eval()
    feats = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True).to(device)
        out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = [(out.hidden_states[l].float() * mask).sum(1) / mask.sum(1) for l in layers]
        feats.append(torch.cat(pooled, -1).cpu())
        if (i // batch) % 20 == 0:
            print(f"{i + len(texts[i:i + batch])}/{len(texts)}", flush=True)
    return torch.cat(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/data.pt")
    ap.add_argument("--out", default="runs/text_hf.pt")
    ap.add_argument("--encoder", choices=["hf", "bow"], default="hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="-1,-8,-14", help="hidden-state layers to mean-pool and concatenate")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    texts = torch.load(args.data, weights_only=False)["texts"]
    if args.encoder == "bow":
        X = bow_features(texts)
    else:
        X = hf_features(texts, args.model, [int(l) for l in args.layers.split(",")], args.batch, args.device)
    torch.save({"features": X.half(), "encoder": args.encoder, "model": args.model}, args.out)
    print("saved", args.out, tuple(X.shape))


if __name__ == "__main__":
    main()
