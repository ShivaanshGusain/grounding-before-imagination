"""Stage B helper (run AFTER Stage A works): inspect Meta's pretrained JEPA-WMs models.

The JEPA-WMs README documents loading via torch.hub, e.g.
    model, preprocessor = torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_wall')
but the exact encoder/predictor call signatures are NOT assumed here. This script only
prints the structure so we can write a correct adapter (encode frames -> latent,
predictor(latent history, action) -> next latent) before swapping it in for our own
world model in writer.py / evaluate.py.

Usage:  python inspect_jepa_wms.py --name jepa_wm_wall
"""
import argparse
import inspect

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="jepa_wm_wall",
                    help="jepa_wm_wall | jepa_wm_pusht | jepa_wm_pointmaze | jepa_wm_metaworld | jepa_wm_droid")
    args = ap.parse_args()
    out = torch.hub.load("facebookresearch/jepa-wms", args.name)
    model, preprocessor = out if isinstance(out, tuple) else (out, None)
    print("model type:", type(model))
    print(model)
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        print(f"child {name}: {type(mod).__name__}, {n / 1e6:.1f}M params")
        if hasattr(mod, "forward"):
            print("   forward", inspect.signature(mod.forward))
    print("model.forward", inspect.signature(model.forward))
    print("public methods:", [m for m in dir(model) if not m.startswith("_") and callable(getattr(model, m))][:80])
    print("preprocessor:", type(preprocessor), preprocessor)


if __name__ == "__main__":
    main()
