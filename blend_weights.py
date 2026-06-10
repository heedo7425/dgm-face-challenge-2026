"""Wise-FT style weight interpolation between the base UNet and a fine-tuned one:
theta = (1 - alpha) * base + alpha * finetuned.

Equivalent to a very conservative fine-tune; keeps base sample quality while
shifting slightly toward the CelebV-HQ distribution. Fully reproducible.
"""
import sys

import torch
from diffusers import UNet2DModel

BASE = sys.argv[1] if len(sys.argv) > 1 else "google/ddpm-ema-celebahq-256"
FT = sys.argv[2] if len(sys.argv) > 2 else "ft_aligned512/unet_step4000_ema"
ALPHA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
OUT = sys.argv[4] if len(sys.argv) > 4 else f"blend_a{ALPHA:.2f}"

base = UNet2DModel.from_pretrained(BASE)
ft = UNet2DModel.from_pretrained(FT)
bs, fs = base.state_dict(), ft.state_dict()
assert bs.keys() == fs.keys()
out = {k: (1.0 - ALPHA) * bs[k].float() + ALPHA * fs[k].float() for k in bs}
base.load_state_dict(out)
base.save_pretrained(OUT)
print(f"saved blend alpha={ALPHA} ({BASE} -> {FT}) to {OUT}")
