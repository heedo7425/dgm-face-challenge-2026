"""Generate N face images from a (fine-tuned) UNet2DModel using DDIM.

Fixed seed for reproducibility (required for top-10 verification).
Outputs flat 256x256 jpgs into --out.
"""
import argparse
import os
import time

import torch
from PIL import Image
from diffusers import DDIMScheduler, UNet2DModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--unet", required=True, help="path to UNet2DModel dir or HF id")
    p.add_argument("--sched_src", default="google/ddpm-ema-celebahq-256",
                   help="source for scheduler betas/clip config")
    p.add_argument("--out", default="samples")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--bs", type=int, default=25)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20260610)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--gpu_duty", type=float, default=1.0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    unet = UNet2DModel.from_pretrained(args.unet).to(device).eval()
    sched = DDIMScheduler.from_pretrained(args.sched_src)
    sched.set_timesteps(args.steps)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    saved = 0
    idx = 0
    while saved < args.n:
        t0 = time.time()
        bs = min(args.bs, args.n - saved)
        x = torch.randn(bs, 3, args.res, args.res, device=device, generator=gen)
        for t in sched.timesteps:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                eps = unet(x, t).sample
            x = sched.step(eps.float(), t, x, eta=args.eta, generator=gen).prev_sample
        imgs = (x.clamp(-1, 1) + 1) / 2
        imgs = (imgs * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for k in range(bs):
            Image.fromarray(imgs[k]).save(os.path.join(args.out, f"img_{idx:04d}.jpg"), quality=95)
            idx += 1
        saved += bs
        if args.gpu_duty < 1.0:
            torch.cuda.synchronize()
            time.sleep((time.time() - t0) * (1.0 / args.gpu_duty - 1.0))
        print(f"sampled {saved}/{args.n}", flush=True)
    print(f"DONE: {saved} images in {args.out}")


if __name__ == "__main__":
    main()
