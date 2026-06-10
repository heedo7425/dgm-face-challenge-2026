"""For each fine-tune checkpoint (+ base model), sample a small batch with a fixed
seed and compute local FID/KID/IS vs a held-out CelebV-HQ reference folder.
Prints a ranked table so we can pick the best checkpoint for the full 1000-image run.
"""
import argparse
import glob
import os
import shutil
import time

import torch
import torch_fidelity
from PIL import Image
from diffusers import DDIMScheduler, UNet2DModel


def sample(unet_path, sched_src, out, n, bs, steps, seed, res=256, duty=1.0):
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(out, "*.jpg")):
        os.remove(f)
    device = "cuda"
    unet = UNet2DModel.from_pretrained(unet_path).to(device).eval()
    sched = DDIMScheduler.from_pretrained(sched_src)
    sched.set_timesteps(steps)
    gen = torch.Generator(device=device).manual_seed(seed)
    idx = 0
    with torch.no_grad():
        while idx < n:
            t0 = time.time()
            b = min(bs, n - idx)
            x = torch.randn(b, 3, res, res, device=device, generator=gen)
            for t in sched.timesteps:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    eps = unet(x, t).sample
                x = sched.step(eps.float(), t, x, eta=0.0, generator=gen).prev_sample
            imgs = ((x.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            for k in range(b):
                Image.fromarray(imgs[k]).save(os.path.join(out, f"s_{idx:04d}.jpg"), quality=95)
                idx += 1
            if duty < 1.0:
                torch.cuda.synchronize()
                time.sleep((time.time() - t0) * (1.0 / duty - 1.0))
    del unet
    torch.cuda.empty_cache()


def metrics(gen_dir, real_dir, kid_subset):
    m = torch_fidelity.calculate_metrics(
        input1=gen_dir, input2=real_dir, cuda=True,
        fid=True, kid=True, isc=True, kid_subset_size=kid_subset, verbose=False,
    )
    return (m["frechet_inception_distance"],
            m["kernel_inception_distance_mean"],
            m["inception_score_mean"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="held-out real reference folder")
    p.add_argument("--ckpt_glob", default="ft_ckpt/unet_step*_ema")
    p.add_argument("--include_base", action="store_true")
    p.add_argument("--sched_src", default="google/ddpm-ema-celebahq-256")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--bs", type=int, default=25)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260610)
    p.add_argument("--gpu_duty", type=float, default=0.7)
    args = p.parse_args()

    n_ref = len([f for f in os.listdir(args.ref) if f.endswith(".jpg")])
    kid_subset = min(args.n, n_ref, 1000)

    cands = sorted(glob.glob(args.ckpt_glob),
                   key=lambda s: int("".join(c for c in s.split("step")[-1] if c.isdigit())))
    if args.include_base:
        cands = [args.sched_src] + cands

    results = []
    for c in cands:
        tag = c.replace("/", "_")
        out = f"sweep_samples/{tag}"
        print(f"\n>>> sampling {c} ({args.n} imgs)...", flush=True)
        sample(c, args.sched_src, out, args.n, args.bs, args.steps, args.seed, duty=args.gpu_duty)
        fid, kid, isc = metrics(out, args.ref, kid_subset)
        results.append((c, fid, kid, isc))
        print(f"    FID {fid:.3f}  KID {kid:.5f}  IS {isc:.3f}", flush=True)

    results.sort(key=lambda r: r[1])  # by FID asc
    print("\n=========== RANKED BY LOCAL FID (lower better) ===========")
    print(f"{'checkpoint':45s} {'FID':>9s} {'KID':>10s} {'IS':>7s}")
    for c, fid, kid, isc in results:
        print(f"{c:45s} {fid:9.3f} {kid:10.5f} {isc:7.3f}")
    print(f"\nBEST (local FID): {results[0][0]}")


if __name__ == "__main__":
    main()
