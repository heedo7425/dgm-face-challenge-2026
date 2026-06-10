"""Local FID/KID/IS between a generated folder and a real reference folder,
using torch-fidelity. This is a *proxy* for the leaderboard (which uses the
full 32,550 CelebV-HQ set); use it to confirm fine-tuning moves us closer.
"""
import argparse

import torch_fidelity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen", required=True)
    p.add_argument("--real", required=True)
    p.add_argument("--kid_subset", type=int, default=1000)
    args = p.parse_args()

    m = torch_fidelity.calculate_metrics(
        input1=args.gen,
        input2=args.real,
        cuda=True,
        fid=True,
        kid=True,
        isc=True,
        kid_subset_size=args.kid_subset,
        verbose=False,
    )
    print("=== local metrics (gen vs real) ===")
    print(f"FID            : {m.get('frechet_inception_distance'):.4f}  (lower better)")
    print(f"KID mean       : {m.get('kernel_inception_distance_mean'):.6f}  (lower better)")
    print(f"KID std        : {m.get('kernel_inception_distance_std'):.6f}")
    print(f"IS mean        : {m.get('inception_score_mean'):.4f}  (higher better)")
    print(f"IS std         : {m.get('inception_score_std'):.4f}")


if __name__ == "__main__":
    main()
