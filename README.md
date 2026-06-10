# DGM Spring 2026 — Face Generation (Heedo Kim)

Reproducible inference/training code for the DGM Spring 2026 Face Generation
Challenge. Generates 1,000 × 256×256 face images from a CelebA-HQ-pretrained DDPM
fine-tuned on CelebV-HQ frames.

## Environment
```bash
python -m venv venv && . venv/bin/activate
# Blackwell (sm_120) needs the cu128 wheels:
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Pipeline
```bash
# 1. data: stream CelebV-HQ videos.tar, extract one 512px frame per clip
python extract_frames.py celebvhq_frames_512 10000 512

# 2. align + quality-filter to 256 (FFHQ-style, insightface)
MIN_IOD=70 MIN_SCORE=0.70 MAX_ASYM=0.35 \
    python align_faces.py celebvhq_frames_512 aligned_frames 256

# 3. fine-tune the base DDPM (UNet2DModel) on the aligned faces
python finetune.py --data aligned_frames --model google/ddpm-ema-celebahq-256 \
    --out ft_ckpt --bs 12 --steps 6000 --lr 1e-5 --ema_decay 0.999 \
    --grad_ckpt --seed 1234

# 4. generate 1000 images (fixed seed -> reproducible)
python sample.py --unet ft_ckpt/<CKPT>_ema --sched_src google/ddpm-ema-celebahq-256 \
    --out submission_images --n 1000 --bs 25 --steps 100 --eta 0.0 --seed 20260610

# 5. local proxy metrics (optional)
python eval_local.py --gen submission_images --real <held_out_ref>
```

Exact checkpoint / seed / sampler settings for the submitted images are recorded
in `SUBMISSION.txt`.

## Notes / findings
This repo also documents a negative result analyzed in the report: fine-tuning on
the freely available (re-encoded) CelebV-HQ frames improved a *local* FID proxy but
*worsened* the leaderboard FID, because that proxy shared the training data's
quality defects while the eval set is the curated high-quality CelebV-HQ. See the
report for the full failure analysis.

## Files
- `extract_frames.py` — stream CelebV-HQ tar, extract face frames.
- `align_faces.py` — FFHQ-style alignment + quality filtering (insightface).
- `finetune.py` — fine-tune the base DDPM UNet (bf16, EMA, optional GPU throttle).
- `sample.py` — DDIM sampling with a fixed seed.
- `sweep_checkpoints.py` — pick the best checkpoint by local FID/KID/IS.
- `eval_local.py` — local FID/KID/IS via torch-fidelity.
