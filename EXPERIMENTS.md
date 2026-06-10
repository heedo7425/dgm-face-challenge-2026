# Experiment log (for the report)

## Setup
- GPU: RTX 5060 Ti 16GB (Blackwell sm_120), torch 2.11.0+cu128.
- Base: google/ddpm-ema-celebahq-256 (UNet2DModel, CelebA-HQ 256, unconditional).
- Eval: leaderboard FID/IS/KID/TopPR vs 32,550 CelebV-HQ faces. Local proxy: torch-fidelity vs held-out CelebV-HQ.

## Local checkpoint sweep (v1 data = naive center-crop CelebV-HQ frames)
500 gen vs 2500 held-out (center-crop) ref:
| ckpt | FID | KID | IS |
|---|---|---|---|
| base | 97.5 | 0.0526 | 2.78 |
| ft step1000 | 111.0 | 0.0589 | 3.76 |
| ft step2000 | 101.5 | 0.0503 | 3.71 |
| ft step3000 | 95.8 | 0.0454 | 3.84 |
| ft step4000 | 89.2 | 0.0375 | 3.79 |
| **ft step5000** | **86.8** | **0.0352** | 3.84 |
| ft step6000 | 88.3 | 0.0372 | 3.94 |
- Finding: local FID bottoms at ~5000 steps then rises (over-fine-tuning).
- step1000 worse than base: EMA was reset at the throttle-resume (step 1000), under-converged early.

## Leaderboard calibration (CRITICAL)
- Submission #1 (ft step5000_ema, v1 center-crop data): leaderboard **FID 69.56, IS 4.17, KID 0.0397**.
- Pre-existing best submission: **FID 45.43**, IS 3.81, KID 0.0179, TopPR 0.9116.
- Top teams: FID 26-29, IS 5-6, KID ~0.001, TopPR 0.84-0.87.
- **KEY INSIGHT**: a clean/base-like model (FID 45) is CLOSER to the eval set than my
  fine-tuned model (FID 69). => the eval CelebV-HQ images are cleaner / more frontal /
  higher-quality than my naive center-cropped middle frames (which kept letterbox bars,
  on-screen text, wide/profile shots). My local FID was misleading because it used the
  same biased reference.
- **Failure mode**: distribution-matching to a *mis-extracted* proxy moved the generator
  away from the true eval distribution. Local proxy ≠ leaderboard when the proxy's
  extraction differs from the eval's.

## Pivot (v2 data = FFHQ-aligned + quality-filtered CelebV-HQ)
- Re-extract frames at 512, detect largest face (insightface), FFHQ-align to 256.
- Quality filter: det_score>=0.70, inter-ocular>=45px, frontality (nose asymmetry<0.35).
  -> drops profiles, tiny/far faces, low-confidence (junk frames). ~60% retention on test.
- Rationale: match the clean, frontal, aligned nature inferred for the eval set, while
  injecting CelebV-HQ identity/color statistics.
## v2 results (FFHQ-aligned, quality-filtered, 512-source -> sharp 256)
Data: re-extract 10k frames @512, insightface detect + FFHQ-align to 256,
filter det_score>=0.70 / inter-ocular>=70px / frontality(asym<0.35) -> 7214 kept
(dropped: small 924, profile 1448, lowscore 388, noface 26). Split 6014 train / 1200 ref.
Fine-tune base, lr 1e-5, EMA 0.999, bs 12, 6000 steps. Sweep (500 gen vs 1200 aligned ref):

| ckpt | FID | KID | IS |
|---|---|---|---|
| base | 76.6 | 0.0347 | 2.78 |
| **ft512 step4000** | **64.1** | **0.0166** | 3.12 |
| ft512 step6000 | 67.0 | 0.0179 | 3.34 |
| ft512 step3000 | 67.1 | 0.0190 | 3.27 |
| ft512 step5000 | 67.7 | 0.0189 | 3.30 |
| ft512 step2000 | 68.1 | 0.0210 | 3.14 |
| ft512 step1000 | 71.8 | 0.0239 | 3.28 |

- All ft512 ckpts beat base on FID & KID; step4000 best (KID halved vs base).
- Note local FID magnitudes dropped vs v1 (base 97.5 -> 76.6) because the v2 ref
  (aligned) is a better-posed reference; v2 numbers are more leaderboard-predictive.
- Calibration extrapolation: base local 76.6 ~ leaderboard 45 (ratio ~1.7);
  predicts ft512-step4000 leaderboard FID ~38, KID ~0.009 -> beats current 45 / 0.018.
- Submission strategy: best-per-metric is retained across submissions, so submitting
  ft512 (strong FID/KID) cannot hurt the IS (4.17) / TopPR (0.91) bests from earlier subs.
- Ablation for report: v1 center-crop (hurt, FID 69 leaderboard) vs v2 aligned+filtered;
  source-resolution effect (256 aligned: 5318 kept; 512 aligned: 7214 kept, sharper).

## v2 leaderboard (submission #530)
ft_aligned512 step4000_ema, DDIM 100, seed 20260610:
  FID 69.39, IS 3.29, KID 0.0368, TopPR 0.808  -> still ~69, no better than v1.

## Gentle fine-tune (lr 1e-6, 2000 steps)
Hypothesis: very low lr preserves base quality while nudging toward CelebV-HQ.
Result: samples converged to the same degraded CelebV-HQ look as the aggressive run
(seed-shared structure + same data). 2000 steps even at 1e-6 is enough to import the
data's quality defects -> no clean "near-base" regime reachable from this data.

## v3: GFPGAN restoration (submission #551)
Post-process v2 outputs with GFPGAN v1.4 (CPU, isolated venv; patched basicsr's
removed torchvision.functional_tensor import). 1000/1000 faces restored, 512px.
Leaderboard: FID 79.84, IS 3.56, KID 0.0501, TopPR 0.762 -> WORSE than v2 (69.4).
Lesson: GFPGAN imprints an FFHQ-idealized texture = a new distribution shift away
from CelebV-HQ. Perceptual cleanliness != lower FID.

## v4: weight interpolation (Wise-FT), submission #561
theta = 0.75*base + 0.25*ft_aligned512_step4000 (blend_weights.py), DDIM 250 steps.
Hoped to dial quality back toward base while keeping a mild CelebV-HQ shift.
Leaderboard: FID 81.27, IS 2.82, KID 0.0509, TopPR 0.625 -> WORSE than BOTH endpoints
(base ~45, full-ft 69). IS collapsed (2.82). The two diffusion-UNet checkpoints are not
linearly mode-connected; weight averaging fell into a degraded basin. Wise-FT intuition
from CLIP classifiers does not transfer to these generative weights.

## Leaderboard mechanic (observed)
Standing uses each team's single best-total submission (NOT per-metric cherry-pick):
our v1 IS 4.17 > displayed 3.81, yet 3.81 (from the old best-total sub) is shown.
So improving rank needs ONE submission balanced-better across all 4 metrics.

## Bottom line
All self-fine-tuned variants (v1/v2/v3) score FID 69-80, worse than the team's
pre-existing best (45.43). Raw base is disallowed (integrity). With the only freely
available (re-encoded) CelebV-HQ frames, we cannot beat 45 this cycle. Current best
(FID 45.43, TopPR 0.9116, rank ~26/43) stands. Value delivered: the failure analysis.
