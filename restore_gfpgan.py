"""Face-restore generated images with GFPGAN (post-process) to remove the
fine-tuning artifacts while keeping identity/pose. Runs on CPU in an isolated env.

in_dir  : folder of generated face jpgs
out_dir : restored jpgs (upscaled)
"""
import os
import sys

import cv2
from gfpgan import GFPGANer

IN = sys.argv[1] if len(sys.argv) > 1 else "submission_images_v2"
OUT = sys.argv[2] if len(sys.argv) > 2 else "submission_images_v3"
MODEL = sys.argv[3] if len(sys.argv) > 3 else "weights/GFPGANv1.4.pth"
UPSCALE = int(sys.argv[4]) if len(sys.argv) > 4 else 2
os.makedirs(OUT, exist_ok=True)

restorer = GFPGANer(model_path=MODEL, upscale=UPSCALE, arch="clean",
                    channel_multiplier=2, bg_upsampler=None)

files = sorted(f for f in os.listdir(IN) if f.lower().endswith((".jpg", ".jpeg", ".png")))
done = miss = 0
for i, f in enumerate(files):
    img = cv2.imread(os.path.join(IN, f))
    if img is None:
        continue
    try:
        _, _, restored = restorer.enhance(img, has_aligned=False,
                                          only_center_face=True, paste_back=True)
    except Exception:
        restored = None
    if restored is None:
        restored = cv2.resize(img, (img.shape[1] * UPSCALE, img.shape[0] * UPSCALE),
                              interpolation=cv2.INTER_LANCZOS4)
        miss += 1
    cv2.imwrite(os.path.join(OUT, f), restored, [cv2.IMWRITE_JPEG_QUALITY, 95])
    done += 1
    if (i + 1) % 100 == 0:
        print(f"restored {done}/{len(files)} (fallback {miss})", flush=True)
print(f"DONE: {done} restored, {miss} fallback (no face), -> {OUT}")
