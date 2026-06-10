"""Re-process raw CelebV-HQ frames into FFHQ/CelebA-HQ-style ALIGNED face crops.

The first dataset used naive center-crops, which kept letterbox bars, on-screen
text and wide shots -> a distribution mismatch with the (aligned) eval set, which
made fine-tuning hurt leaderboard FID. Here we detect the largest face with
insightface (onnxruntime, no torch conflict), apply the FFHQ alignment quad from
its 5 landmarks, and drop frames with no detectable face.
"""
import os
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis

IN_DIR = sys.argv[1] if len(sys.argv) > 1 else "celebvhq_frames"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "aligned_frames"
RES = int(sys.argv[3]) if len(sys.argv) > 3 else 256
os.makedirs(OUT_DIR, exist_ok=True)


def ffhq_align(img, kps, out_size):
    """FFHQ-style alignment from insightface 5 pts:
    [left_eye, right_eye, nose, left_mouth, right_mouth]."""
    eye_left, eye_right = kps[0], kps[1]
    mouth_left, mouth_right = kps[3], kps[4]
    eye_avg = (eye_left + eye_right) * 0.5
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_eye = eye_right - eye_left
    eye_to_mouth = mouth_avg - eye_avg

    # build an oriented crop quad (FFHQ recipe)
    x = eye_to_eye.copy()
    x /= (np.hypot(*x) + 1e-8)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.array([-x[1], x[0]])
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y]).astype(np.float32)

    dst = np.array([[0, 0], [0, out_size], [out_size, out_size], [out_size, 0]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (out_size, out_size),
                               flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


def main():
    app = FaceAnalysis(allowed_modules=["detection"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(512, 512))

    # quality thresholds (keep clean, frontal, large faces -> closer to eval set)
    MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.70"))
    MIN_IOD = float(os.environ.get("MIN_IOD", "45"))      # inter-ocular dist (px, orig)
    MAX_ASYM = float(os.environ.get("MAX_ASYM", "0.35"))  # frontality (nose asymmetry)

    files = sorted(f for f in os.listdir(IN_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    saved = noface = lowscore = small = profile = 0
    for i, f in enumerate(files):
        img = cv2.imread(os.path.join(IN_DIR, f))
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            noface += 1
            continue
        faces.sort(key=lambda fc: (fc.bbox[2] - fc.bbox[0]) * (fc.bbox[3] - fc.bbox[1]),
                   reverse=True)
        fc = faces[0]
        kps = fc.kps.astype(np.float32)
        if float(getattr(fc, "det_score", 1.0)) < MIN_SCORE:
            lowscore += 1
            continue
        eye_l, eye_r, nose = kps[0], kps[1], kps[2]
        iod = np.hypot(*(eye_r - eye_l))
        if iod < MIN_IOD:
            small += 1
            continue
        # frontality: nose should be roughly equidistant from both eyes
        dl, dr = np.hypot(*(nose - eye_l)), np.hypot(*(nose - eye_r))
        if abs(dl - dr) / (iod + 1e-6) > MAX_ASYM:
            profile += 1
            continue
        try:
            crop = ffhq_align(img, kps, RES)
        except Exception:
            noface += 1
            continue
        cv2.imwrite(os.path.join(OUT_DIR, f"al_{saved:06d}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
        if (i + 1) % 500 == 0:
            print(f"processed {i+1}/{len(files)}  saved {saved}  "
                  f"(noface {noface} lowscore {lowscore} small {small} profile {profile})",
                  flush=True)
    print(f"DONE: saved {saved} aligned faces out of {len(files)} | "
          f"dropped: noface {noface}, lowscore {lowscore}, small {small}, profile {profile}")


if __name__ == "__main__":
    main()
