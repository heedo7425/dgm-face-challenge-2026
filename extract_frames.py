"""Stream SwayStar123/CelebV-HQ's videos.tar over HTTP and extract one frame
per mp4 clip as a 256x256 center-cropped face image. Stops after TARGET clips,
so it only downloads the prefix of the 42GB tar (~1MB/clip), not the whole thing.

Used as fine-tune data + local FID/KID reference set.
"""
import io
import os
import sys
import tarfile

import av
import requests
from PIL import Image

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "celebvhq_frames"
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
RES = int(sys.argv[3]) if len(sys.argv) > 3 else 256
URL = "https://huggingface.co/datasets/SwayStar123/CelebV-HQ/resolve/main/videos.tar"
os.makedirs(OUT_DIR, exist_ok=True)


def middle_frame_from_mp4(mp4_bytes):
    container = av.open(io.BytesIO(mp4_bytes))
    stream = container.streams.video[0]
    n = stream.frames or 0
    target_idx = n // 2 if n > 0 else 8
    img = None
    for i, frame in enumerate(container.decode(video=0)):
        img = frame.to_image()
        if i >= target_idx:
            break
        if i > 400:
            break
    container.close()
    return img


def center_square_resize(img, res):
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((res, res), Image.LANCZOS)


def main():
    existing = len([f for f in os.listdir(OUT_DIR) if f.endswith(".jpg")])
    if existing >= TARGET:
        print(f"Already have {existing} >= {TARGET}, done.")
        return
    saved, seen = existing, 0
    headers = {}
    tok = os.environ.get("HF_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    with requests.get(URL, stream=True, headers=headers, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        tar = tarfile.open(fileobj=r.raw, mode="r|")
        for member in tar:
            if not member.isfile():
                continue
            name = member.name.lower()
            if not name.endswith((".mp4", ".mov", ".webm", ".mkv")):
                continue
            seen += 1
            try:
                f = tar.extractfile(member)
                data = f.read()
                img = middle_frame_from_mp4(data)
                if img is None:
                    continue
                img = center_square_resize(img.convert("RGB"), RES)
                img.save(os.path.join(OUT_DIR, f"cvhq_{saved:06d}.jpg"), quality=95)
                saved += 1
            except Exception as e:
                if seen % 500 == 0:
                    print(f"  skip @{seen}: {e}", flush=True)
                continue
            if saved % 200 == 0:
                print(f"saved {saved}/{TARGET} (scanned {seen})", flush=True)
            if saved >= TARGET:
                break
    print(f"DONE: saved {saved} frames from {seen} clips into {OUT_DIR}")


if __name__ == "__main__":
    main()
