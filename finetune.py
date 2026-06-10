"""Fine-tune a pretrained CelebA-HQ-256 DDPM UNet on CelebV-HQ frames.

Start from google/ddpm-ema-celebahq-256 (unconditional UNet2DModel, pure PyTorch,
no custom CUDA ops -> Blackwell-safe). Standard DDPM eps-prediction loss, bf16,
EMA weights, periodic checkpoints. Designed for a single 16GB GPU.
"""
import argparse
import copy
import os
import random
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.training_utils import EMAModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="celebvhq_frames")
    p.add_argument("--model", default="google/ddpm-ema-celebahq-256",
                   help="UNet weights source (base id or a resume checkpoint dir)")
    p.add_argument("--sched_src", default="google/ddpm-ema-celebahq-256",
                   help="scheduler config source (always the base model)")
    p.add_argument("--out", default="ft_ckpt")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--bs", type=int, default=12)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--start_step", type=int, default=0,
                   help="offset for checkpoint naming when resuming")
    p.add_argument("--gpu_duty", type=float, default=1.0,
                   help="target GPU duty cycle in (0,1]; <1 sleeps between steps")
    return p.parse_args()


class FaceFolder(Dataset):
    def __init__(self, root, res):
        self.paths = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.paths.sort()
        self.tf = transforms.Compose(
            [
                transforms.Resize(res, antialias=True),
                transforms.CenterCrop(res),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # -> [-1, 1]
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda"

    ds = FaceFolder(args.data, args.res)
    print(f"dataset size: {len(ds)}")
    dl = DataLoader(
        ds, batch_size=args.bs, shuffle=True, num_workers=8,
        drop_last=True, pin_memory=True, persistent_workers=True,
    )

    unet = UNet2DModel.from_pretrained(args.model).to(device)
    if args.grad_ckpt:
        unet.enable_gradient_checkpointing()
    sched = DDPMScheduler.from_pretrained(args.sched_src)
    ema = EMAModel(unet.parameters(), decay=args.ema_decay)

    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    unet.train()
    step = 0
    total = args.start_step + args.steps
    data_iter = iter(dl)
    while step < args.steps:
        t0 = time.time()
        try:
            x = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            x = next(data_iter)
        x = x.to(device, non_blocking=True)
        noise = torch.randn_like(x)
        bsz = x.shape[0]
        t = torch.randint(0, sched.config.num_train_timesteps, (bsz,), device=device).long()
        noisy = sched.add_noise(x, noise, t)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = unet(noisy, t).sample
            loss = F.mse_loss(pred.float(), noise.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        ema.step(unet.parameters())
        step += 1
        gstep = args.start_step + step
        # GPU duty-cycle throttle: leave the card idle a fraction of each step
        if args.gpu_duty < 1.0:
            torch.cuda.synchronize()
            work = time.time() - t0
            time.sleep(work * (1.0 / args.gpu_duty - 1.0))
        if step % 50 == 0:
            print(f"step {gstep}/{total}  loss {loss.item():.4f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            ckpt = os.path.join(args.out, f"unet_step{gstep}")
            unet.save_pretrained(ckpt)
            # save EMA copy (store online weights, swap in EMA, save, restore)
            ema.store(unet.parameters())
            ema.copy_to(unet.parameters())
            unet.save_pretrained(ckpt + "_ema")
            ema.restore(unet.parameters())
            print(f"saved {ckpt} (+_ema)", flush=True)
    print("FINETUNE DONE")


if __name__ == "__main__":
    main()
