"""Fine-tune Real-ESRGAN x2plus on aligned DVD / Blu-ray pairs (from make_pairs.py).

usage:
  python train.py --pairs pairs --out run1 --pretrained RealESRGAN_x2plus.pth

  RealESRGAN_x2plus.pth:
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

Made for a 6 GB laptop GPU (RTX 3060): fp16 autocast, 96 px patches, batches of 8. If it runs out
of memory: --batch 4, or --checkpoint (slower, much less memory), or --patch 64.

Stage 1 (default): L1 loss only. The model learns "what the Blu-ray shows where the DVD shows this"
with nothing invented; the picture gets truer, not more dramatic. Do this first.
Stage 2 (optional): --perceptual 0.5 --gan 0.05 adds texture, at the risk of drawing things that
aren't there. Start it from stage 1:  --pretrained run1/dvd2bd_latest.pth --out run2

Writes in --out:  dvd2bd_latest.pth, dvd2bd_best.pth (best on the held-out frames), train_state.pt
(training resumes by itself when you run the same command again), and a held-out report that
compares your model against the untouched pretrained one.
Held-out frames: every 20th pair, never trained on.

needs: pip install torch torchvision numpy opencv-python   (torch with CUDA, for the GPU)
"""
import argparse, copy, math, random, time
from pathlib import Path
import numpy as np
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---- the network: the same layout (and weight names) as the official RealESRGAN_x2plus.pth ----
class RDB(nn.Module):
    def __init__(s, nf=64, gc=32):
        super().__init__()
        s.conv1 = nn.Conv2d(nf, gc, 3, 1, 1); s.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        s.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1); s.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        s.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1); s.lrelu = nn.LeakyReLU(0.2)

    def forward(s, x):
        x1 = s.lrelu(s.conv1(x)); x2 = s.lrelu(s.conv2(torch.cat((x, x1), 1)))
        x3 = s.lrelu(s.conv3(torch.cat((x, x1, x2), 1)))
        x4 = s.lrelu(s.conv4(torch.cat((x, x1, x2, x3), 1)))
        return s.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(s, nf, gc=32):
        super().__init__(); s.rdb1, s.rdb2, s.rdb3 = RDB(nf, gc), RDB(nf, gc), RDB(nf, gc)

    def forward(s, x):
        return s.rdb3(s.rdb2(s.rdb1(x))) * 0.2 + x


class RRDBNetX2(nn.Module):
    def __init__(s, nf=64, nb=23, gc=32):
        super().__init__()
        s.conv_first = nn.Conv2d(3 * 4, nf, 3, 1, 1)
        s.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        s.conv_body = nn.Conv2d(nf, nf, 3, 1, 1); s.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        s.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1); s.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        s.conv_last = nn.Conv2d(nf, 3, 3, 1, 1); s.lrelu = nn.LeakyReLU(0.2)
        s.use_checkpoint = False

    def forward(s, x):
        feat = s.conv_first(F.pixel_unshuffle(x, 2))
        body = feat
        for blk in s.body:
            body = checkpoint(blk, body, use_reentrant=False) if s.use_checkpoint and s.training else blk(body)
        feat = feat + s.conv_body(body)
        feat = s.lrelu(s.conv_up1(F.interpolate(feat, scale_factor=2.0, mode="nearest")))
        feat = s.lrelu(s.conv_up2(F.interpolate(feat, scale_factor=2.0, mode="nearest")))
        return s.conv_last(s.lrelu(s.conv_hr(feat)))


def load_weights(net, path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(sd.get("params_ema", sd.get("params", sd)), strict=True)


# ---- data ----
class Pairs(torch.utils.data.Dataset):
    """Each item: `crops` random patches of one random pair -> (lr [n,3,p,p], hr [n,3,2p,2p])."""

    def __init__(s, root, names, patch, crops):
        s.root, s.names, s.p, s.n = Path(root), names, patch, crops

    def __len__(s):
        return 10 ** 7

    def __getitem__(s, _):
        name = random.choice(s.names)
        lr = cv2.imread(str(s.root / "lr" / f"{name}.png"))
        hr = cv2.imread(str(s.root / "hr" / f"{name}.png"))
        h, w = lr.shape[:2]
        p, lrs, hrs = s.p, [], []
        for _ in range(s.n):
            best = None
            for _ in range(3):             # of 3 random spots, the one with the most going on
                y, x = random.randint(0, h - p), random.randint(0, w - p)
                v = lr[y:y + p, x:x + p].std()
                if best is None or v > best[0]:
                    best = (v, y, x)
            _, y, x = best
            a, b = lr[y:y + p, x:x + p], hr[2 * y:2 * (y + p), 2 * x:2 * (x + p)]
            if random.random() < 0.5:
                a, b = a[:, ::-1], b[:, ::-1]
            lrs.append(a); hrs.append(b)
        t = lambda L: torch.from_numpy(np.ascontiguousarray(np.stack(L)[..., ::-1])).permute(0, 3, 1, 2).float() / 255
        return t(lrs), t(hrs)


def collate(items):
    return torch.cat([i[0] for i in items]), torch.cat([i[1] for i in items])


def load_frame(root, name, crop):
    lr = cv2.imread(str(Path(root) / "lr" / f"{name}.png"))
    hr = cv2.imread(str(Path(root) / "hr" / f"{name}.png"))
    h, w = lr.shape[:2]
    c = min(crop, h - h % 2, w - w % 2)
    y, x = (h - c) // 2, (w - c) // 2
    t = lambda im: torch.from_numpy(np.ascontiguousarray(im[..., ::-1])).permute(2, 0, 1).float()[None] / 255
    return t(lr[y:y + c, x:x + c]), t(hr[2 * y:2 * (y + c), 2 * x:2 * (x + c)])


# ---- losses ----
class VGGLoss(nn.Module):
    """Real-ESRGAN's perceptual loss: VGG19 conv features (weights download on first use)."""
    LAYERS = {2: 0.1, 7: 0.1, 16: 1.0, 25: 1.0, 34: 1.0}       # conv1_2 .. conv5_4

    def __init__(s):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        s.vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:35].eval()
        for q in s.vgg.parameters():
            q.requires_grad = False
        s.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        s.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def feats(s, x):
        x, out = (x - s.mean) / s.std, {}
        for i, layer in enumerate(s.vgg):
            x = layer(x)
            if i in s.LAYERS:
                out[i] = x
        return out

    def forward(s, pred, target):
        fp, ft = s.feats(pred), s.feats(target.detach())
        return sum(w * F.l1_loss(fp[i], ft[i]) for i, w in s.LAYERS.items())


def discriminator():
    sn = nn.utils.spectral_norm
    c = lambda i, o, k, st: sn(nn.Conv2d(i, o, k, st, k // 2))
    return nn.Sequential(c(3, 64, 3, 1), nn.LeakyReLU(0.2), c(64, 64, 4, 2), nn.LeakyReLU(0.2),
                         c(64, 128, 3, 1), nn.LeakyReLU(0.2), c(128, 128, 4, 2), nn.LeakyReLU(0.2),
                         c(128, 256, 3, 1), nn.LeakyReLU(0.2), c(256, 256, 4, 2), nn.LeakyReLU(0.2),
                         c(256, 1, 3, 1))                        # PatchGAN: a score per patch


# ---- evaluation ----
@torch.no_grad()
def psnr_on(net, dev, root, names, crop):
    net.eval()
    vals = []
    for n in names:
        lr, hr = load_frame(root, n, crop)
        with torch.autocast(dev.type, torch.float16, enabled=dev.type == "cuda"):
            out = net(lr.to(dev))
        mse = F.mse_loss(out.float().clamp(0, 1), hr.to(dev)).item()
        vals.append(-10 * math.log10(max(mse, 1e-10)))
    return float(np.mean(vals))


@torch.no_grad()
def psnr_plain(root, names, crop):
    """PSNR of a plain bicubic 2x of the DVD frame: the floor any model must beat."""
    vals = []
    for n in names:
        lr, hr = load_frame(root, n, crop)
        up = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
        vals.append(-10 * math.log10(max(F.mse_loss(up, hr).item(), 1e-10)))
    return float(np.mean(vals))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True, help="folder made by make_pairs.py")
    p.add_argument("--out", required=True)
    p.add_argument("--pretrained", required=True, help="RealESRGAN_x2plus.pth (or 'none' for a test run)")
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8, help="patches per step (default 8)")
    p.add_argument("--patch", type=int, default=96, help="DVD patch size in pixels (default 96)")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--perceptual", type=float, default=0.0)
    p.add_argument("--gan", type=float, default=0.0)
    p.add_argument("--checkpoint", action="store_true", help="trade speed for much less GPU memory")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--val-crop", type=int, default=384, help="held-out frames are judged on this centre square (DVD px)")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        print("WARNING: no CUDA GPU found, training on the processor (very slow)")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.pairs)
    names = sorted(f.stem for f in (root / "lr").glob("*.png") if (root / "hr" / f.name).exists())
    val = names[::20]
    train = [n for n in names if n not in set(val)]
    if len(train) < 8:
        raise SystemExit(f"only {len(names)} pairs in {root}: make more with make_pairs.py")
    print(f"{len(train)} training pairs, {len(val)} held out; device {dev}")

    net = RRDBNetX2().to(dev)
    if a.pretrained != "none":
        load_weights(net, a.pretrained)
    else:
        print("WARNING: random start (--pretrained none): only for testing the scripts")
    net.use_checkpoint = a.checkpoint
    ema = copy.deepcopy(net).eval()
    for q in ema.parameters():
        q.requires_grad = False
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    vgg = VGGLoss().to(dev) if a.perceptual > 0 else None
    disc = discriminator().to(dev) if a.gan > 0 else None
    d_opt = torch.optim.Adam(disc.parameters(), lr=a.lr, betas=(0.9, 0.99)) if disc else None
    d_scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    step, best = 0, -1e9
    state_path = out / "train_state.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location=dev, weights_only=False)   # (our own file)
        net.load_state_dict(st["net"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        if disc and "disc" in st:
            disc.load_state_dict(st["disc"]); d_opt.load_state_dict(st["d_opt"])
        step, best = st["step"], st["best"]
        print(f"resuming at step {step}")

    # where we start from, on frames never trained on
    floor = psnr_plain(root, val, a.val_crop)
    base = psnr_on(ema, dev, root, val, a.val_crop) if step == 0 else None
    if base is not None:
        print(f"held-out PSNR: plain bicubic {floor:.2f} dB, starting model {base:.2f} dB", flush=True)

    per = 4
    ds = Pairs(root, train, a.patch, per)
    dl = torch.utils.data.DataLoader(ds, batch_size=max(1, a.batch // per), num_workers=a.workers,
                                     collate_fn=collate, persistent_workers=a.workers > 0)
    it = iter(dl)
    t0, run = time.time(), {}
    net.train()
    while step < a.iters:
        lr_img, hr_img = (x.to(dev, non_blocking=True) for x in next(it))
        lr_now = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.iters)))     # cosine to 10%
        for g in opt.param_groups:
            g["lr"] = lr_now
        with torch.autocast(dev.type, torch.float16, enabled=dev.type == "cuda"):
            pred = net(lr_img)
            loss = F.l1_loss(pred, hr_img)
            logs = {"l1": loss.item()}
            if vgg:
                lp = vgg(pred.clamp(0, 1), hr_img)
                loss = loss + a.perceptual * lp; logs["vgg"] = lp.item()
            if disc:
                lg = F.softplus(-disc(pred)).mean()           # fool the discriminator
                loss = loss + a.gan * lg; logs["g"] = lg.item()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if disc:
            with torch.autocast(dev.type, torch.float16, enabled=dev.type == "cuda"):
                ld = F.softplus(-disc(hr_img)).mean() + F.softplus(disc(pred.detach())).mean()
            d_opt.zero_grad(set_to_none=True)
            d_scaler.scale(ld).backward(); d_scaler.step(d_opt); d_scaler.update()
            logs["d"] = ld.item()
        step += 1
        decay = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.mul_(decay).add_(pn.detach(), alpha=1 - decay)
        for k, v in logs.items():
            run[k] = run.get(k, 0) + v
        if step % 100 == 0:
            el = time.time() - t0
            print(f"step {step}/{a.iters}  " + "  ".join(f"{k} {v / 100:.4f}" for k, v in run.items())
                  + f"  lr {lr_now:.2e}  {el / 100:.2f} s/step", flush=True)
            run, t0 = {}, time.time()
        if step % a.save_every == 0 or step == a.iters:
            score = psnr_on(ema, dev, root, val, a.val_crop)
            net.train()
            note = ""
            if score > best:
                best = score; note = " (best)"
                torch.save({"params_ema": ema.state_dict()}, out / "dvd2bd_best.pth")
            torch.save({"params_ema": ema.state_dict()}, out / "dvd2bd_latest.pth")
            torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "step": step, "best": best,
                        **({"disc": disc.state_dict(), "d_opt": d_opt.state_dict()} if disc else {})}, state_path)
            print(f"== step {step}: held-out PSNR {score:.2f} dB{note}   (bicubic {floor:.2f}"
                  + (f", start {base:.2f}" if base is not None else "") + ")", flush=True)
    print(f"done. Convert with:  python export_ncnn.py {out / 'dvd2bd_latest.pth'} --name dvd2bd-x2")


if __name__ == "__main__":
    main()
