"""realesrgan-x2plus.param/.bin for realesrgan-ncnn-vulkan, made from the official
RealESRGAN_x2plus.pth (BSD-3-Clause, Xintao Wang):
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

usage: python make_x2plus_ncnn.py RealESRGAN_x2plus.pth OUT_DIR      (needs: pip install torch numpy)

The ncnn files are written directly (pnnx ran out of memory on this network): input blob "data",
output "output", RGB 0-1, as realesrgan-ncnn-vulkan expects. pixel_unshuffle(2) + the first 3x3
conv are folded into one 6x6 stride-2 conv (the same sums), so no special layer is needed.
Checked: ncnn on the CPU matches PyTorch at 82 dB PSNR (fp16 weights); in realesrgan-ncnn-vulkan
it differs from PyTorch as much as the official x4plus model does (46 dB inside, the frame
edges padded by the upscaler itself).
"""
import struct, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

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
    def forward(s, x):
        feat = s.conv_first(F.pixel_unshuffle(x, 2))
        feat = feat + s.conv_body(s.body(feat))
        feat = s.lrelu(s.conv_up1(F.interpolate(feat, scale_factor=2.0, mode="nearest")))
        feat = s.lrelu(s.conv_up2(F.interpolate(feat, scale_factor=2.0, mode="nearest")))
        return s.conv_last(s.lrelu(s.conv_hr(feat)))


pth, out_dir = sys.argv[1], sys.argv[2]
sd = torch.load(pth, map_location="cpu", weights_only=True)   # (weights only: no pickled code runs)
sd = sd.get("params_ema", sd.get("params", sd))
RRDBNetX2().load_state_dict(sd, strict=True)                  # the right file: every weight in place
ops, binbuf = [], bytearray()

def fp16_weights(w):
    a = w.detach().cpu().numpy().astype(np.float16).tobytes()
    a += b"\0" * (-len(a) % 4)
    return struct.pack("<I", 0x01306B47) + a

def conv(name, inp, outp, key, k=3, s=1, p=1, w=None, b=None):
    w = sd[key + ".weight"] if w is None else w
    b = sd[key + ".bias"] if b is None else b
    o = w.shape[0]
    ops.append(["Convolution", name, [inp], [outp],
                f"0={o} 1={k} 3={s} 4={p} 5=1 6={w.numel()}"])
    binbuf.extend(fp16_weights(w)); binbuf.extend(b.detach().numpy().astype("<f4").tobytes())

def relu(name, inp, outp):
    ops.append(["ReLU", name, [inp], [outp], "0=2.000000e-01"])

def cat(name, ins, outp):
    ops.append(["Concat", name, ins, [outp], "0=0"])

def mul(name, inp, outp, v):
    ops.append(["BinaryOp", name, [inp], [outp], f"0=2 1=1 2={v:e}"])

def add(name, a, b, outp):
    ops.append(["BinaryOp", name, [a, b], [outp], "0=0"])

def up2(name, inp, outp):
    ops.append(["Interp", name, [inp], [outp], "0=1 1=2.000000e+00 2=2.000000e+00"])

# first conv folded: W6[o, c, 2*ky+i, 2*kx+j] = W3[o, c*4 + i*2 + j, ky, kx]
w3 = sd["conv_first.weight"]
w6 = torch.zeros(w3.shape[0], 3, 6, 6)
for c in range(3):
    for i in range(2):
        for j in range(2):
            w6[:, c, i::2, j::2] = w3[:, c * 4 + i * 2 + j]
ops.append(["Input", "data", [], ["data"], ""])
conv("conv_first", "data", "feat", "conv_first", k=6, s=2, p=2, w=w6)

x = "feat"
for n in range(23):
    rin = x
    for r in (1, 2, 3):
        pre = f"body.{n}.rdb{r}"
        t = pre.replace(".", "_")
        conv(t + "_c1", x, t + "_a1", pre + ".conv1"); relu(t + "_r1", t + "_a1", t + "_x1")
        cat(t + "_cat2", [x, t + "_x1"], t + "_k2")
        conv(t + "_c2", t + "_k2", t + "_a2", pre + ".conv2"); relu(t + "_r2", t + "_a2", t + "_x2")
        cat(t + "_cat3", [x, t + "_x1", t + "_x2"], t + "_k3")
        conv(t + "_c3", t + "_k3", t + "_a3", pre + ".conv3"); relu(t + "_r3", t + "_a3", t + "_x3")
        cat(t + "_cat4", [x, t + "_x1", t + "_x2", t + "_x3"], t + "_k4")
        conv(t + "_c4", t + "_k4", t + "_a4", pre + ".conv4"); relu(t + "_r4", t + "_a4", t + "_x4")
        cat(t + "_cat5", [x, t + "_x1", t + "_x2", t + "_x3", t + "_x4"], t + "_k5")
        conv(t + "_c5", t + "_k5", t + "_x5", pre + ".conv5")
        mul(t + "_m", t + "_x5", t + "_x5s", 0.2)
        add(t + "_add", t + "_x5s", x, t + "_out")
        x = t + "_out"
    b = f"body_{n}"
    mul(b + "_m", x, b + "_s", 0.2)
    add(b + "_add", b + "_s", rin, b + "_out")
    x = b + "_out"
conv("conv_body", x, "bodyf", "conv_body")
add("trunk_add", "feat", "bodyf", "feat2")
up2("up1", "feat2", "u1"); conv("conv_up1", "u1", "u1c", "conv_up1"); relu("r_up1", "u1c", "u1r")
up2("up2", "u1r", "u2"); conv("conv_up2", "u2", "u2c", "conv_up2"); relu("r_up2", "u2c", "u2r")
conv("conv_hr", "u2r", "hrc", "conv_hr"); relu("r_hr", "hrc", "hrr")
conv("conv_last", "hrr", "output", "conv_last")

# ncnn: a blob read by several layers goes through a Split, one copy per reader
readers = {}
for li, op in enumerate(ops):
    for k, b in enumerate(op[2]):
        readers.setdefault(b, []).append((li, k))
final, made = [], 0
for li, op in enumerate(ops):
    final.append(op)
    for b in op[3]:
        rs = readers.get(b, [])
        if len(rs) > 1:
            names = [f"{b}_sp{m}" for m in range(len(rs))]
            final.append(["Split", f"split_{b}", [b], names, ""])
            for (rli, k), nm in zip(rs, names):
                ops[rli][2][k] = nm
blobs = set()
for op in final:
    blobs.update(op[2]); blobs.update(op[3])
lines = ["7767517", f"{len(final)} {len(blobs)}"]
for t, name, ins, outs, params in final:
    lines.append(f"{t:<16} {name:<24} {len(ins)} {len(outs)} " + " ".join(ins + outs)
                 + (" " + params if params else ""))
open(f"{out_dir}/realesrgan-x2plus.param", "w").write("\n".join(lines) + "\n")
open(f"{out_dir}/realesrgan-x2plus.bin", "wb").write(binbuf)
print(len(final), "layers,", len(blobs), "blobs,", len(binbuf), "bytes of weights")
