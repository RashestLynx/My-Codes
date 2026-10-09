"""Aligned DVD / Blu-ray training pairs for fine-tuning the upscaler.

usage:
  python make_pairs.py --dvd movie_dvd.mkv --bluray movie_bd.mkv --out pairs --count 3000

For frames spread over the movie it writes
  pairs/lr/000001.png   the DVD frame, exactly as dvd_upscale.py hands it to the model
                        (square pixels, deinterlaced, the same hqdn3d denoise)
  pairs/hr/000001.png   the matching Blu-ray frame, 2x the size, lined up to the pixel
  pairs/pairs.csv       per pair: DVD time, Blu-ray time, alignment score, sharpness
Frames that don't line up well, or have nothing in them (black, fades), are dropped.

How the two movies are lined up:
  1. black bars are found on both (ffmpeg cropdetect) and cut off, so only the picture is compared
  2. the time offset between the discs is measured every --anchor-every seconds by comparing
     tiny thumbnails. It may change along the movie (a longer logo, an extra scene on one disc):
     each sample uses the offset of the nearest anchors. (--offset sets one offset by hand;
     --speed 1.0427 for a PAL disc, which runs 4% fast, against a 24 fps Blu-ray)
  3. for each sample the Blu-ray frame that looks most like the DVD frame is picked from the
     frames within --window seconds of the expected time (thumbnail comparison)
  4. a small affine warp (cv2 ECC) removes the leftover shift/scale of the two transfers
  5. the Blu-ray's slow colour/brightness differences are matched to the DVD's, so the model
     learns detail, not the other disc's colour grade (--color-match none to turn that off)

needs: ffmpeg + ffprobe on the PATH, pip install numpy opencv-python
"""
import argparse, csv, json, os, random, re, subprocess, sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
import numpy as np
import cv2

THUMB = (64, 36)


def run(cmd, binary=False):
    p = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL)
    if p.returncode:
        sys.exit(f"ffmpeg failed: {' '.join(map(str, cmd))}\n{p.stderr.decode(errors='replace')[-800:]}")
    return p if binary else p.stdout.decode(errors="replace") + p.stderr.decode(errors="replace")


def probe(path):
    o = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,sample_aspect_ratio,r_frame_rate:format=duration",
                        "-of", "json", str(path)]))
    s = o["streams"][0]
    try:
        fps = float(Fraction(s.get("r_frame_rate", "24000/1001")))
    except (ZeroDivisionError, ValueError):
        fps = 24000 / 1001
    return s["width"], s["height"], s.get("sample_aspect_ratio", "1:1"), float(o["format"]["duration"]), fps


def dvd_chain(a, width, height, sar):
    """The filters dvd_upscale.py applies before the upscaler, for one frame."""
    f = ["yadif=deint=interlaced"]                  # (only frames the decoder flags as combed)
    if a.dar:
        n, d = (float(x) for x in a.dar.replace("/", ":").split(":"))
        f.append(f"scale=trunc(ih*{n / d}/2)*2:ih:flags=lanczos")
    else:
        sn, sd = (float(x) for x in (sar if ":" in sar and sar != "0:1" else "1:1").split(":"))
        f.append(f"scale=trunc(iw*{sn / sd}/2)*2:ih:flags=lanczos")
    f += ["setsar=1", f"hqdn3d={a.denoise}"]
    return f


def find_crop(path, pre, t0, secs=90):
    """Most common cropdetect result on `secs` seconds of the movie (the black bars)."""
    log = run(["ffmpeg", "-hide_banner", "-ss", f"{t0:.2f}", "-t", str(secs), "-i", str(path),
               "-vf", ",".join(pre + ["cropdetect=limit=24:round=2:reset=0"]), "-f", "null", "-"])
    seen = Counter(re.findall(r"crop=(\d+:\d+:\d+:\d+)", log))
    if not seen:
        return None
    return tuple(int(v) for v in seen.most_common(1)[0][0].split(":"))      # w, h, x, y


def read_frames(cmd, w, h, channels):
    raw = run(cmd, binary=True).stdout
    n = len(raw) // (w * h * channels)
    return np.frombuffer(raw[:n * w * h * channels], np.uint8).reshape(n, h, w, channels) if n else None


def thumbs(path, t0, secs, rate, vf):
    """Tiny grey frames: `rate` per second, or every frame when rate is None."""
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(t0, 0):.3f}", "-t", f"{secs:.3f}", "-i", str(path),
           "-vf", ",".join(vf + ([f"fps={rate}"] if rate else []) + [f"scale={THUMB[0]}:{THUMB[1]}:flags=area", "format=gray"]),
           "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
    return read_frames(cmd, THUMB[0], THUMB[1], 1)


def norm(x):
    x = x.astype(np.float32).reshape(len(x), -1)
    x -= x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)


def anchor_offset(a, t, guess, span, dvd_pre, bd_pre, win=30, rate=2):
    """Blu-ray time minus DVD-time*speed around DVD time t, and how well the thumbnails match."""
    d = thumbs(a.dvd, t, win, rate, dvd_pre)
    start = max(t * a.speed + guess - span, 0)
    b = thumbs(a.bluray, start, win * a.speed + 2 * span, rate, bd_pre)
    if d is None or b is None or len(b) < len(d):
        return None, 0.0
    nd, nb = norm(d), norm(b)
    scores = [float((nd * nb[k:k + len(nd)]).sum(1).mean()) for k in range(len(nb) - len(nd) + 1)]
    k = int(np.argmax(scores))
    return start + k / rate - t * a.speed, scores[k]


def build_offsets(a, dvd_pre, bd_pre, ddur):
    """[(dvd time, offset seconds)] along the movie. The offset may change (extra scenes)."""
    if a.offset is not None:
        return [(0.0, a.offset)]
    anchors, guess, span = [], 0.0, a.search
    t, end = ddur * a.skip, ddur * (1 - a.skip) - 30
    n_total = max(int((end - t) / a.anchor_every) + 1, 1)
    i = 0
    while t < end:
        i += 1
        off, sc = anchor_offset(a, t, guess, span, dvd_pre, bd_pre)
        ok = off is not None and sc >= a.min_anchor
        print(f"  offset {i}/{n_total} at {t / 60:5.1f} min: "
              + (f"Blu-ray {off:+7.2f} s, match {sc:.3f}" if ok else f"no clear match ({sc:.2f})"), flush=True)
        if ok:
            anchors.append((t, off)); guess, span = off, 40
        else:
            span = min(span * 2, a.search)              # lost: look wider next time
        t += a.anchor_every
    if len(anchors) < 3:
        sys.exit("can't follow the offset between the discs (too few clear matches): the discs may be "
                 "different cuts. Try --offset, --search, or lower --min-anchor.")
    return anchors


def offset_candidates(anchors, t):
    """The offsets of the nearest anchors (nearest first), at most two different ones."""
    out = []
    for _, o in sorted(anchors, key=lambda x: abs(x[0] - t)):
        if all(abs(o - p) > 0.75 for p in out):
            out.append(o)
        if len(out) == 2:
            break
    return out


def ecc_align(lr, hr_small):
    """Warp for hr_small (BGR) so it sits on lr (BGR), both the same size. -> warp, score."""
    g1 = cv2.GaussianBlur(cv2.cvtColor(lr, cv2.COLOR_BGR2GRAY), (0, 0), 1.2).astype(np.float32)
    g2 = cv2.GaussianBlur(cv2.cvtColor(hr_small, cv2.COLOR_BGR2GRAY), (0, 0), 1.2).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(g1, g2, warp, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
                                        None, 5)
    except cv2.error:
        return None, 0.0
    return warp, float(cc)


def low_freq_match(hr, lr, sigma=24):
    """Move the Blu-ray frame's slow colour/brightness onto the DVD's; detail untouched."""
    h, w = hr.shape[:2]
    lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    hr_f = hr.astype(np.float32)
    diff = cv2.GaussianBlur(lr_up, (0, 0), sigma * 2) - cv2.GaussianBlur(hr_f, (0, 0), sigma * 2)
    return np.clip(hr_f + diff, 0, 255).astype(np.uint8)


def make_pair(a, dvd_t, dvd_crop, bd_crop, dvd_pre, lrw, lrh, anchors, bd_fps):
    """-> (lr, hr, info) or (None, None, reason)."""
    c = lambda cr: [f"crop={cr[0]}:{cr[1]}:{cr[2]}:{cr[3]}"] if cr else []
    lr = read_frames(["ffmpeg", "-v", "error", "-ss", f"{dvd_t:.3f}", "-i", str(a.dvd), "-frames:v", "1",
                      "-vf", ",".join(dvd_pre + c(dvd_crop) + [f"scale={lrw}:{lrh}:flags=lanczos", "format=bgr24"]),
                      "-f", "rawvideo", "-"], lrw, lrh, 3)
    if lr is None:
        return None, None, "no dvd frame"
    lr = lr[0]
    gray = cv2.cvtColor(lr, cv2.COLOR_BGR2GRAY)
    if lr.std() < 12 or lr.mean() < 20:
        return None, None, "flat or dark"
    if cv2.Laplacian(gray, cv2.CV_32F).var() < 20:
        return None, None, "no detail"
    dthumb = norm(cv2.resize(gray, THUMB, interpolation=cv2.INTER_AREA)[None])

    # the Blu-ray frame that looks most like this one, among those near the expected time
    # (offsets of the nearest anchors: the nearest first)
    best = None
    for off in offset_candidates(anchors, dvd_t):
        ss = max(dvd_t * a.speed + off - a.window, 0)
        th = thumbs(a.bluray, ss, 2 * a.window, None, c(bd_crop))
        if th is None:
            continue
        sc = (norm(th) @ dthumb.T)[:, 0]
        k = int(np.argmax(sc))
        if best is None or sc[k] > best[0]:
            best = (float(sc[k]), ss, k)
        if sc[k] >= a.min_thumb:
            break
    if best is None or best[0] < a.min_thumb:
        return None, None, "thumb"
    _, ss, k = best
    hr = read_frames(["ffmpeg", "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{2 * a.window:.3f}", "-i", str(a.bluray),
                      "-vf", ",".join(c(bd_crop) + [f"select=eq(n\\,{k})", f"scale={2 * lrw}:{2 * lrh}:flags=lanczos", "format=bgr24"]),
                      "-fps_mode", "passthrough", "-frames:v", "1", "-f", "rawvideo", "-"], 2 * lrw, 2 * lrh, 3)
    if hr is None:
        return None, None, "no blu-ray frame"
    hr = hr[0]
    small = cv2.resize(hr, (lrw, lrh), interpolation=cv2.INTER_AREA)
    warp, cc = ecc_align(lr, small)
    if warp is None or cc < a.min_ecc:
        return None, None, f"ecc {cc:.2f}"
    if np.abs(warp[:, :2] - np.eye(2)).max() > a.max_warp:
        return None, None, "warp too large"
    # (translation of the warp is in LR pixels: x2 for the HR image; the 2x2 part is unchanged)
    hr = cv2.warpAffine(hr, warp * np.array([[1, 1, 2], [1, 1, 2]], np.float32), (2 * lrw, 2 * lrh),
                        flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    if a.color_match == "low":
        hr = low_freq_match(hr, lr)
    # final check: the Blu-ray frame shrunk to DVD size must look like the DVD frame
    g = lambda x: cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (0, 0), 2).astype(np.float32)
    ref = g(lr)
    shrunk = cv2.resize(hr, (lrw, lrh), interpolation=cv2.INTER_AREA)
    ncc = float(cv2.matchTemplate(g(shrunk)[8:-8, 8:-8], ref[8:-8, 8:-8], cv2.TM_CCOEFF_NORMED)[0, 0])
    if ncc < a.min_ncc:
        return None, None, f"ncc {ncc:.2f}"
    return lr, hr, dict(t_dvd=round(dvd_t, 3), t_bd=round(ss + k / bd_fps, 3),
                        ecc=round(cc, 4), ncc=round(ncc, 4),
                        sharp_ratio=round(float(cv2.Laplacian(g(hr), cv2.CV_32F).var()
                                                / (cv2.Laplacian(ref, cv2.CV_32F).var() + 1e-6)), 2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dvd", required=True)
    p.add_argument("--bluray", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=3000, help="pairs wanted (default 3000)")
    p.add_argument("--denoise", default="2:1.5:6:5", help="hqdn3d, as dvd_upscale.py's live preset")
    p.add_argument("--dar", default=None, help="force the DVD picture aspect, e.g. 16:9 (as dvd_upscale.py --dar)")
    p.add_argument("--offset", type=float, default=None, help="one fixed Blu-ray time minus DVD time, seconds (default: tracked along the movie)")
    p.add_argument("--speed", type=float, default=1.0, help="Blu-ray seconds per DVD second (PAL disc: 1.0427)")
    p.add_argument("--search", type=float, default=120, help="seconds either side searched for the first offset")
    p.add_argument("--anchor-every", type=float, default=120, help="seconds between offset measurements")
    p.add_argument("--min-anchor", type=float, default=0.6, help="thumbnail match an offset measurement needs")
    p.add_argument("--window", type=float, default=1.5, help="seconds either side of the expected time searched for the matching Blu-ray frame")
    p.add_argument("--min-thumb", type=float, default=0.85, help="thumbnail match a Blu-ray frame needs")
    p.add_argument("--skip", type=float, default=0.05, help="fraction of the movie skipped at each end")
    p.add_argument("--color-match", choices=["low", "none"], default="low")
    p.add_argument("--min-ecc", type=float, default=0.90)
    p.add_argument("--min-ncc", type=float, default=0.93)
    p.add_argument("--max-warp", type=float, default=0.05, help="largest scale/shear the alignment may apply")
    p.add_argument("--workers", type=int, default=3, help="frames made at the same time (default 3)")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    out = Path(a.out)
    (out / "lr").mkdir(parents=True, exist_ok=True)
    (out / "hr").mkdir(exist_ok=True)
    dw, dh, dsar, ddur, _ = probe(a.dvd)
    bw, bh, _, bdur, bd_fps = probe(a.bluray)
    print(f"DVD {dw}x{dh} sar {dsar}, {ddur / 60:.1f} min;  Blu-ray {bw}x{bh}, {bdur / 60:.1f} min")
    dvd_pre = dvd_chain(a, dw, dh, dsar)
    t_probe = ddur * 0.3
    dvd_crop = find_crop(a.dvd, dvd_pre, t_probe)
    bd_crop = find_crop(a.bluray, [], t_probe * a.speed)
    print(f"picture area: DVD (after square pixels) {dvd_crop}, Blu-ray {bd_crop}")
    if not dvd_crop:
        sys.exit("cropdetect found nothing on the DVD")
    lrw, lrh = dvd_crop[0], dvd_crop[1]
    if bd_crop and abs(bd_crop[0] / bd_crop[1] / (lrw / lrh) - 1) > 0.02:
        print(f"WARNING: picture aspect differs: DVD {lrw / lrh:.3f}, Blu-ray {bd_crop[0] / bd_crop[1]:.3f} "
              "(a re-framed transfer: pairs will be rejected or distorted)")
    lrw -= lrw % 2
    lrh -= lrh % 2

    cropped_bd_pre = ([f"crop={bd_crop[0]}:{bd_crop[1]}:{bd_crop[2]}:{bd_crop[3]}"] if bd_crop else [])
    cropped_dvd_pre = dvd_pre + [f"crop={dvd_crop[0]}:{dvd_crop[1]}:{dvd_crop[2]}:{dvd_crop[3]}"]
    print("measuring the offset between the discs along the movie (it can change: extra logos, scenes)...")
    anchors = build_offsets(a, cropped_dvd_pre, cropped_bd_pre, ddur)
    offs = [o for _, o in anchors]
    print(f"offsets found: {min(offs):+.1f} to {max(offs):+.1f} s ({len(anchors)} measurements)")

    rng = random.Random(a.seed)
    lo, hi = ddur * a.skip, ddur * (1 - a.skip)
    # more candidates than wanted: some are rejected. Jittered, evenly spread over the movie
    n_try = int(a.count * 1.6)
    times = [lo + (hi - lo) * (i + rng.random()) / n_try for i in range(n_try)]
    rng.shuffle(times)
    done, rejects = 0, Counter()
    work = lambda t: make_pair(a, t, dvd_crop, bd_crop, dvd_pre, lrw, lrh, anchors, bd_fps)
    with open(out / "pairs.csv", "w", newline="") as fh, ThreadPoolExecutor(max(1, a.workers)) as ex:
        wr = csv.writer(fh)
        wr.writerow(["name", "t_dvd", "t_bd", "ecc", "ncc", "sharp_ratio"])
        futs = [ex.submit(work, t) for t in times]
        try:
            for f in as_completed(futs):
                lr, hr, info = f.result()
                if lr is None:
                    rejects[info.split()[0]] += 1
                    continue
                done += 1
                name = f"{done:06d}"
                cv2.imwrite(str(out / "lr" / f"{name}.png"), lr)
                cv2.imwrite(str(out / "hr" / f"{name}.png"), hr)
                wr.writerow([name, info["t_dvd"], info["t_bd"], info["ecc"], info["ncc"], info["sharp_ratio"]])
                fh.flush()
                if done % 25 == 0:
                    print(f"  {done}/{a.count} pairs  (rejected: {dict(rejects)})", flush=True)
                if done >= a.count:
                    break
        finally:
            for f in futs:
                f.cancel()
    print(f"done: {done} pairs in {out}  (rejected: {dict(rejects)})")
    if done < a.count // 2:
        print("few pairs survived: check the picture areas above, or lower --min-thumb/--min-ecc/--min-ncc")


if __name__ == "__main__":
    main()
