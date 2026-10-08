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
  2. the time offset between the discs is found by comparing tiny thumbnails (--offset sets it
     by hand; --speed 1.0427 for a PAL disc, which runs 4% fast, against a 24 fps Blu-ray)
  3. for each sample the Blu-ray frame closest to the DVD frame is chosen from a few neighbours
  4. a small affine warp (cv2 ECC) removes the leftover shift/scale of the two transfers
  5. the Blu-ray's slow colour/brightness differences are matched to the DVD's, so the model
     learns detail, not the other disc's colour grade (--color-match none to turn that off)

needs: ffmpeg + ffprobe on the PATH, pip install numpy opencv-python
"""
import argparse, csv, json, os, random, re, subprocess, sys
from collections import Counter
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
                        "stream=width,height,sample_aspect_ratio,display_aspect_ratio:format=duration",
                        "-of", "json", str(path)]))
    s = o["streams"][0]
    return s["width"], s["height"], s.get("sample_aspect_ratio", "1:1"), float(o["format"]["duration"])


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
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(t0, 0):.3f}", "-t", f"{secs:.3f}", "-i", str(path),
           "-vf", ",".join(vf + [f"fps={rate}", f"scale={THUMB[0]}:{THUMB[1]}:flags=area", "format=gray"]),
           "-f", "rawvideo", "-"]
    return read_frames(cmd, THUMB[0], THUMB[1], 1)


def norm(x):
    x = x.astype(np.float32).reshape(len(x), -1)
    x -= x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)


def find_offset(a, dvd, bd, dvd_pre, bd_pre, dvd_dur):
    """Seconds to add to a DVD time to get the Blu-ray time of the same picture."""
    rate, win = 2, 40
    best = None
    # three places in the movie: the offset is checked in each, and must agree
    for frac in (0.25, 0.5, 0.75):
        t = dvd_dur * frac
        d = thumbs(a.dvd, t, win, rate, dvd_pre)
        span = a.search
        b = thumbs(a.bluray, t * a.speed - span, win * a.speed + 2 * span, rate, bd_pre)
        if d is None or b is None or len(b) < len(d):
            continue
        nd, nb = norm(d), norm(b)
        scores = [float((nd * nb[k:k + len(nd)]).sum(1).mean()) for k in range(len(nb) - len(nd) + 1)]
        k = int(np.argmax(scores))
        best = best or []
        best.append((scores[k], t * a.speed - span + k / rate - t * a.speed))
        print(f"  offset check at {t / 60:.1f} min: Blu-ray is {best[-1][1]:+.2f} s, "
              f"match {scores[k]:.3f}", flush=True)
    if not best:
        sys.exit("can't find the offset: pass --offset by hand")
    good = [o for s, o in best if s > 0.5]
    if not good or max(good) - min(good) > 1.0:
        sys.exit("the three offset checks disagree or match badly: the discs may be different cuts, "
                 "or the picture crops differ. Try --offset, --speed or --search.")
    return float(np.median(good))


def ecc_align(lr, hr_small):
    """Warp for hr_small (BGR) so it sits on lr (BGR), both the same size. -> warped, score."""
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


def make_pair(a, dvd_t, dvd_crop, bd_crop, dvd_pre, lrw, lrh, bdw, bdh, bd_offset):
    """-> (lr, hr, info) or (None, None, reason)."""
    c = lambda cr: [f"crop={cr[0]}:{cr[1]}:{cr[2]}:{cr[3]}"] if cr else []
    lr = read_frames(["ffmpeg", "-v", "error", "-ss", f"{dvd_t:.3f}", "-i", str(a.dvd), "-frames:v", "1",
                      "-vf", ",".join(dvd_pre + c(dvd_crop) + [f"scale={lrw}:{lrh}:flags=lanczos", "format=bgr24"]),
                      "-f", "rawvideo", "-"], lrw, lrh, 3)
    if lr is None:
        return None, None, "no dvd frame"
    lr = lr[0]
    if lr.std() < 12 or lr.mean() < 20:
        return None, None, "flat or dark"
    if cv2.Laplacian(cv2.cvtColor(lr, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var() < 20:
        return None, None, "no detail"
    # a few Blu-ray frames around the expected time; the nearest picture wins
    bt = dvd_t * a.speed + bd_offset
    span = a.neighbours / 24.0
    cands = read_frames(["ffmpeg", "-v", "error", "-ss", f"{max(bt - span, 0):.3f}", "-t", f"{2 * span + 0.05:.3f}",
                         "-i", str(a.bluray), "-vf", ",".join(c(bd_crop) + [f"scale={2 * lrw}:{2 * lrh}:flags=lanczos", "format=bgr24"]),
                         "-f", "rawvideo", "-"], 2 * lrw, 2 * lrh, 3)
    if cands is None:
        return None, None, "no blu-ray frame"
    small = [cv2.resize(f, (lrw, lrh), interpolation=cv2.INTER_AREA) for f in cands]
    g = lambda x: cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (0, 0), 2).astype(np.float32)
    ref = g(lr)
    err = [float(np.abs(g(s) - ref).mean()) for s in small]
    k = int(np.argmin(err))
    warp, cc = ecc_align(lr, small[k])
    if warp is None or cc < a.min_ecc:
        return None, None, f"ecc {cc:.2f}"
    hr = cv2.warpAffine(cands[k], warp * np.array([[1, 1, 2], [1, 1, 2]], np.float32), (2 * lrw, 2 * lrh),
                        flags=cv2.INTER_LANCZOS4 + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    # (translation of the warp is in LR pixels: x2 for the HR image; the 2x2 part is unchanged)
    if np.abs(warp[:, :2] - np.eye(2)).max() > a.max_warp:
        return None, None, "warp too large"
    if a.color_match == "low":
        hr = low_freq_match(hr, lr)
    # final check: the Blu-ray frame shrunk to DVD size must look like the DVD frame
    shrunk = cv2.resize(hr, (lrw, lrh), interpolation=cv2.INTER_AREA)
    ncc = float(cv2.matchTemplate(g(shrunk)[8:-8, 8:-8], ref[8:-8, 8:-8], cv2.TM_CCOEFF_NORMED)[0, 0])
    if ncc < a.min_ncc:
        return None, None, f"ncc {ncc:.2f}"
    return lr, hr, dict(t_dvd=round(dvd_t, 3), t_bd=round(bt + (k - len(cands) // 2) / 24.0, 3),
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
    p.add_argument("--offset", type=float, default=None, help="Blu-ray time minus DVD time, seconds (default: found)")
    p.add_argument("--speed", type=float, default=1.0, help="Blu-ray seconds per DVD second (PAL disc: 1.0427)")
    p.add_argument("--search", type=float, default=120, help="seconds either side searched for the offset")
    p.add_argument("--neighbours", type=int, default=3, help="Blu-ray frames either side tried per sample")
    p.add_argument("--skip", type=float, default=0.05, help="fraction of the movie skipped at each end")
    p.add_argument("--color-match", choices=["low", "none"], default="low")
    p.add_argument("--min-ecc", type=float, default=0.90)
    p.add_argument("--min-ncc", type=float, default=0.93)
    p.add_argument("--max-warp", type=float, default=0.05, help="largest scale/shear the alignment may apply")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    out = Path(a.out)
    (out / "lr").mkdir(parents=True, exist_ok=True)
    (out / "hr").mkdir(exist_ok=True)
    dw, dh, dsar, ddur = probe(a.dvd)
    bw, bh, _, bdur = probe(a.bluray)
    print(f"DVD {dw}x{dh} sar {dsar}, {ddur / 60:.1f} min;  Blu-ray {bw}x{bh}, {bdur / 60:.1f} min")
    dvd_pre = dvd_chain(a, dw, dh, dsar)
    t_probe = ddur * 0.3
    dvd_crop = find_crop(a.dvd, dvd_pre, t_probe)
    bd_crop = find_crop(a.bluray, [], t_probe * a.speed)
    print(f"picture area: DVD (after square pixels) {dvd_crop}, Blu-ray {bd_crop}")
    # (the DVD crop is found on the filtered frame, so the crop filter goes after the filters)
    lrw, lrh = (dvd_crop[0], dvd_crop[1]) if dvd_crop else (None, None)
    if not dvd_crop:
        sys.exit("cropdetect found nothing on the DVD")
    if bd_crop and abs(bd_crop[0] / bd_crop[1] / (lrw / lrh) - 1) > 0.02:
        print(f"WARNING: picture aspect differs: DVD {lrw / lrh:.3f}, Blu-ray {bd_crop[0] / bd_crop[1]:.3f} "
              "(a re-framed transfer: pairs will be rejected or distorted)")
    lrw -= lrw % 2
    lrh -= lrh % 2

    bd_pre = []
    cropped_bd_pre = bd_pre + ([f"crop={bd_crop[0]}:{bd_crop[1]}:{bd_crop[2]}:{bd_crop[3]}"] if bd_crop else [])
    cropped_dvd_pre = dvd_pre + [f"crop={dvd_crop[0]}:{dvd_crop[1]}:{dvd_crop[2]}:{dvd_crop[3]}"]
    if a.offset is None:
        print("finding the offset between the discs...")
        a.offset = find_offset(a, a.dvd, a.bluray, cropped_dvd_pre, cropped_bd_pre, ddur)
    print(f"Blu-ray = DVD x {a.speed} {a.offset:+.2f} s")

    rng = random.Random(a.seed)
    lo, hi = ddur * a.skip, ddur * (1 - a.skip)
    # more candidates than wanted: some are rejected. Jittered, evenly spread over the movie
    n_try = int(a.count * 1.6)
    times = [lo + (hi - lo) * (i + rng.random()) / n_try for i in range(n_try)]
    rng.shuffle(times)
    done, rejects = 0, Counter()
    with open(out / "pairs.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["name", "t_dvd", "t_bd", "ecc", "ncc", "sharp_ratio"])
        for t in times:
            lr, hr, info = make_pair(a, t, dvd_crop, bd_crop, dvd_pre, lrw, lrh, bw, bh, a.offset)
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
    print(f"done: {done} pairs in {out}  (rejected: {dict(rejects)})")
    if done < a.count // 2:
        print("few pairs survived: check the offset, the picture areas above, or lower --min-ecc/--min-ncc")


if __name__ == "__main__":
    main()
