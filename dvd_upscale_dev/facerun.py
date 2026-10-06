"""Run dvd_upscale.py with and without --faces on a face test clip and compare the two.
usage: python3 facerun.py LAYOUT [STRENGTH] [script args...]   (LAYOUT: faces/<LAYOUT>/sd.mpg,
60 frames of 640x480 at 25 fps with a real face; made by make_test_media.sh faces)
Both runs: --cpu --type live --mode progressive --chunk-frames 20 --height 960 with the bicubic
stand-in upscaler. Prints one JSON line: exit codes, output frame counts, the mean absolute
difference (luma) between the runs inside the face box and outside it (the box tripled), per
frame, and the frame-to-frame change of the in-face difference at the chunk seams (19/20, 39/40)
against inside the chunks (a pulse at the seams would show as a larger change there), and the
same for the extra flicker in the face box (frame-to-frame change with --faces minus without).
REUSE=1: measure the outputs of the last runs again instead of running the script."""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
SCRIPT = str(HERE.parent / "dvd_upscale.py")
FAKE = str(HERE / "fakeesrgan" / "realesrgan-ncnn-vulkan")
MODELS = HERE / "faces" / "face_models"
layout, *rest = sys.argv[1:]
strength = rest.pop(0) if rest and not rest[0].startswith("-") else None
src = HERE / "faces" / layout / "sd.mpg"
d = HERE / "runs" / f"faces_{layout}"
REUSE = os.environ.get("REUSE") == "1" and (d / "with" / "out.mkv").exists()
if not REUSE:
    subprocess.run(["rm", "-rf", str(d)])
    d.mkdir(parents=True)
COMMON = ["--cpu", "--type", "live", "--mode", "progressive", "--chunk-frames", "20",
          "--height", "960", "--esrgan", FAKE, "--face-models", str(MODELS), *rest]


def run(name, extra):
    out = d / name / "out.mkv"
    if REUSE:
        return out, {"rc": 0, "reused": True}
    t0 = time.time()
    p = subprocess.run([sys.executable, SCRIPT, str(src), str(out), *COMMON, *extra,
                        "--work", str(d / name / "work")], capture_output=True, text=True)
    res = {"rc": p.returncode, "secs": round(time.time() - t0),
           "said": [l.strip() for l in p.stdout.splitlines()
                    if l.strip().startswith(("Faces", "NOTE", "WARNING", "chunk"))]}
    if p.returncode:
        res["error"] = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()][-8:]
    return out, res


def frames(path):
    w, h = map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                    "-show_entries", "stream=width,height", "-of", "csv=p=0",
                                    str(path)], capture_output=True, text=True).stdout.split(","))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w)


with_out, with_res = run("with", ["--faces", *([strength] if strength else [])])
without_out, without_res = run("without", [])
res = {"layout": layout, "strength": strength or "default", "with": with_res,
       "without": without_res}
if not with_res["rc"] and not without_res["rc"]:
    fa, fb = frames(with_out), frames(without_out)
    res["frames"] = [len(fa), len(fb)]
    h, w = fb.shape[1:]
    det = cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2023mar.onnx"), "",
                                    (w, h), 0.7, 0.3, 5)
    inside, outside, flicker, box, prev = [], [], [], None, None
    for a, b in zip(fa, fb):
        found = det.detect(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR))[1]
        if found is not None:
            x, y, bw, bh = found[0][:4]
            box = (int(x), int(y), int(x + bw), int(y + bh))
        diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float32)
        if box is None:
            inside.append(None)
            outside.append(round(float(diff.mean()), 3))
            continue
        x0, y0, x1, y1 = box
        inside.append(round(float(diff[max(0, y0):y1, max(0, x0):x1].mean()), 3))
        face = (slice(max(0, y0), y1), slice(max(0, x0), x1))
        if prev is not None:
            pa, pb = prev
            fl = (np.abs(a[face].astype(np.int16) - pa[face]).mean()
                  - np.abs(b[face].astype(np.int16) - pb[face]).mean())
            flicker.append(round(float(fl), 3))
        else:
            flicker.append(None)
        prev = (a, b)
        bw, bh = x1 - x0, y1 - y0
        keep = np.ones(diff.shape, bool)
        keep[max(0, y0 - bh):y1 + bh, max(0, x0 - bw):x1 + bw] = False
        outside.append(round(float(diff[keep].mean()), 3))
    res["face_diff"] = inside
    res["outside_diff"] = outside
    vals = [v for v in inside if v is not None]
    res["face_diff_mean"] = round(float(np.mean(vals)), 3) if vals else None
    res["outside_diff_max"] = max(outside)
    steps = {t: abs(inside[t] - inside[t - 1]) for t in range(1, len(inside))
             if inside[t] is not None and inside[t - 1] is not None}
    seams = [t for t in (20, 40) if t in steps]
    res["seam_change"] = {f"{t - 1}/{t}": round(steps[t], 3) for t in seams}
    within = [v for t, v in steps.items() if t not in seams]
    res["within_change"] = {"median": round(float(np.median(within)), 3),
                            "p90": round(float(np.percentile(within, 90)), 3),
                            "max": round(float(np.max(within)), 3)} if within else None
    # (flicker[t]: frames t-1 -> t)
    fl = {t: v for t, v in enumerate(flicker) if v is not None}
    res["extra_flicker_seams"] = {f"{t - 1}/{t}": fl[t] for t in (20, 40) if t in fl}
    rest = [v for t, v in fl.items() if t not in (20, 40)]
    res["extra_flicker_within"] = {"median": round(float(np.median(rest)), 3),
                                   "p90": round(float(np.percentile(rest, 90)), 3),
                                   "max": round(float(np.max(rest)), 3)} if rest else None
print(json.dumps(res), flush=True)
