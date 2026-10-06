"""End-to-end runs of dvd_upscale.py on synthetic sources with frame-number barcodes.

usage: python3 e2e.py SCRIPT TAG [case ...]
Each case runs the script into runs/TAG/CASE/, then checks the output's frame order and
picture-vs-sound offset (output pts against the source frame's pts minus the source start).
"""
import json, os, shutil, subprocess, sys, time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import barcode

FAKE = str(HERE / "fakeesrgan" / "realesrgan-ncnn-vulkan")
NTSC, FILM, PAL = Fraction(1001, 30000), Fraction(1001, 24000), Fraction(1, 25)
COMMON = ["--cpu", "--height", "240"]

CASES = {
    # name: (source, extra args, frame period, expected frames, env)
    "mpeg_midgop": ("sync/cut.mpg", ["--fast", "--type", "live", "--mode", "progressive",
                                     "--chunk-frames", "30", "--test", "10"], NTSC, 300, {}),
    "ntsc_film_tff": ("tc/tc.mpg", ["--fast", "--type", "live", "--mode", "telecine",
                                    "--chunk-frames", "48", "--test", "20"], FILM, 480, {}),
    "ntsc_film_bff": ("tc/tcb.mpg", ["--fast", "--type", "live", "--mode", "telecine",
                                     "--chunk-frames", "48", "--test", "20"], FILM, 480, {}),
    "ai_path": ("sync/cut.mpg", ["--type", "live", "--mode", "progressive", "--chunk-frames", "30",
                                 "--test", "6", "--esrgan", FAKE], NTSC, 180, {}),
    "two_gpus": ("sync/cut.mpg", ["--type", "live", "--mode", "progressive", "--chunk-frames", "30",
                                  "--test", "12", "--gpu", "0,1", "--esrgan", FAKE], NTSC, 360,
                 {"FAKE_SLOW_GPU": "1", "FAKE_SLOW_SECS": "6"}),
    "pal_film_tff": ("pal/bc_tff.mpg", ["--fast", "--type", "live", "--mode", "telecine",
                                        "--chunk-frames", "50", "--test", "10"], PAL, 250, {}),
    "pal_film_bff": ("pal/bc_bff.mpg", ["--fast", "--type", "live", "--mode", "telecine",
                                        "--chunk-frames", "50", "--test", "10"], PAL, 250, {}),
    "pal_film_prog": ("pal/bc_prog.mpg", ["--fast", "--type", "live", "--mode", "telecine",
                                          "--chunk-frames", "50", "--test", "10"], PAL, 250, {}),
    "vhs_avi_mpeg4": ("avi/cap.avi", ["--fast", "--type", "vhs", "--mode", "progressive",
                                      "--chroma-delay", "off", "--chunk-frames", "60", "--test", "6"],
                      NTSC, 180, {}),
    # AVIs with sound: frame n belongs at n/fps (an AVI stores no picture timestamps)
    "avi_mpeg4_dvd": ("avi/av_mpeg4.avi", ["--fast", "--type", "live", "--mode", "progressive",
                                           "--chunk-frames", "60"], NTSC, 240, {}, 0.0),
    "avi_h264_dvd": ("avi/av_h264.avi", ["--fast", "--type", "live", "--mode", "progressive",
                                         "--chunk-frames", "60"], NTSC, 240, {}, 0.0),
    "avi_mpeg4_vhs": ("avi/av_mpeg4.avi", ["--fast", "--type", "vhs", "--mode", "progressive",
                                           "--chroma-delay", "off", "--chunk-frames", "60"],
                      NTSC, 240, {}, 0.0),
    "avi_h264_vhs": ("avi/av_h264.avi", ["--fast", "--type", "vhs", "--mode", "progressive",
                                         "--chroma-delay", "off", "--chunk-frames", "60"],
                     NTSC, 240, {}, 0.0),
    "vhs_avi_h264": ("avi/h264cap.avi", ["--fast", "--type", "vhs", "--mode", "progressive",
                                         "--chroma-delay", "off", "--chunk-frames", "30"], NTSC, 120, {}),
}


def start_time(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of",
                          "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def run_case(script, tag, name):
    src, extra, period, want, env, *p0_fixed = CASES[name]
    src = HERE / src
    d = HERE / "runs" / tag / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    out = d / "out.mkv"
    t0 = time.time()
    p = subprocess.run([sys.executable, script, str(src), str(out), *COMMON, *extra,
                        "--work", str(d / "work")], capture_output=True, text=True,
                       env=dict(os.environ, **env), cwd=d)
    took = time.time() - t0
    res = {"case": name, "rc": p.returncode, "secs": round(took)}
    if p.returncode or not out.exists():
        lines = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()]
        res["error"] = lines[-1][:160] if lines else "?"
        return res
    first = barcode.decode(src)[0]
    p0 = p0_fixed[0] if p0_fixed else first[0] - first[1] * float(period)   # frame 0's time
    cs = start_time(src)
    frames = barcode.decode(out)
    seq = [n for _, n in frames]
    errs = [t - (p0 + n * float(period) - cs) for t, n in frames]
    breaks = [(i, seq[i - 1], seq[i]) for i in range(1, len(seq)) if seq[i] != seq[i - 1] + 1]
    res.update(frames=len(seq), want=want, first=seq[:3], breaks=len(breaks),
               first_breaks=breaks[:3],
               sync_ms=[round(min(errs) * 1000), round(max(errs) * 1000)],
               chunk_sync_ms=[round(sum(errs[i:i + 60]) / len(errs[i:i + 60]) * 1000)
                              for i in range(0, len(errs), 60)])
    return res


if __name__ == "__main__":
    script, tag, *names = sys.argv[1:]
    script = str(Path(script).resolve())      # (each case runs in its own folder)
    for name in names or list(CASES):
        r = run_case(script, tag, name)
        print(json.dumps(r), flush=True)
