"""Golden reference: proof that a change to dvd_upscale.py (a refactor) leaves the picture alone.

usage: python3 golden.py record NAME [scenario ...]    run the scenarios, save golden/NAME.json
       python3 golden.py compare BASE NEW               differences between two recordings

For every scenario it runs the script on the test media (make_test_media.sh) and records:
- every ffmpeg / ffprobe / upscaler command line it ran (through logging stand-ins on the PATH),
  with the run's own paths replaced by placeholders, sorted (threads run them in any order)
- an MD5 of every decoded frame of the output (video and sound): equal means bit-identical
- the work folder's settings.json and detected.json (what a resumed run is compared against)
- the messages it printed, with times, speeds and clock times blanked out (reported, not gating)
Equal commands and equal frame MD5s: the change can't have touched the picture or the sound.
"""
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = Path(os.environ.get("GOLDEN_SCRIPT") or HERE.parent / "dvd_upscale.py")  # (a copy: GOLDEN_SCRIPT=path)
FAKE = HERE / "fakeesrgan" / "realesrgan-ncnn-vulkan"
RUNS = HERE / "runs" / "golden"
OUT = HERE / "golden"
COMMON = ["--cpu"]

# name: (source, args); "@ESRGAN" becomes the logging stand-in for the upscaler
SCENARIOS = {
    "live_ai": ("sync/cut.mpg", ["--type", "live", "--mode", "progressive", "--chunk-frames", "30",
                                 "--test", "3", "--height", "240", "--esrgan", "@ESRGAN"]),
    "live_two_gpus": ("sync/cut.mpg", ["--type", "live", "--mode", "progressive", "--chunk-frames",
                                       "30", "--test", "3", "--height", "240", "--gpu", "0,1",
                                       "--esrgan", "@ESRGAN"]),
    "anime_ai": ("sync/cut.mpg", ["--type", "anime", "--mode", "progressive", "--chunk-frames", "30",
                                  "--test", "2", "--height", "240", "--esrgan", "@ESRGAN"]),
    "options_ai": ("sync/cut.mpg", ["--type", "live", "--mode", "progressive", "--chunk-frames", "30",
                                    "--test", "2", "--height", "360", "--ai-blend", "0.5",
                                    "--smooth", "0", "--sharpen", "1", "--dar", "16:9",
                                    "--esrgan", "@ESRGAN"]),
    "hevc_fast": ("sync/cut.mpg", ["--fast", "--type", "live", "--mode", "progressive", "--test", "2",
                                   "--height", "240", "--hevc"]),
    "ntsc_film_fast": ("tc/tc.mpg", ["--fast", "--type", "live", "--mode", "telecine",
                                     "--chunk-frames", "48", "--test", "6", "--height", "240"]),
    "ntsc_film_ai": ("tc/tcb.mpg", ["--type", "live", "--mode", "telecine", "--chunk-frames", "48",
                                    "--test", "4", "--height", "240", "--esrgan", "@ESRGAN"]),
    "pal_film_ai": ("pal/bc_tff.mpg", ["--type", "live", "--mode", "telecine", "--chunk-frames", "50",
                                       "--test", "4", "--height", "240", "--esrgan", "@ESRGAN"]),
    "interlaced_fast": ("pal/bc_tff.mpg", ["--fast", "--type", "live", "--mode", "interlaced",
                                           "--test", "2", "--height", "240"]),
    "vhs_ai": ("avi/cap.avi", ["--type", "vhs", "--mode", "progressive", "--chroma-delay", "off",
                               "--chunk-frames", "30", "--test", "2", "--height", "240",
                               "--esrgan", "@ESRGAN"]),
    "vhs_camcorder_fast": ("avi/cap.avi", ["--fast", "--type", "vhs", "--mode", "interlaced",
                                           "--chunk-frames", "60", "--test", "2", "--height", "240"]),
    "vhs_auto_fast": ("avi/h264cap.avi", ["--fast", "--type", "vhs", "--chunk-frames", "30",
                                          "--height", "240"]),
    "stabilize_ai": ("shaky.mpg", ["--type", "live", "--mode", "progressive", "--stabilize",
                                   "--chunk-frames", "30", "--test", "3", "--height", "240",
                                   "--esrgan", "@ESRGAN"]),
    "stabilize_strong_fast": ("shaky.mpg", ["--fast", "--type", "vhs", "--mode", "progressive",
                                            "--chroma-delay", "off", "--stabilize", "strong",
                                            "--test", "3", "--height", "240"]),
    "faces_ai": ("faces/medium/sd.mpg", ["--type", "live", "--mode", "progressive", "--faces",
                                         "--chunk-frames", "20", "--height", "480",
                                         "--face-models", str(HERE / "faces" / "face_models"),
                                         "--esrgan", "@ESRGAN"]),
    "detect_auto": ("tc/tc.mpg", ["--analyze"]),
    "detect_auto_vhs": ("avi/cap.avi", ["--analyze"]),
}
SPECIAL = ["clip", "queue", "all_folder"]


def shims(d, log):
    """Logging stand-ins: record the command line, then run the real tool."""
    d.mkdir(parents=True, exist_ok=True)
    real = {t: shutil.which(t) for t in ("ffmpeg", "ffprobe")}
    real["realesrgan-ncnn-vulkan"] = str(FAKE)
    for name, target in real.items():
        p = d / name
        p.write_text("#!/usr/bin/env python3\n"
                     "import json, os, sys\n"
                     f"with open({str(log)!r}, 'a') as f:\n"
                     "    f.write(json.dumps({'tool': os.path.basename(sys.argv[0]), "
                     "'cwd': os.getcwd(), 'argv': sys.argv[1:]}) + '\\n')\n"
                     f"os.execv({target!r}, [{target!r}] + sys.argv[1:])\n")
        p.chmod(0o755)
    return d / "realesrgan-ncnn-vulkan"


def normalize(text, run_dir):
    text = text.replace(str(run_dir), "<RUN>").replace(str(HERE), "<DEV>")
    text = text.replace(str(SCRIPT.parent), "<REPO>")
    return re.sub(r"/tmp/[^\s'\"]+", "<TMPFILE>", text)


VOLATILE = [
    (re.compile(r"0x[0-9a-fA-F]{6,}"), "<ADDR>"),          # (ffmpeg's object addresses)
    (re.compile(r"\d+(\.\d+)?\s*(s|sec|secs|seconds|min|mins|minutes|h|hours?)\b"), "<T>"),
    (re.compile(r"\d+:\d\d( ?[AP]M)?"), "<CLOCK>"),
    (re.compile(r"\d+(\.\d+)? ?(frames/s|fps|x realtime|MB/s|GB|MB|%)"), "<N>"),
]


def messages(text, run_dir):
    out = []
    for line in normalize(text, run_dir).replace("\r", "\n").splitlines():
        line = line.rstrip()
        if not line or "upscaling frame" in line or "encoding" in line.lower() and "%" in line:
            continue
        for rx, rep in VOLATILE:
            line = rx.sub(rep, line)
        out.append(line)
    return out


def frame_md5(path):
    p = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v?",
                        "-map", "0:a?", "-f", "framemd5", "-"], capture_output=True, text=True)
    return [l.split(",")[-1].strip() + "@" + l.split(",")[0].strip()
            for l in p.stdout.splitlines() if l and not l.startswith("#")]


def run_one(name, src, args, run_dir, cwd=None, env_extra=None, outputs=None):
    """outputs: the files to hash, or a function giving them once the run is done"""
    log = run_dir / "commands.jsonl"
    esrgan = shims(run_dir / "shims", log)
    args = [str(esrgan) if a == "@ESRGAN" else a for a in args]
    # OMP_NUM_THREADS=1: vid.stab's motion detection (--stabilize) runs on OpenMP threads, and
    # which of its many local measurements come out as outliers varies with their timing (the
    # stabilized picture of two runs measured the same: PSNR inf); one thread makes it repeat
    env = dict(os.environ, PATH=str(run_dir / "shims") + os.pathsep + os.environ["PATH"],
               OMP_NUM_THREADS="1", **(env_extra or {}))
    t0 = time.time()
    p = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True,
                       env=env, cwd=cwd or run_dir)
    rec = {"rc": p.returncode, "secs": round(time.time() - t0)}
    cmds = []
    if log.exists():
        for line in log.read_text().splitlines():
            c = json.loads(line)
            cmds.append(normalize(json.dumps([c["tool"], c["argv"], c["cwd"]]), run_dir))
    rec["commands"] = sorted(cmds)
    rec["messages"] = messages(p.stdout + p.stderr, run_dir)
    outs = outputs() if callable(outputs) else outputs
    outs = outs if outs is not None else ([run_dir / "out.mkv"] if "--analyze" not in args else [])
    outs = outs or [run_dir / "missing-output"]
    rec["frames"] = {normalize(str(o), run_dir): frame_md5(o) if o.exists() else None for o in outs}
    for f in ("settings.json", "detected.json"):
        for w in sorted(run_dir.glob(f"**/{f}")):
            rec.setdefault("work", {})[normalize(str(w), run_dir)] = normalize(w.read_text(), run_dir)
    return rec


def run_scenario(name):
    run_dir = RUNS / name
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    if name in SCENARIOS:
        src, args = SCENARIOS[name]
        out = [] if "--analyze" in args else [str(run_dir / "out.mkv")]
        return run_one(name, src, [str(HERE / src), *out, *COMMON, *args,
                                   *([] if "--analyze" in args else ["--work", str(run_dir / "work")])],
                       run_dir)
    if name == "clip":
        shutil.copy2(HERE / "tc" / "tc.mpg", run_dir / "tc.mpg")
        return run_one(name, None, ["--clip", "tc.mpg", "0:05", "3"], run_dir,
                       outputs=lambda: sorted(run_dir.glob("*clip*.mkv")))
    if name == "queue":
        for f in ("sync/cut.mpg", "tc/tc.mpg"):
            shutil.copy2(HERE / f, run_dir / Path(f).name)
        (run_dir / "queue.txt").write_text(
            "cut.mpg q1.mkv --fast --type live --mode progressive --test 2 --cpu --height 240\n"
            "tc.mpg q2.mkv --fast --type live --mode telecine --test 3 --cpu --height 240\n")
        return run_one(name, None, ["--queue", "queue.txt"], run_dir,
                       outputs=[run_dir / "q1.mkv", run_dir / "q2.mkv"])
    if name == "all_folder":
        (run_dir / "movies" / "vhs").mkdir(parents=True)
        shutil.copy2(HERE / "avi" / "cap.avi", run_dir / "movies" / "vhs" / "tape.avi")
        return run_one(name, None, ["--all", "movies", "--fast", "--cpu", "--height", "240",
                                   "--chroma-delay", "off"], run_dir,
                       outputs=lambda: sorted((run_dir / "movies").glob("* Upscale/**/*.mkv")))
    raise SystemExit(f"unknown scenario {name}")


def record(tag, names):
    OUT.mkdir(exist_ok=True)
    names = names or [*SCENARIOS, *SPECIAL]
    path = OUT / f"{tag}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for n in names:
        r = run_scenario(n)
        res[n] = r
        nframes = sum(len(v or []) for v in r["frames"].values())
        print(f"{n:24} rc={r['rc']} {r['secs']:4d}s  {len(r['commands']):3d} commands  "
              f"{nframes:5d} frame/sound hashes", flush=True)
        path.write_text(json.dumps(res, indent=1))


def compare(a_tag, b_tag):
    a = json.loads((OUT / f"{a_tag}.json").read_text())
    b = json.loads((OUT / f"{b_tag}.json").read_text())
    bad = 0
    for n in a:
        if n not in b:
            print(f"{n:24} MISSING in {b_tag}"); bad += 1; continue
        x, y = a[n], b[n]
        issues = []
        if x["rc"] != y["rc"]:
            issues.append(f"exit code {x['rc']} -> {y['rc']}")
        if x["frames"] != y["frames"]:
            issues.append("OUTPUT FRAMES DIFFER")
        if x["commands"] != y["commands"]:
            sx, sy = set(x["commands"]), set(y["commands"])
            issues.append(f"COMMANDS DIFFER (-{len(sx - sy)} +{len(sy - sx)})")
            for c in sorted(sx - sy)[:3]:
                print(f"      - {c[:300]}")
            for c in sorted(sy - sx)[:3]:
                print(f"      + {c[:300]}")
        if x.get("work") != y.get("work"):
            issues.append("SETTINGS FILES DIFFER")
        gate = bool(issues)
        if x["messages"] != y["messages"]:
            mx, my = set(x["messages"]), set(y["messages"])
            issues.append(f"messages differ (-{len(mx - my)} +{len(my - mx)}): "
                          f"{sorted(mx - my)[:2]} / {sorted(my - mx)[:2]}")
        bad += gate
        print(f"{n:24} {'IDENTICAL' if not issues else '; '.join(issues)}")
    print(f"\n{'PASS: identical picture, sound and commands' if not bad else f'FAIL: {bad} scenarios'}")
    return bad


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    if cmd == "record":
        record(rest[0], rest[1:])
    elif cmd == "compare":
        sys.exit(1 if compare(rest[0], rest[1]) else 0)
    else:
        raise SystemExit(__doc__)
