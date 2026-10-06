"""Run dvd_upscale.py with --stabilize on the shaky clip and check the result.
usage: python3 stabrun.py NAME ZOOM [script args...]   (prints one JSON line)"""
import json, os, subprocess, sys, time
from pathlib import Path
import stabtest

HERE = Path(__file__).resolve().parent
SCRIPT = str(HERE.parent / "dvd_upscale.py")
name, zoom, *args = sys.argv[1:]
d = HERE / "runs" / name
subprocess.run(["rm", "-rf", str(d)])
d.mkdir(parents=True)
src = HERE / os.environ.get("SRC", "shaky.mpg")
PERIOD = eval(os.environ.get("PERIOD", "1001/30000"))
t0 = time.time()
p = subprocess.run([sys.executable, SCRIPT, str(src), str(d / "out.mkv"), "--cpu", "--height", "480",
                    "--work", str(d / "work"), *args], capture_output=True, text=True)
res = {"run": name, "rc": p.returncode, "secs": round(time.time() - t0)}
if p.returncode:
    res["error"] = [l for l in (p.stdout + p.stderr).splitlines() if l.strip()][-3:]
else:
    rows = stabtest.measure(d / "out.mkv", float(zoom))
    seq, breaks = stabtest.order(rows)
    src_rows = stabtest.measure(src)
    p0 = src_rows[0][0] - src_rows[0][1] * PERIOD
    cs = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time",
                               "-of", "csv=p=0", str(src)], capture_output=True, text=True).stdout)
    errs = [t - (p0 + n * PERIOD - cs) for t, n, *_ in rows]
    jx, jy = stabtest.jitter(rows)
    res.update(frames=len(rows), breaks=len(breaks), first_breaks=breaks[:3],
               jitter=[round(jx, 2), round(jy, 2)],
               sync_ms=[round(min(errs) * 1000), round(max(errs) * 1000)],
               notes=[l.strip() for l in p.stdout.splitlines() if "NOTE" in l or "camera shake" in l])
print(json.dumps(res), flush=True)
