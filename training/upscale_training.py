"""DVD -> Blu-ray training in one command: pairs, training, and the model for dvd_upscale.py.

usage:
  python upscale_training.py --dvd movie_dvd.mkv --bluray movie_bd.mkv

It does three steps, one after the other, and tells you which one it is on:
  1. pairs   aligned DVD / Blu-ray frame pairs from the two movies        (a few hours)
  2. train   fine-tunes Real-ESRGAN x2plus on them                        (about 2 hours on an RTX 3060)
  3. export  writes upscale-training-x2.param / .bin into the upscaler's models folder (seconds)
Everything goes in a folder next to the DVD file (upscale_training_work/pairs, upscale_training_work/run); --work puts it elsewhere.

Ctrl+C at any time, then run the same command again: it carries on where it stopped (steps already
finished are skipped; the pairs and the training both resume). --stages picks steps, e.g.
  --stages train,export        (pairs already made)       --stages export     (just convert again)

Then upscale with:  python dvd_upscale.py <movie> --model upscale-training-x2 --scale 2

While it runs, PowerShell shows one live line with an ETA, and a page opens on port 8643 for a browser
or a phone on the same Wi-Fi (the address is printed; --web 0 turns it off; read-only, no password).

How the two movies are lined up (step 1):
  1. black bars are found on both (ffmpeg cropdetect) and cut off, so only the picture is compared
  2. the time offset between the discs is measured every --anchor-every seconds from tiny thumbnails.
     It may change along the movie (a longer logo, an extra scene on one disc): each sample uses the
     offset of the nearest measurements. (--offset sets one by hand; --speed 1.0427 for a PAL disc)
  3. for each sample the Blu-ray frame that looks most like the DVD frame is picked from the frames
     within --window seconds of the expected time
  4. a small affine warp (cv2 ECC) removes the leftover shift/scale of the two transfers
  5. the Blu-ray's slow colour/brightness differences are matched to the DVD's, so the model learns
     detail, not the other disc's colour grade (--color-match none turns that off)
Step 2 uses L1 loss only by default (the model learns what the Blu-ray shows, nothing invented);
--perceptual 0.5 --gan 0.05 add texture, and can invent detail. It judges itself on every 20th pair,
never trained on, and prints that PSNR next to plain bicubic and the starting model.

needs: ffmpeg + ffprobe on the PATH, pip install numpy opencv-python torch torchvision
"""
import argparse, copy, csv, datetime, http.server, json, math, random, re, socket, struct
import os, shutil, subprocess, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
import numpy as np
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

PRETRAINED_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"


# =============================== live progress: a page and a console line ===============================
PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Progress</title>
<style>
:root{--bg:#f6f7f9;--fg:#1b1f24;--mut:#6a737d;--card:#fff;--line:#d9dde3;--acc:#2563eb;--ok:#15803d}
@media(prefers-color-scheme:dark){:root{--bg:#0f1318;--fg:#e6e9ee;--mut:#98a2b0;--card:#171c23;--line:#2a313b;--acc:#60a5fa;--ok:#4ade80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.4 system-ui,sans-serif}
main{max-width:640px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 2px}
.sub{color:var(--mut);font-size:14px;margin-bottom:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:12px}
.bar{height:14px;background:var(--line);border-radius:7px;overflow:hidden}
.bar>div{height:100%;background:var(--acc);width:0;transition:width .5s}
.big{font-size:26px;font-weight:600}.row{display:flex;gap:16px;flex-wrap:wrap}
.m{min-width:90px}.m b{display:block;font-size:18px}.m span{color:var(--mut);font-size:13px}
canvas{width:100%;height:120px;display:block}h2{font-size:14px;color:var(--mut);margin:0 0 6px;font-weight:600}
pre{margin:0;font:12px/1.35 ui-monospace,Consolas,monospace;white-space:pre-wrap;word-break:break-word;max-height:220px;overflow:auto}
.done{color:var(--ok)}
</style></head><body><main>
<h1 id="title">Progress</h1><div class="sub" id="sub">connecting...</div>
<div class="card"><div class="row"><div class="big" id="pct">-</div><div class="m"><b id="eta">-</b><span>time left</span></div>
<div class="m"><b id="speed">-</b><span id="speedl">speed</span></div></div>
<div class="bar" style="margin-top:10px"><div id="fill"></div></div></div>
<div class="card" id="metrics"></div><div id="charts"></div>
<div class="card"><h2>Latest</h2><pre id="log"></pre></div></main>
<script>
const $=id=>document.getElementById(id);
function fmt(s){if(s==null||!isFinite(s))return '-';s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);
return h?h+' h '+m+' min':m?m+' min':s+' s'}
function chart(el,name,pts,refs){const c=el.querySelector('canvas'),r=devicePixelRatio||1;c.width=c.clientWidth*r;c.height=c.clientHeight*r;
const g=c.getContext('2d');g.scale(r,r);const W=c.clientWidth,H=c.clientHeight,st=getComputedStyle(document.body);
let ys=pts.map(p=>p[1]).concat(Object.values(refs||{}));if(!ys.length)return;let lo=Math.min(...ys),hi=Math.max(...ys);if(hi==lo){hi+=1;lo-=1}
const pad=(hi-lo)*.1;lo-=pad;hi+=pad;const xs=pts.map(p=>p[0]),x0=Math.min(...xs),x1=Math.max(...xs)||1;
const X=x=>6+(W-12)*(x1==x0?1:(x-x0)/(x1-x0)),Y=y=>H-14-(H-22)*(y-lo)/(hi-lo);
g.font='11px system-ui';g.fillStyle=st.getPropertyValue('--mut');
Object.entries(refs||{}).forEach(([k,v])=>{g.strokeStyle=st.getPropertyValue('--line');g.setLineDash([4,4]);g.beginPath();g.moveTo(0,Y(v));g.lineTo(W,Y(v));g.stroke();g.setLineDash([]);g.fillText(k+' '+v.toFixed(2),8,Y(v)-3)});
g.strokeStyle=st.getPropertyValue('--acc');g.lineWidth=2;g.beginPath();pts.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));g.stroke();
const l=pts[pts.length-1];g.fillStyle=st.getPropertyValue('--fg');g.fillText(l[1].toFixed(4).replace(/0+$/,''),W-60,12)}
async function tick(){try{const s=await (await fetch('state.json',{cache:'no-store'})).json();
$('title').textContent=s.title;$('sub').textContent=s.phase+' - updated '+Math.round(Date.now()/1000-s.updated)+' s ago';
const done=s.total?s.step/s.total:0;$('pct').textContent=s.total?(done*100).toFixed(1)+' %':'-';
$('pct').className='big'+(s.finished?' done':'');$('fill').style.width=(done*100)+'%';$('eta').textContent=s.finished?'finished':fmt(s.eta);
$('speed').textContent=s.speed_text||'-';
$('metrics').innerHTML='<div class="row"><div class="m"><b>'+s.step+(s.total?' / '+s.total:'')+'</b><span>'+(s.unit||'steps')+'</span></div>'+
Object.entries(s.metrics||{}).map(([k,v])=>'<div class="m"><b>'+v+'</b><span>'+k+'</span></div>').join('')+'</div>';
const ch=$('charts');Object.entries(s.series||{}).forEach(([k,pts])=>{let el=document.getElementById('c_'+k);
if(!el){el=document.createElement('div');el.id='c_'+k;el.className='card';el.innerHTML='<h2>'+k+'</h2><canvas></canvas>';ch.appendChild(el)}
if(pts.length>1)chart(el,k,pts,(s.refs||{})[k])});
$('log').textContent=(s.log||[]).join('\\n');}catch(e){$('sub').textContent='cannot reach the PC...'}}
tick();setInterval(tick,3000);
</script></body></html>"""


class Progress:
    def __init__(self, title, port, unit="steps"):
        self.s = {"title": title, "phase": "starting", "step": 0, "total": 0, "unit": unit, "eta": None,
                  "speed_text": "", "metrics": {}, "series": {}, "refs": {}, "log": [],
                  "finished": False, "updated": time.time()}
        self.port, self.lock = port, threading.Lock()
        self.last_hit = 0.0

    def set(self, **kw):
        with self.lock:
            self.s.update(kw)
            self.s["updated"] = time.time()

    def metric(self, **kw):
        with self.lock:
            self.s["metrics"].update({k: v for k, v in kw.items()})
            self.s["updated"] = time.time()

    def ref(self, series, label, value):
        with self.lock:
            self.s["refs"].setdefault(series, {})[label] = value

    def point(self, series, x, y):
        with self.lock:
            pts = self.s["series"].setdefault(series, [])
            pts.append([x, y])
            if len(pts) > 600:
                del pts[::2]               # keep it light: thin out the old points
            self.s["updated"] = time.time()

    def log(self, text):
        with self.lock:
            self.s["log"] = (self.s["log"] + [text])[-40:]
            self.s["updated"] = time.time()

    def reset(self, title):
        """A new stage: clear the numbers and charts (the log stays)."""
        with self.lock:
            self.s.update(title=title, phase="starting", step=0, total=0, eta=None, speed_text="",
                          metrics={}, series={}, refs={}, finished=False, unit="steps", updated=time.time())

    def hold(self, seconds=60):
        """If someone has the page open, keep it up a little after the end so they see 'finished'."""
        if not getattr(self, "started", False) or time.time() - self.last_hit > 120:
            return
        print(f"(the progress page stays up {seconds} more seconds; Ctrl+C closes it)", flush=True)
        try:
            time.sleep(seconds)
        except KeyboardInterrupt:
            pass

    def snapshot(self):
        with self.lock:
            return json.dumps(self.s).encode()

    def start(self):
        prog = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                prog.last_hit = time.time()
                if self.path.split("?")[0] in ("/", "/index.html"):
                    body, kind = PAGE.encode(), "text/html; charset=utf-8"
                elif self.path.split("?")[0] == "/state.json":
                    body, kind = prog.snapshot(), "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        try:
            srv = http.server.ThreadingHTTPServer(("0.0.0.0", self.port), H)
        except OSError as e:
            print(f"progress page: can't use port {self.port} ({e}); continuing without it")
            return self
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.started = True
        print("Progress page (open in a browser, on a phone too if on the same Wi-Fi):")
        for ip in addresses():
            print(f"   http://{ip}:{self.port}")
        print("   (if Windows asks, allow Python on private networks)", flush=True)
        return self


def addresses():
    """This PC's IPv4 addresses worth trying: the Wi-Fi/LAN one first, Tailscale (100.x) too."""
    found = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        found.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in found and not ip.startswith("127.") and not ip.startswith("169.254."):
                found.append(ip)
    except OSError:
        pass
    return found or ["localhost"]


# ---- one live status line in the console (PowerShell) ----
def hms(sec):
    sec = max(int(sec), 0)
    h, m = sec // 3600, sec % 3600 // 60
    return f"{h} h {m:02d} min" if h else (f"{m} min {sec % 60:02d} s" if m else f"{sec} s")


def clock_in(sec):
    t = datetime.datetime.now() + datetime.timedelta(seconds=max(sec, 0))
    return t.strftime("%I:%M %p").lstrip("0")


def bar(frac, width=24):
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "[" + "#" * n + "-" * (width - n) + "]"


class LiveLine:
    """A status line that rewrites itself; ordinary lines are printed above it."""

    def __init__(self):
        self.n = 0
        self.tty = sys.stdout.isatty()

    def show(self, text):
        if not self.tty:
            return
        text = text[:max(shutil_width() - 1, 20)]
        print("\r" + text.ljust(self.n), end="", flush=True)
        self.n = len(text)

    def fields(self, parts):
        """Show as many of the parts (most important first) as fit the window."""
        width = shutil_width() - 1
        text = parts[0]
        for p in parts[1:]:
            if len(text) + 2 + len(p) > width:
                break
            text += "  " + p
        self.show(text)

    def clear(self):
        if self.tty and self.n:
            print("\r" + " " * self.n + "\r", end="", flush=True)
            self.n = 0

    def say(self, text):
        self.clear()
        print(text, flush=True)


def shutil_width():
    try:
        import shutil
        return shutil.get_terminal_size((120, 20)).columns
    except OSError:
        return 120


# =============================== hardware: GPU watts, load, heat, CPU ===============================
THROTTLE_BITS = ((0x4, "power cap"), (0x8, "hardware slowdown"), (0x20, "thermal"),
                 (0x40, "hardware thermal"), (0x80, "power brake"))
GPU_FIELDS = ("name,power.draw,power.limit,utilization.gpu,temperature.gpu,clocks.sm,clocks.max.sm,"
              "memory.used,memory.total,clocks_throttle_reasons.active")


def cpu_times():
    """(busy, total) CPU time since boot, whole machine; None if it can't be read."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            i, k, u = (wintypes.FILETIME() for _ in range(3))
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
            v = lambda t: (t.dwHighDateTime << 32) | t.dwLowDateTime
            idle, tot = v(i), v(k) + v(u)             # (kernel time includes idle)
            return tot - idle, tot
        f = open("/proc/stat").readline().split()[1:]
        vals = [int(x) for x in f]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        return sum(vals) - idle, sum(vals)
    except Exception:
        return None


class HwMonitor:
    """Samples the NVIDIA GPU (nvidia-smi: watts, load, heat, clock, memory, why it slows down) and the
    processor load every 2 s in a thread. CPU watts are not readable without a driver or admin tool
    (HWiNFO / Intel Power Gadget / Ryzen Master show them): CPU load is what is shown instead."""

    def __init__(self, every=2.0):
        self.every, self.gpu, self.cpu, self.err = every, None, None, ""
        self.smi = shutil.which("nvidia-smi")
        self.hist = []                                 # recent samples, for the verdict
        self.lock = threading.Lock()
        self._cpu_prev = cpu_times()
        threading.Thread(target=self.loop, daemon=True).start()

    def sample(self):
        now = cpu_times()
        if now and self._cpu_prev and now[1] > self._cpu_prev[1]:
            self.cpu = 100 * (now[0] - self._cpu_prev[0]) / (now[1] - self._cpu_prev[1])
        self._cpu_prev = now
        if not self.smi:
            return
        try:
            r = subprocess.run([self.smi, "--query-gpu=" + GPU_FIELDS, "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=8, stdin=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            v = [x.strip() for x in r.stdout.splitlines()[0].split(",")]
            num = lambda x: float(x) if re.fullmatch(r"[\d.]+", x) else None
            try:
                bits = int(v[9], 16)
            except ValueError:
                bits = 0
            z = lambda x: num(x) or 0.0                  # (a "[N/A]" reading shows as 0, never crashes the training)
            self.gpu = dict(name=v[0], watts=num(v[1]), watts_max=num(v[2]), busy=z(v[3]), temp=z(v[4]),
                            clock=z(v[5]), clock_max=z(v[6]), mem=z(v[7]), mem_max=z(v[8]),
                            slowed=[n for b, n in THROTTLE_BITS if bits & b], bits=bits)
        except (OSError, subprocess.SubprocessError, IndexError, ValueError) as e:
            self.err = str(e)

    def loop(self):
        while True:
            try:
                self.sample()
                with self.lock:
                    self.hist = (self.hist + [(self.gpu, self.cpu)])[-15:]       # ~30 s
            except Exception as e:
                self.err = str(e)
            time.sleep(self.every)

    def line(self):
        """Short pieces for the console line."""
        g, out = self.gpu, []
        if g:
            if g["watts"] is not None:
                out.append(f"GPU {g['watts']:.0f}/{g['watts_max'] or 0:.0f} W")
            out.append(f"{g['busy']:.0f}% busy {g['temp']:.0f}C")
        if self.cpu is not None:
            out.append(f"CPU {self.cpu:.0f}%")
        return out

    def metrics(self):
        g, m = self.gpu, {}
        if g:
            if g["watts"] is not None:
                m["GPU power"] = f"{g['watts']:.0f} / {g['watts_max'] or 0:.0f} W"
            m["GPU busy"] = f"{g['busy']:.0f} %"
            m["GPU temp"] = f"{g['temp']:.0f} C"
            m["GPU clock"] = f"{g['clock']:.0f} / {g['clock_max']:.0f} MHz"
            m["GPU memory"] = f"{g['mem']:.0f} / {g['mem_max']:.0f} MiB"
            m["slowed by"] = ", ".join(g["slowed"]) or "nothing"
        if self.cpu is not None:
            m["CPU"] = f"{self.cpu:.0f} %"
        return m

    def avg(self, key, which=0):
        with self.lock:
            vals = [(h[0] or {}).get(key) if which == 0 else h[1] for h in self.hist]
        vals = [x for x in vals if x is not None]
        return sum(vals) / len(vals) if vals else None


def verdict(hw, wait_frac, step_s):
    """One sentence: what is holding the training back right now. wait_frac = share of each step the GPU
    spent waiting for the next batch from the data loader."""
    g = hw.gpu
    if wait_frac > 0.25:
        cpu = hw.avg(None, 1)
        return (f"SLOW: the GPU waits {100 * wait_frac:.0f}% of each step for data (disk/CPU loading PNGs)"
                + (f", CPU {cpu:.0f}%" if cpu is not None else "")
                + ". Try --train-workers 4 (or more) and keep the pairs folder on an SSD.")
    if not g:
        return "no nvidia-smi: can't see the GPU (data wait is %.0f%%)" % (100 * wait_frac)
    recent = [(h[0] or {}).get("bits", 0) for h in hw.hist[-8:]]
    if recent and sum(1 for b in recent if b & (0x20 | 0x40)) > len(recent) / 2:
        return f"SLOW: the GPU is too hot ({g['temp']:.0f} C) and lowers its clock: more airflow / fan curve."
    if recent and sum(1 for b in recent if b & 0x4) > len(recent) / 2:
        return "the GPU is at its power limit (normal when fully loaded; it is working as hard as it is allowed)."
    if recent and sum(1 for b in recent if b & 0x8) > len(recent) / 2:
        return "SLOW: hardware slowdown flagged by the GPU (power supply or cable, or heat)."
    if g["mem_max"] and g["mem"] / g["mem_max"] > 0.95:
        return "GPU memory is nearly full: Windows may be swapping it; try --batch 4 or --checkpoint."
    busy = hw.avg("busy")
    if busy is not None and busy < 70:
        return (f"GPU only {busy:.0f}% busy but data wait is {100 * wait_frac:.0f}%: something else on the "
                "PC is using it, or the steps are too small.")
    return "the GPU is the limit (busy, cool, not waiting for data): this is as fast as it goes. Lower --batch/--patch or --iters to finish sooner."


# =============================== step 1: the DVD / Blu-ray pairs ===============================
THUMB = (64, 36)
PROG = None                 # the live progress page (--web), when asked for


LIVE = None                 # the status line in the console
HW = None                   # the hardware monitor (GPU watts, load, heat, CPU)


def say(msg):
    if LIVE:
        LIVE.say(msg)
    else:
        print(msg, flush=True)
    if PROG:
        PROG.log(msg)


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
        say(f"  offset {i}/{n_total} at {t / 60:5.1f} min: "
            + (f"Blu-ray {off:+7.2f} s, match {sc:.3f}" if ok else f"no clear match ({sc:.2f})"))
        if PROG:
            PROG.set(phase="measuring the offset between the discs", step=i, total=n_total,
                     unit="offset measurements")
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



def stage_pairs(a, out):
    """Step 1: aligned DVD / Blu-ray frame pairs into out/lr and out/hr. Carries on from pairs already there.
    True when done, False when stopped with Ctrl+C."""
    (out / "lr").mkdir(parents=True, exist_ok=True)
    (out / "hr").mkdir(exist_ok=True)
    csv_path = out / "pairs.csv"
    if a.fresh:
        for sub in ("lr", "hr"):
            for f in (out / sub).glob("*.png"):
                f.unlink()
        if csv_path.exists():
            csv_path.unlink()
    done_times, have = set(), 0
    if csv_path.exists():
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                done_times.add(round(float(r["t_dvd"]), 3))
                have = max(have, int(r["name"]))
    if have:
        say(f"found {have} pairs already in {out}: carrying on from there (--fresh starts over)")
        if have >= a.count:
            say(f"already {have} pairs, which is --count {a.count} or more: nothing to do")
            return True
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
    times = [t for t in times if round(t, 3) not in done_times]
    done, rejects = have, Counter()
    t_start = time.time()
    if PROG:
        PROG.set(phase="making pairs", step=have, total=a.count, unit="pairs", eta=None, speed_text="")
    def work(t):
        try:
            return make_pair(a, t, dvd_crop, bd_crop, dvd_pre, lrw, lrh, anchors, bd_fps)
        except (SystemExit, Exception) as e:          # (run() exits on an ffmpeg error: skip this frame, not the whole step)
            return None, None, "error " + str(e).strip().splitlines()[0][:60] if str(e).strip() else "error"
    stopped = False
    with open(csv_path, "a" if have else "w", newline="") as fh, ThreadPoolExecutor(max(1, a.pair_workers)) as ex:
        wr = csv.writer(fh)
        if not have:
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
                    say(f"  {done}/{a.count} pairs  (rejected: {dict(rejects)})")
                el = time.time() - t_start
                new = done - have                       # (made in this run: the speed and ETA are from these)
                eta = el / new * (a.count - done)
                LIVE.fields([f"{bar(done / a.count, 20)} {100 * done / a.count:5.1f}%  {done}/{a.count}",
                             f"ETA {hms(eta)} (~{clock_in(eta)})",
                             f"{new / el * 60:.1f} pairs/min", f"elapsed {hms(el)}",
                             f"rejected {sum(rejects.values())}"])
                if PROG:
                    PROG.set(step=done, eta=eta, speed_text=f"{new / el * 60:.1f} pairs/min")
                    PROG.metric(rejected=sum(rejects.values()))
                if done >= a.count:
                    break
        except KeyboardInterrupt:
            stopped = True
            LIVE.clear()
            print("\nStopping (finishing the frames in progress)...", flush=True)
        finally:
            for f in futs:
                f.cancel()
    LIVE.clear()
    if stopped:
        say(f"Stopped with {done} pairs saved in {out}. Run the same command again to carry on from here.")
        if PROG:
            PROG.set(phase=f"stopped at {done} pairs (run the command again to resume)")
        return False
    say(f"done: {done} pairs in {out}  (rejected: {dict(rejects)})")
    if PROG:
        PROG.set(phase="finished", step=done, eta=0, finished=True)
    if done < a.count // 2:
        say("few pairs survived: check the picture areas above, or lower --min-thumb/--min-ecc/--min-ncc")
    return True


# =============================== step 2: training (the network, the data, the losses) ===============================
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
        for _ in range(10):
            name = random.choice(s.names)
            lr = cv2.imread(str(s.root / "lr" / f"{name}.png"))
            hr = cv2.imread(str(s.root / "hr" / f"{name}.png"))
            if lr is not None and hr is not None and hr.shape[0] == 2 * lr.shape[0] and hr.shape[1] == 2 * lr.shape[1]:
                break
        else:
            raise RuntimeError(f"can't read a valid pair from {s.root} (10 tries)")
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



def stage_train(a, root, out):
    """Step 2: fine-tune on the pairs in root. Carries on from out/train_state.pt.
    True when finished, False when stopped with Ctrl+C."""
    prog, live = PROG, LIVE
    if a.pretrained != "none" and not Path(a.pretrained).exists():
        raise SystemExit(f"starting model not found: {a.pretrained}\nDownload RealESRGAN_x2plus.pth from\n"
                         "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth\n"
                         "and put it in the training folder (or pass its path with --pretrained).")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        print("WARNING: no CUDA GPU found, training on the processor (very slow)")
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(f.stem for f in (root / "lr").glob("*.png") if (root / "hr" / f.name).exists())
    val = names[::20]
    held = set(val)
    train = [n for n in names if n not in held]
    if len(train) < 8:
        raise SystemExit(f"only {len(names)} pairs in {root}: make more in the pairs step")
    say(f"{len(train)} training pairs, {len(val)} held out; device {dev}")
    probe_img = cv2.imread(str(root / "lr" / f"{names[0]}.png"))
    if probe_img is None or min(probe_img.shape[:2]) < a.patch:
        raise SystemExit(f"the DVD pictures in {root / 'lr'} are smaller than --patch {a.patch}: lower --patch")
    if prog:
        prog.set(title=f"Training: {out.name}", total=a.iters, phase="starting")

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
        if "scaler" in st:
            scaler.load_state_dict(st["scaler"])
        say(f"resuming at step {step}")

    # where we start from, on frames never trained on
    floor = psnr_plain(root, val, a.val_crop)
    base = psnr_on(ema, dev, root, val, a.val_crop) if step == 0 else None
    if base is not None:
        say(f"held-out PSNR: plain bicubic {floor:.2f} dB, starting model {base:.2f} dB")
    if prog:
        prog.ref("held-out PSNR (dB)", "bicubic", floor)
        if base is not None:
            prog.ref("held-out PSNR (dB)", "start", base)
        prog.set(step=step, phase="training")

    per = 4
    ds = Pairs(root, train, a.patch, per)
    dl = torch.utils.data.DataLoader(ds, batch_size=max(1, a.batch // per), num_workers=a.train_workers,
                                     collate_fn=collate, persistent_workers=a.train_workers > 0)
    def save_state():
        torch.save({"params_ema": ema.state_dict()}, out / "upscale_training_latest.pth")
        torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "step": step, "best": best, "scaler": scaler.state_dict(),
                    **({"disc": disc.state_dict(), "d_opt": d_opt.state_dict()} if disc else {})}, state_path)

    it = iter(dl)
    t0, run = time.time(), {}
    run_t0, start_step, last_l1 = time.time(), step, float("nan")
    wait_t, loop_t, tick = 0.0, 0.0, time.time()      # data-loader wait vs whole step, for the verdict
    net.train()
    try:
        while step < a.iters:
            w0 = time.time()
            batch = next(it)                       # (blocks while the data loader is behind)
            wait_t += time.time() - w0
            lr_img, hr_img = (x.to(dev, non_blocking=True) for x in batch)
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
            last_l1 = logs["l1"]
            loop_t += time.time() - tick
            tick = time.time()
            if step % 10 == 0:
                el = time.time() - run_t0
                sps = el / max(step - start_step, 1)          # (average over this run, saves included)
                eta = (a.iters - step) * sps
                live.fields([f"{bar(step / a.iters, 20)} {100 * step / a.iters:5.1f}%  {step}/{a.iters}",
                             f"ETA {hms(eta)} (~{clock_in(eta)})",
                             f"{sps:.2f} s/step", f"elapsed {hms(el)}",
                             f"l1 {last_l1:.4f}"] + HW.line() + ([f"best {best:.2f} dB"] if best > -1e8 else []))
                if prog:
                    prog.set(step=step, eta=eta, speed_text=f"{sps:.2f} s/step")
                    prog.metric(**HW.metrics())
            if step % 100 == 0:
                el = time.time() - t0
                say(f"step {step}/{a.iters}  " + "  ".join(f"{k} {v / 100:.4f}" for k, v in run.items())
                    + f"  lr {lr_now:.2e}  {el / 100:.2f} s/step")
                if prog:
                    prog.point("loss (L1)", step, run["l1"] / 100)
                    prog.metric(**{k: f"{v / 100:.4f}" for k, v in run.items()})
                wf = wait_t / loop_t if loop_t else 0.0
                why = verdict(HW, wf, el / 100)
                say(f"   hardware: " + "  ".join(f"{k} {v}" for k, v in HW.metrics().items()) + f"  | data wait {100 * wf:.0f}%")
                say(f"   -> {why}")
                if prog:
                    prog.metric(**{"data wait": f"{100 * wf:.0f} %", "what limits it": why})
                    if HW.gpu and HW.gpu["watts"] is not None:
                        prog.point("GPU power (W)", step, HW.gpu["watts"])
                    if HW.gpu:
                        prog.point("GPU busy (%)", step, HW.gpu["busy"])
                    if HW.cpu is not None:
                        prog.point("CPU (%)", step, HW.cpu)
                run, t0, wait_t, loop_t = {}, time.time(), 0.0, 0.0
            if step % a.save_every == 0 or step == a.iters:
                score = psnr_on(ema, dev, root, val, a.val_crop)
                net.train()
                note = ""
                if score > best:
                    best = score; note = " (best)"
                    torch.save({"params_ema": ema.state_dict()}, out / "upscale_training_best.pth")
                save_state()
                say(f"== step {step}: held-out PSNR {score:.2f} dB{note}   (bicubic {floor:.2f}"
                    + (f", start {base:.2f}" if base is not None else "") + ")")
                if prog:
                    prog.point("held-out PSNR (dB)", step, score)
                    prog.metric(**{"best PSNR": f"{best:.2f} dB"})
    except KeyboardInterrupt:
        live.clear()
        print("\nStopping: saving where you are...", flush=True)
        save_state()
        print(f"Saved at step {step}/{a.iters}. Run the same command again to carry on from here.")
        if prog:
            prog.set(phase=f"stopped at step {step} (run the command again to resume)")
        return False
    if prog:
        prog.set(phase="finished", step=a.iters, eta=0, finished=True)
    live.clear()
    say(f"training finished in {hms(time.time() - run_t0)}")
    return True


# =============================== step 3: the model for the upscaler ===============================
def export_ncnn(pth_file, dest_dir, model_name):
    """The trained .pth as name.param / name.bin for realesrgan-ncnn-vulkan (the converter that made
    realesrgan-x2plus, kept as it was). The weights are read back strictly: a wrong file stops here."""
    sd = torch.load(pth_file, map_location="cpu", weights_only=True)   # (weights only: no pickled code runs)
    sd = sd.get("params_ema", sd.get("params", sd))
    RRDBNetX2().load_state_dict(sd, strict=True)
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
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / f"{model_name}.param").write_text("\n".join(lines) + "\n")
    (dest_dir / f"{model_name}.bin").write_bytes(bytes(binbuf))
    return len(final), len(binbuf)


def find_models_dir():
    """The models folder next to realesrgan-ncnn-vulkan, looked for from here upwards."""
    here = Path(__file__).resolve().parent
    for d in [here, *here.parents][:6]:
        if (d / "models").is_dir() and any((d / n).exists() for n in ("realesrgan-ncnn-vulkan.exe", "realesrgan-ncnn-vulkan")):
            return d / "models"
    return None


def stage_export(a, run_dir, work):
    """Step 3: the trained model as .param/.bin where dvd_upscale.py finds it."""
    best, latest = run_dir / "upscale_training_best.pth", run_dir / "upscale_training_latest.pth"
    # best = highest held-out PSNR, which only means something for the plain L1 training
    pth = best if best.exists() and not (a.gan or a.perceptual) else latest
    if not pth.exists():
        say(f"no trained model in {run_dir}: run the train step first")
        return False
    models = Path(a.models) if a.models else find_models_dir()
    if models is None:
        models = work / "models"
        say(f"couldn't find the upscaler's models folder, so the files go to {models}: copy them there by hand "
            "(or pass --models)")
    n_layers, n_bytes = export_ncnn(pth, models, a.name)
    say(f"wrote {models / (a.name + '.param')} and {a.name}.bin ({n_layers} layers, {n_bytes / 1e6:.0f} MB, from {pth.name})")
    say(f"use it:  python dvd_upscale.py <movie> --model {a.name} --scale 2")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dvd", help="the DVD movie file")
    p.add_argument("--bluray", help="the Blu-ray movie file (the same film)")
    p.add_argument("--work", help="folder for the pairs and the training (default: upscale_training_work next to the DVD file)")
    p.add_argument("--stages", default="pairs,train,export", help="which steps to run (default: pairs,train,export)")
    p.add_argument("--web", type=int, default=8643, metavar="PORT",
                   help="live progress page for a browser or phone (default port 8643; 0 turns it off)")
    g = p.add_argument_group("step 1: pairs")
    g.add_argument("--count", type=int, default=3000, help="pairs wanted (default 3000)")
    g.add_argument("--fresh", action="store_true", help="start the pairs over (deletes those already made)")
    g.add_argument("--denoise", default="2:1.5:6:5", help="hqdn3d, as dvd_upscale.py's live preset")
    g.add_argument("--dar", default=None, help="force the DVD picture aspect, e.g. 16:9 (as dvd_upscale.py --dar)")
    g.add_argument("--offset", type=float, default=None, help="one fixed Blu-ray time minus DVD time, seconds (default: tracked along the movie)")
    g.add_argument("--speed", type=float, default=1.0, help="Blu-ray seconds per DVD second (PAL disc: 1.0427)")
    g.add_argument("--search", type=float, default=120, help="seconds either side searched for the first offset")
    g.add_argument("--anchor-every", type=float, default=120, help="seconds between offset measurements")
    g.add_argument("--min-anchor", type=float, default=0.6, help="thumbnail match an offset measurement needs")
    g.add_argument("--window", type=float, default=1.5, help="seconds either side of the expected time searched for the matching Blu-ray frame")
    g.add_argument("--min-thumb", type=float, default=0.85, help="thumbnail match a Blu-ray frame needs")
    g.add_argument("--skip", type=float, default=0.05, help="fraction of the movie skipped at each end")
    g.add_argument("--color-match", choices=["low", "none"], default="low")
    g.add_argument("--min-ecc", type=float, default=0.90)
    g.add_argument("--min-ncc", type=float, default=0.93)
    g.add_argument("--max-warp", type=float, default=0.05, help="largest scale/shear the alignment may apply")
    g.add_argument("--pair-workers", type=int, default=3, help="frames made at the same time (default 3)")
    g.add_argument("--seed", type=int, default=1)
    g = p.add_argument_group("step 2: train")
    g.add_argument("--pretrained", default=str(Path(__file__).resolve().parent / "RealESRGAN_x2plus.pth"),
                   help="the starting model (default: RealESRGAN_x2plus.pth next to this script; 'none' for a test run)")
    g.add_argument("--iters", type=int, default=20000)
    g.add_argument("--batch", type=int, default=8, help="patches per step (default 8)")
    g.add_argument("--patch", type=int, default=96, help="DVD patch size in pixels (default 96)")
    g.add_argument("--lr", type=float, default=5e-5)
    g.add_argument("--perceptual", type=float, default=0.0)
    g.add_argument("--gan", type=float, default=0.0)
    g.add_argument("--checkpoint", action="store_true", help="trade speed for much less GPU memory")
    g.add_argument("--train-workers", type=int, default=2, help="data loading processes (default 2)")
    g.add_argument("--save-every", type=int, default=1000)
    g.add_argument("--val-crop", type=int, default=384, help="held-out frames are judged on this centre square (DVD px)")
    g = p.add_argument_group("step 3: export")
    g.add_argument("--name", default="upscale-training-x2", help="model name for dvd_upscale.py --model (default upscale-training-x2)")
    g.add_argument("--models", help="the upscaler's models folder (default: found next to realesrgan-ncnn-vulkan)")
    a = p.parse_args()

    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    if not stages or set(stages) - {"pairs", "train", "export"}:
        p.error("--stages is a list of: pairs, train, export")
    if "pairs" in stages and not (a.dvd and a.bluray):
        p.error("--dvd and --bluray are needed to make the pairs")
    if not a.work and not a.dvd:
        p.error("--work (or --dvd, to put it next to the DVD) is needed")
    work = Path(a.work) if a.work else Path(a.dvd).resolve().parent / "upscale_training_work"
    pairs_dir, run_dir = work / "pairs", work / "run"
    # fail now, not after hours of pairs
    if "train" in stages and a.pretrained != "none" and not Path(a.pretrained).exists():
        sys.exit(f"starting model not found: {a.pretrained}\nDownload RealESRGAN_x2plus.pth from\n{PRETRAINED_URL}\n"
                 "and put it in the same folder as upscale_training.py (or pass its path with --pretrained).")
    for f in (a.dvd, a.bluray):
        if f and "pairs" in stages and not Path(f).exists():
            sys.exit(f"file not found: {f}")

    global PROG, LIVE, HW
    LIVE = LiveLine()
    HW = HwMonitor()
    PROG = Progress("DVD to Blu-ray training", a.web).start() if a.web else None
    if "train" in stages and not torch.cuda.is_available():
        print("WARNING: no CUDA GPU found: training would take weeks on the processor")
    work.mkdir(parents=True, exist_ok=True)
    say(f"working folder: {work}")
    titles = {"pairs": "making the DVD / Blu-ray pairs", "train": "training", "export": "making the model for the upscaler"}
    for n, stage in enumerate(stages, 1):
        say(f"=== step {n} of {len(stages)}: {titles[stage]} ===")
        if PROG:
            PROG.reset(f"Step {n} of {len(stages)}: {titles[stage]}")
        if stage == "pairs":
            ok = stage_pairs(a, pairs_dir)
        elif stage == "train":
            ok = stage_train(a, pairs_dir, run_dir)
        else:
            ok = stage_export(a, run_dir, work)
        if not ok:
            return
    if PROG:
        PROG.set(phase="all done", finished=True, eta=0)
    if "export" in stages:
        say(f"All done. Upscale with:  python dvd_upscale.py <movie> --model {a.name} --scale 2")
    if PROG:
        PROG.hold()


if __name__ == "__main__":
    main()
