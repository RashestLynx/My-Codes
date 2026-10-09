"""DVD -> Blu-ray training in one command: pairs, training, and the model for dvd_upscale.py.

usage:
  python upscale_training.py --dvd movie_dvd.mkv --bluray movie_bd.mkv

It does three steps, one after the other, and tells you which one it is on:
  1. pairs   aligned DVD / Blu-ray frame pairs from the two movies        (a few hours)
  2. train   fine-tunes Real-ESRGAN x2plus on them                        (about 2 hours on an RTX 3060)
  3. export  writes the model (.param / .bin) into the upscaler's models folder (seconds)
Everything goes in a folder of its own next to the DVD file (Movie_dvd.mkv: Movie_dvd_training, with pairs and
run in it); --work puts it elsewhere. Movies sorted into folders named anime, live, cgi (or 3d) or vhs get
that type, as with dvd_upscale.py: e.g. Movies\\cgi\\Shrek_dvd.mkv and Movies\\cgi\\Shrek_bd.mkv. Movies not in
such a folder can carry the type at the start of the name instead: cgi_Shrek_dvd.mkv, anime-DBZ_dvd.mkv.

Before step 1 it asks dvd_upscale.py (found next to this script, or a folder or two up) what the DVD is:
anime, cgi (3D animation), live (live action) or vhs (a tape), the same detection it does when upscaling.
The DVD frames of the pairs are then prepared with exactly the filters it uses for that movie (inverse
telecine or deinterlacing, square pixels, that type's denoise, the VHS clean-up), so the model learns from
frames as the upscaler will hand them over. The model is named after the kind: ai-anime-x2, ai-cgi-x2,
ai-live-x2 or ai-vhs-x2 (an older one of the same name is kept as ai-...-x2-before).
--type sets the kind by hand, --name sets the model name, and --analyze only shows what it finds, then stops:
  python upscale_training.py --dvd movie_dvd.mkv --analyze

Several movies, one after another (overnight, say), with one model per kind of movie:
  python upscale_training.py --all "D:\\Movies"            every DVD + Blu-ray pair in that folder and its
                                                        anime/live/cgi/vhs folders: Shrek_dvd.mkv + Shrek_bd.mkv
  python upscale_training.py --queue                     the movies in training_queue.txt, one per line:
      --dvd "Shrek_dvd.mkv" --bluray "Shrek_bd.mkv"
      --dvd "C:\\Movies\\DBZ_dvd.mkv" --bluray "C:\\Movies\\DBZ_bd.mkv" --type anime    # a comment
  First the pairs of every movie (each in its own <name>_training folder), then ONE training per kind on the
  pairs of all the movies of that kind: ai-cgi-x2 learns from every CGI movie, ai-anime-x2 from every anime.
  Options after --all / --queue go to every movie (--count 2000, --iters 30000 ...). A queue line with --name
  gets a model of its own from that movie alone. The list is read again before each movie (add movies while it
  runs); a failed movie is logged in training_queue.log and the rest go on; Ctrl+C stops it all (run the
  same command again to carry on: finished movies and models are skipped). Movies added later: the model of
  their kind carries on from the model it already is (in the models folder), on all its movies, the earlier ones
  included, with half of each step's frames from the new movies so they are really learned (also when the queue
  lists only the new ones: a model only gains movies). The steps grow with the movies: 10000 per movie for a
  new model (20000 at least), 10000 per new movie when it carries on (--iters sets them by hand). Then the new
  model and the one in use are both judged on the held-out minutes of ALL its movies (never trained on by
  either): the new one replaces it only when it scores higher, else the one in use stays (and its movies are
  learned again with the next movie added). So the model in use only gets better, with no limit on movies.
  Delete its ai-<kind>-x2_training folder and its models (.param/.bin/.pth) to start it over from x2plus.
  It starts with a list of what it found (movies, their kind, how far each is, the models it will make, a rough
  time); --list shows just that list and stops. --shutdown turns the PC off at the end.
  --delete-originals: once a model is made, the DVD and Blu-ray files of its movies and their pairs go to the
  Recycle Bin, so only the models stay; a later movie of that kind then carries on training from the model (its
  .pth, kept next to it in the models folder).

Ctrl+C at any time, then run the same command again: it carries on where it stopped (steps already
finished are skipped; the pairs and the training both resume). --stages picks steps, e.g.
  --stages train,export        (pairs already made)       --stages export     (just convert again)

Then upscale with:
  python dvd_upscale.py <movie> --trained     (picks ai-<type>-x2 for the type it detects in that movie)

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
--perceptual 0.5 --gan 0.05 add texture, and can invent detail. It judges itself on one minute in every
20 of the movie, never trained on, and prints that PSNR next to plain bicubic and the starting model.

needs: ffmpeg + ffprobe on the PATH, pip install numpy opencv-python torch torchvision
"""
import argparse, copy, csv, datetime, http.server, json, math, random, re, socket, struct
import atexit, os, shutil, subprocess, sys, threading, time
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")      # (--gpus numbers match nvidia-smi's; set before CUDA starts)
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
.qi{display:flex;gap:8px;padding:6px 0;border-top:1px solid var(--line);font-size:14px;align-items:baseline}
.qi .st{width:16px;flex:none;text-align:center}.qi .l{overflow-wrap:anywhere}
.qi .d{color:var(--mut);margin-left:auto;text-align:right;flex:none;max-width:50%}
.qi.running{font-weight:600}.qi.done .st{color:var(--ok)}.qi.failed .st,.qi.failed .d{color:#dc2626}
</style></head><body><main>
<h1 id="title">Progress</h1><div class="sub" id="sub">connecting...</div>
<div class="card" id="queue" style="display:none"></div>
<div class="card"><div class="row"><div class="big" id="pct">-</div><div class="m"><b id="eta">-</b><span>time left</span></div>
<div class="m"><b id="el">-</b><span>elapsed</span></div><div class="m"><b id="speed">-</b><span id="speedl">speed</span></div></div>
<div class="bar" style="margin-top:10px"><div id="fill"></div></div></div>
<div class="card" id="metrics"></div><div id="charts"></div>
<div class="card"><h2>Latest</h2><pre id="log"></pre></div></main>
<script>
const $=id=>document.getElementById(id);
const esc=t=>String(t==null?'':t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const IC={done:'\u2713',running:'\u25b6',waiting:'\u00b7',failed:'\u2717',left:'\u2013'};
function queue(q){const e=$('queue');if(!q||!q.items){e.style.display='none';return}e.style.display='';
e.innerHTML='<h2>Queue: '+q.done+' of '+q.total+' done</h2><div class="row"><div class="m"><b>'+fmt(q.eta)+'</b><span>queue time left'+
(q.eta_clock?' (about '+esc(q.eta_clock)+')':'')+'</span></div><div class="m"><b>'+fmt(q.elapsed)+'</b><span>queue elapsed</span></div></div>'+
'<div class="bar" style="margin:8px 0 6px"><div style="width:'+(100*(q.frac||0))+'%"></div></div>'+
q.items.map(i=>'<div class="qi '+i.state+'"><span class="st">'+(IC[i.state]||'')+'</span><span class="l">'+esc(i.label)+
'</span><span class="d">'+esc(i.detail)+'</span></div>').join('')}
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
$('speed').textContent=s.speed_text||'-';$('el').textContent=fmt(s.elapsed);queue(s.queue);
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
                del pts[1:-1:2]            # keep it light: thin out the old points (first and newest stay)
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

    def write_to(self, path, every=2.0):
        """Instead of a page of its own: the state written to `path` every few seconds, for the page of the
        --all / --queue run that started this one (it shows the whole queue)."""
        def loop():
            while True:
                try:
                    tmp = Path(str(path) + ".tmp")
                    tmp.write_bytes(self.snapshot())
                    os.replace(tmp, path)
                except OSError:
                    pass                        # (Windows: the queue reading it just then; next time)
                time.sleep(every)
        threading.Thread(target=loop, daemon=True).start()
        return self

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

        class Server(http.server.ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                # a phone that locks its screen or a browser that drops the page cuts the connection
                # mid-request (WinError 10054): nothing is wrong, so no traceback over the status lines
                if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
                    return
                super().handle_error(request, client_address)

        try:
            srv = Server(("0.0.0.0", self.port), H)
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


def queue_parts(eta):
    """[queue 2 of 4, queue ETA, queue elapsed] for the console line of a run that --all / --queue started
    (it passes the time the rest of the queue needs and when the queue started); [] otherwise."""
    pos = os.environ.get("TRAINING_QUEUE_POS")
    if not pos:
        return []
    try:
        rest, t0 = float(os.environ.get("TRAINING_QUEUE_REST", "0")), float(os.environ.get("TRAINING_QUEUE_T0", "0"))
    except ValueError:
        return []
    left = rest + (eta or 0)
    return [f"queue ETA {hms(left)} (~{clock_in(left)})", f"elapsed {hms(time.time() - t0)}" if t0 else "",
            pos.split(":")[0]]


def clock_in(sec):
    t = datetime.datetime.now() + datetime.timedelta(seconds=max(sec, 0))
    return t.strftime("%I:%M %p").lstrip("0")


def bar(frac, width=24):
    n = int(round(max(0.0, min(1.0, frac)) * width))
    return "[" + "#" * n + "-" * (width - n) + "]"


def enable_vt():
    """Escape codes (cursor up, clear line) for a status of two lines: Windows Terminal, and the classic
    console once this switches them on (Windows 10+); False if they can't be used (then one line)."""
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-11)                          # (STD_OUTPUT_HANDLE)
        mode = ctypes.c_uint32()
        if not k32.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        return bool(mode.value & 0x4) or bool(k32.SetConsoleMode(h, mode.value | 0x4))   # (VIRTUAL_TERMINAL_PROCESSING)
    except (OSError, AttributeError):
        return False


class LiveLine:
    """A status that rewrites itself at the bottom (two lines where the console allows, else one); ordinary
    lines are printed above it."""

    def __init__(self):
        self.n, self.rows = 0, 0
        self.tty = sys.stdout.isatty()
        self.vt = enable_vt()

    def show(self, text):
        self.show_lines([text])

    def show_lines(self, lines):
        if not self.tty:
            return
        width = max(shutil_width() - 1, 20)
        lines = [ln[:width] for ln in lines if ln]
        if not self.vt or len(lines) == 1 and self.rows <= 1:
            text = lines[0] if len(lines) == 1 else "  ".join(lines)[:width]
            print("\r" + text.ljust(self.n), end="", flush=True)
            self.n, self.rows = len(text), 1
            return
        out = "\r" + "\x1b[1A" * max(self.rows - 1, 0)          # (back to the start of the status)
        out += "\n".join("\x1b[2K" + ln for ln in lines) + "\x1b[J"
        print(out, end="", flush=True)
        self.n, self.rows = max(len(ln) for ln in lines), len(lines)

    def pick(self, parts):
        """As many of the parts (most important first) as fit one line of the window."""
        width = shutil_width() - 1
        parts = [p for p in parts if p]
        text = parts[0] if parts else ""
        for p in parts[1:]:
            if len(text) + 2 + len(p) > width:
                break
            text += "  " + p
        return text

    def fields(self, parts, more=None):
        """The status: `parts` on the first line, `more` (hardware, queue) on a second one; with no room
        for two lines, one line of all of them, most important first."""
        if more and self.vt:
            self.show_lines([self.pick(parts), self.pick(more)])
        else:
            self.show(self.pick(parts[:3] + (more or [])[:1] + (more or [])[1:] + parts[3:]))

    def clear(self):
        if not (self.tty and self.n):
            return
        if self.vt and self.rows > 1:
            print("\r" + "\x1b[1A" * (self.rows - 1) + "\x1b[J", end="", flush=True)
        else:
            print("\r" + " " * self.n + "\r", end="", flush=True)
        self.n = self.rows = 0

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
              "memory.used,memory.total")
REASON_FIELDS = ("clocks_throttle_reasons.active", "clocks_event_reasons.active", None)   # (renamed in newer drivers)


def combine_gpus(rows):
    """One reading for several GPUs, for the verdict: watts add up, the load is the average, and the
    hottest / fullest / most throttled card counts (the slowest card sets the pace)."""
    known = lambda k: [r[k] for r in rows if r[k] is not None]
    avg = lambda k: sum(known(k)) / len(known(k)) if known(k) else None
    full = max(rows, key=lambda r: (r["mem"] or 0) / (r["mem_max"] or 1))
    bits = 0
    for r in rows:
        bits |= r["bits"]
    return dict(name=f"{len(rows)} GPUs", watts=sum(known("watts")) if known("watts") else None,
                watts_max=sum(known("watts_max")) if known("watts_max") else None, busy=avg("busy"),
                temp=max(known("temp")) if known("temp") else None, clock=avg("clock"), clock_max=avg("clock_max"),
                mem=full["mem"], mem_max=full["mem_max"], bits=bits,
                slowed=[n for b, n in THROTTLE_BITS if bits & b])


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
        self.reason_field = 0                          # which of REASON_FIELDS this driver understands
        self.focus, self.gpus = [0], []                # (nvidia-smi numbers of the GPUs in use, and their latest readings)
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
        while True:
            field = REASON_FIELDS[self.reason_field]
            try:
                r = subprocess.run([self.smi, "--query-gpu=" + GPU_FIELDS + (("," + field) if field else ""),
                                    "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=8, stdin=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except (OSError, subprocess.SubprocessError) as e:
                self.err = str(e)
                return
            if r.returncode == 0 and r.stdout.strip():
                break
            if self.reason_field + 1 >= len(REASON_FIELDS):      # (even the plain query fails: give up quietly)
                self.err = (r.stderr or "nvidia-smi failed").strip()[:80]
                return
            self.reason_field += 1                                # (this driver doesn't know that field: try the next)
        num = lambda x: float(x) if re.fullmatch(r"[\d.]+", x) else None          # ("[N/A]" -> None)
        rows = []
        for text in r.stdout.splitlines():
            try:
                v = [x.strip() for x in text.split(",")]
                try:
                    bits = int(v[9], 16) if len(v) > 9 else 0
                except ValueError:
                    bits = 0
                rows.append(dict(name=v[0], watts=num(v[1]), watts_max=num(v[2]), busy=num(v[3]), temp=num(v[4]),
                                 clock=num(v[5]), clock_max=num(v[6]), mem=num(v[7]), mem_max=num(v[8]),
                                 slowed=[n for b, n in THROTTLE_BITS if bits & b], bits=bits))
            except (IndexError, ValueError) as e:
                self.err = str(e)
        sel = [rows[i] for i in self.focus if i < len(rows)] or rows[:1]       # (the GPUs the training uses)
        self.gpus = sel
        self.gpu = sel[0] if len(sel) == 1 else combine_gpus(sel) if sel else None

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
        """Short pieces for the console line: GPU heat, watts, load and memory in one, then the CPU."""
        g, out = self.gpu, []
        if g:
            gb = lambda v: "n/a" if v is None else f"{v / 1024:.1f}"
            out.append(f"GPU {fv(g['temp'], '{:.0f}C')} {fv(g['watts'], '{:.0f}W')} {fv(g['busy'], '{:.0f}%')} "
                       f"mem {gb(g['mem'])}/{gb(g['mem_max'])}GB")
            if len(self.gpus) > 1:
                out.append("(" + " ".join(f"{fv(r['busy'], '{:.0f}')}%" for r in self.gpus) + ")")
        if self.cpu is not None:
            out.append(f"CPU {self.cpu:.0f}%")
        return out

    def metrics(self):
        g, m = self.gpu, {}
        if g:
            if g["watts"] is not None:
                # (laptops often don't report a power limit: then just the watts)
                m["GPU power"] = f"{g['watts']:.0f} / {g['watts_max']:.0f} W" if g["watts_max"] else f"{g['watts']:.0f} W"
            m["GPU busy"] = fv(g["busy"], "{:.0f} %")
            m["GPU temp"] = fv(g["temp"], "{:.0f} C")
            m["GPU clock"] = fv(g["clock"], "{:.0f}") + " / " + fv(g["clock_max"], "{:.0f} MHz")
            m["GPU memory"] = fv(g["mem"], "{:.0f}") + " / " + fv(g["mem_max"], "{:.0f} MiB")
            m["slowed by"] = ", ".join(g["slowed"]) or "nothing"
            if len(self.gpus) > 1:
                for i, r in zip(self.focus, self.gpus):
                    m[f"GPU {i}"] = (f"{fv(r['busy'], '{:.0f}')}% {fv(r['watts'], '{:.0f}')} W {fv(r['temp'], '{:.0f}')} C")
        if self.cpu is not None:
            m["CPU"] = f"{self.cpu:.0f} %"
        return m

    def avg(self, key, which=0):
        with self.lock:
            vals = [(h[0] or {}).get(key) if which == 0 else h[1] for h in self.hist]
        vals = [x for x in vals if x is not None]
        return sum(vals) / len(vals) if vals else None


def fv(v, fmt):
    """A GPU reading for display; the driver's "[N/A]" is shown as n/a."""
    return "n/a" if v is None else fmt.format(v)


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
        return f"SLOW: the GPU is too hot ({fv(g['temp'], '{:.0f} C')}) and lowers its clock: more airflow / fan curve."
    if recent and sum(1 for b in recent if b & 0x4) > len(recent) / 2:
        return "the GPU is at its power limit (normal when fully loaded; it is working as hard as it is allowed)."
    if recent and sum(1 for b in recent if b & 0x8) > len(recent) / 2:
        return "SLOW: hardware slowdown flagged by the GPU (power supply or cable, or heat)."
    if g["mem_max"] and g["mem"] is not None and g["mem"] / g["mem_max"] > 0.95:
        return "GPU memory is nearly full: Windows may be swapping it; try --batch 4 or --checkpoint."
    busy = hw.avg("busy")
    if busy is not None and busy < 70:
        return (f"GPU only {busy:.0f}% busy but data wait is {100 * wait_frac:.0f}%: the steps are too small for it "
                "(try --batch auto), or something else on the PC is using it.")
    return "the GPU is the limit (busy, cool, not waiting for data): this is as fast as it goes. Lower --batch/--patch or --iters to finish sooner."


# =============================== keeping a long run safe (the same habits as dvd_upscale.py) ===============================
def fsync_file(path):
    """Push a just-written file to the disk, so a power cut right after can't leave it cut short."""
    try:
        with open(path, "rb+") as f:
            os.fsync(f.fileno())
    except OSError:
        pass


def replace_file(src, dst):
    """os.replace, retried for a few seconds on Windows, where a virus scanner or the search indexer
    briefly holds a file it has just seen written."""
    for attempt in range(20):
        try:
            return os.replace(src, dst)
        except PermissionError:
            if os.name != "nt" or attempt == 19:
                raise
            time.sleep(0.5)


def save_durably(obj, path, keep_prev=False):
    """torch.save so that a crash or power cut never leaves a half-written file: a temporary file, flushed to
    disk, then renamed over the old one (which stays as name.prev with keep_prev)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    fsync_file(tmp)
    if keep_prev and path.exists():
        replace_file(path, path.with_name(path.name + ".prev"))
    replace_file(tmp, path)


def write_png_durably(path, img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise OSError(f"can't write {path}")
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(buf.tobytes())
        f.flush()
        os.fsync(f.fileno())
    replace_file(tmp, path)


def write_json_durably(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    replace_file(tmp, path)


def load_state(path, dev):
    """(state, file) from train_state.pt, or from the one before it when the newest can't be read.
    (None, None) when there is neither: a fresh start. Stops if files exist but none can be read."""
    path, found = Path(path), False
    for p in (path, path.with_name(path.name + ".prev")):
        if p.exists():
            found = True
            try:
                return torch.load(p, map_location=dev, weights_only=False), p      # (our own file)
            except Exception as e:
                say(f"{p.name} can't be read ({str(e).strip().splitlines()[0][:70] if str(e).strip() else 'damaged'}): trying the one before it")
    if found:
        sys.exit(f"the saved training in {path.parent} can't be read (both {path.name} and its backup are damaged).\n"
                 "Move that folder away to start the training over (the pairs are kept if they are in --work).")
    return None, None


def check_disk(folder, need, what):
    """A warning, not a stop, when the drive of `folder` has less than `need` bytes free."""
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(folder).free
    except OSError:
        return                                            # (a share that can't say how much is free)
    if free < need:
        say(f"WARNING: the drive of {folder} has {free / 1e9:.1f} GB free; {what} needs about {need / 1e9:.1f} GB there.")


def keep_awake():
    """CPU/GPU work doesn't count as activity: keep the computer from sleeping while this runs (released
    when it exits; closing a laptop's lid still sleeps). On Windows also run above normal priority, so a
    browser or antivirus scan doesn't take the processor first and leave the GPU waiting for data."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        try:
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass(ctypes.c_void_p(k32.GetCurrentProcess()), 0x00008000)      # (the loader processes inherit it)
        except (OSError, AttributeError):
            pass
        return
    pid = str(os.getpid())
    for cmd in (["caffeinate", "-i", "-w", pid],
                ["systemd-inhibit", "--what=sleep:idle", "--who=upscale_training.py",
                 "--why=training an upscaling model", "tail", f"--pid={pid}", "-f", "/dev/null"]):
        if shutil.which(cmd[0]):
            try:
                subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                pass
            return


# ---- keeping a long run safe: stray clicks and keys, shutdown (Windows) — the same as dvd_upscale.py ----
# A night's run is lost to small things: a click in the window (QuickEdit mode freezes the program until a key
# is pressed), keys typed while it runs (PowerShell runs them as a command once the script ends), a shutdown or
# a Windows Update restart. protect_run guards against them while it runs; all of it is undone when the script
# ends. --no-guard turns it off (sleep is still prevented). Ctrl+C still stops the run, on purpose.
_NO_WINDOW = ({"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)} if os.name == "nt" else {})
_GUARD = {"console": None, "window": None, "thread": None, "reason": "", "proc": None, "movie": ""}
WM_APP_REASON = 0x8000 + 1          # (WM_APP + 1: the reason text changed)


def protect_run(a, what):
    """Once the run really starts: no freeze from a click, no stray keys, no shutdown or restart (Windows
    asks first and shows `what`). A run started by --all / --queue leaves the shutdown block to the queue."""
    if os.name != "nt" or getattr(a, "no_guard", False):
        return
    console_guard()
    if not os.environ.get("TRAINING_QUEUE_POS"):
        block_shutdown(f"upscale_training.py is {what}: shutting down now stops it (it carries on from "
                       "where it was when run again)")
        run_warnings()


def console_guard():
    """QuickEdit and Insert mode off for this console while the run lasts (a click or a
    selection in the window then can't freeze the run), and keys typed meanwhile thrown away at
    the end (else PowerShell would run them as a command). The console's own mode comes back
    when the script ends."""
    if _GUARD["console"] is not None:
        return
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetStdHandle.restype = wintypes.HANDLE
        k32.GetStdHandle.argtypes = [wintypes.DWORD]
        k32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.FlushConsoleInputBuffer.argtypes = [wintypes.HANDLE]
        h = k32.GetStdHandle(wintypes.DWORD(-10 & 0xFFFFFFFF))       # STD_INPUT_HANDLE
        mode = wintypes.DWORD()
        if not h or not k32.GetConsoleMode(h, ctypes.byref(mode)):
            return                          # (no console: started from a GUI or a pipe)
        old = mode.value
        ENABLE_INSERT_MODE, ENABLE_QUICK_EDIT_MODE, ENABLE_EXTENDED_FLAGS = 0x20, 0x40, 0x80
        if k32.SetConsoleMode(h, (old | ENABLE_EXTENDED_FLAGS)
                              & ~(ENABLE_QUICK_EDIT_MODE | ENABLE_INSERT_MODE)):
            _GUARD["console"] = (k32, h, old)
            atexit.register(console_unguard)
    except (OSError, AttributeError, ValueError):
        pass


def console_unguard():
    g = _GUARD["console"]
    if g is None:
        return
    k32, h, old = g
    _GUARD["console"] = None
    try:
        k32.FlushConsoleInputBuffer(h)      # (keys typed during the run: not run as a command)
        k32.SetConsoleMode(h, old)
    except (OSError, AttributeError, ValueError):
        pass


def block_shutdown(reason):
    """Windows asks every program before a shutdown or restart: this one says no while it runs,
    so Windows shows "This app is preventing shutdown" with the reason (and the latest progress)
    and lets the user choose. (A forced shutdown, such as shutdown /f, a power cut or a held power
    button, can't be stopped by any program.) A hidden window of its own, in its own thread: the
    console window belongs to the console, not to this script."""
    if _GUARD["window"] is not None or _GUARD["thread"] is not None:
        return
    _GUARD["reason"] = reason
    started = threading.Event()

    def window_thread():
        try:
            import ctypes
            from ctypes import wintypes
            u32 = ctypes.WinDLL("user32", use_last_error=True)
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            LRESULT = ctypes.c_ssize_t
            WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT,
                                         wintypes.WPARAM, wintypes.LPARAM)

            class WNDCLASSW(ctypes.Structure):
                _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
                            ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                            ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                            ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                            ("lpszMenuName", wintypes.LPCWSTR),
                            ("lpszClassName", wintypes.LPCWSTR)]
            u32.DefWindowProcW.restype = LRESULT
            u32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM,
                                           wintypes.LPARAM]
            u32.RegisterClassW.restype = wintypes.ATOM
            u32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
            u32.CreateWindowExW.restype = wintypes.HWND
            u32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
                                            wintypes.DWORD, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                            wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
            u32.ShutdownBlockReasonCreate.restype = wintypes.BOOL
            u32.ShutdownBlockReasonCreate.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
            u32.ShutdownBlockReasonDestroy.restype = wintypes.BOOL
            u32.ShutdownBlockReasonDestroy.argtypes = [wintypes.HWND]
            u32.GetMessageW.restype = wintypes.BOOL
            u32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                        wintypes.UINT, wintypes.UINT]
            u32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
            u32.DispatchMessageW.restype = LRESULT
            u32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
            u32.DestroyWindow.argtypes = [wintypes.HWND]
            u32.PostQuitMessage.argtypes = [ctypes.c_int]
            k32.GetModuleHandleW.restype = wintypes.HMODULE
            k32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
            k32.SetProcessShutdownParameters.argtypes = [wintypes.DWORD, wintypes.DWORD]
            WM_DESTROY, WM_CLOSE, WM_QUERYENDSESSION, WM_ENDSESSION = 0x2, 0x10, 0x11, 0x16

            def proc(hwnd, msg, wparam, lparam):
                try:
                    if msg == WM_QUERYENDSESSION:
                        return 0                    # (FALSE: not now)
                    if msg == WM_ENDSESSION:
                        return 0
                    if msg == WM_APP_REASON:
                        u32.ShutdownBlockReasonCreate(hwnd, _GUARD["reason"][:250])
                        return 0
                    if msg == WM_CLOSE:
                        u32.DestroyWindow(hwnd)
                        return 0
                    if msg == WM_DESTROY:
                        u32.ShutdownBlockReasonDestroy(hwnd)
                        u32.PostQuitMessage(0)
                        return 0
                except Exception:
                    pass
                return u32.DefWindowProcW(hwnd, msg, wparam, lparam)
            _GUARD["proc"] = WNDPROC(proc)      # (kept: Windows calls it as long as it lives)
            inst = k32.GetModuleHandleW(None)
            wc = WNDCLASSW()
            wc.lpfnWndProc = _GUARD["proc"]
            wc.hInstance = inst
            wc.lpszClassName = "upscale_training_shutdown_guard"
            u32.RegisterClassW(ctypes.byref(wc))       # (0 if already registered: fine)
            # a top-level window that is never shown: only those are asked before a shutdown
            hwnd = u32.CreateWindowExW(0, wc.lpszClassName, "upscale_training.py", 0, 0, 0, 0, 0,
                                       None, None, inst, None)
            if not hwnd:
                return
            # asked among the first, before programs that close themselves without asking
            k32.SetProcessShutdownParameters(0x3FF, 0)
            u32.ShutdownBlockReasonCreate(hwnd, _GUARD["reason"][:250])
            _GUARD["window"] = hwnd
            started.set()
            msg = wintypes.MSG()
            while u32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                u32.TranslateMessage(ctypes.byref(msg))
                u32.DispatchMessageW(ctypes.byref(msg))
        except Exception:
            pass
        finally:
            _GUARD["window"] = None
            started.set()

    t = threading.Thread(target=window_thread, name="shutdown guard", daemon=True)
    _GUARD["thread"] = t
    t.start()
    started.wait(5)
    atexit.register(allow_shutdown)


def _post_to_guard(msg):
    hwnd = _GUARD["window"]
    if hwnd:
        try:
            import ctypes
            from ctypes import wintypes
            u32 = ctypes.WinDLL("user32", use_last_error=True)
            u32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM,
                                         wintypes.LPARAM]
            u32.PostMessageW(hwnd, msg, 0, 0)
        except (OSError, AttributeError, ValueError):
            pass


def shutdown_reason(text):
    """The latest progress, shown on Windows' "preventing shutdown" screen."""
    if _GUARD["window"] and text and text != _GUARD["reason"]:
        _GUARD["reason"] = text
        _post_to_guard(WM_APP_REASON)


def allow_shutdown():
    """Lets Windows shut down again (before --shutdown, and when the script ends)."""
    t = _GUARD["thread"]
    if t is None:
        return
    _post_to_guard(0x10)                    # (WM_CLOSE: the window goes, the block with it)
    t.join(5)
    _GUARD["thread"] = None


def run_warnings():
    """Once, at the start: what could still stop a long run that the script can't prevent."""
    try:
        import ctypes
        from ctypes import wintypes

        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte),
                        ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", wintypes.DWORD),
                        ("BatteryFullLifeTime", wintypes.DWORD)]
        st = SYSTEM_POWER_STATUS()
        k32 = ctypes.WinDLL("kernel32")
        k32.GetSystemPowerStatus.argtypes = [ctypes.POINTER(SYSTEM_POWER_STATUS)]
        on_battery = bool(k32.GetSystemPowerStatus(ctypes.byref(st))) and \
            st.ACLineStatus == 0 and st.BatteryFlag != 128      # (128: no battery)
        if on_battery:
            print("WARNING: the laptop is running on battery: plug it in (the GPU slows down a "
                  "lot on battery, and the run stops when the battery runs out)", flush=True)
    except (OSError, AttributeError, ValueError):
        on_battery = False
    # closing the lid sleeps the laptop whatever a program asks: say so if that is set
    try:
        out = subprocess.run(["powercfg", "/query", "SCHEME_CURRENT", "SUB_BUTTONS", "LIDACTION"],
                             capture_output=True, stdin=subprocess.DEVNULL, timeout=15,
                             **_NO_WINDOW).stdout.decode("utf-8", "replace")
        # (the last two numbers are the plugged-in and the battery setting, in any language)
        vals = [int(x, 16) for x in re.findall(r"0x([0-9a-fA-F]{8})", out)][-2:]
        if len(vals) == 2 and vals[1 if on_battery else 0] != 0:
            print("NOTE: closing the laptop's lid puts it to sleep, which stops the run (it "
                  "resumes when run again). Keep the lid open, or set Control Panel > Power "
                  "Options > \"Choose what closing the lid does\" to \"Do nothing\" when plugged "
                  "in (and keep the laptop out of a bag: it runs hot)", flush=True)
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    # Windows Update with a restart waiting can restart the PC in the night (outside the active
    # hours), and that kind of restart doesn't wait for programs
    try:
        import winreg
        winreg.CloseKey(winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion"
            r"\WindowsUpdate\Auto Update\RebootRequired"))
        print("WARNING: Windows Update is waiting to restart the PC and may do it during this "
              "run. Restart first, or pause updates (Settings > Windows Update > Pause updates) "
              "for a long run.", flush=True)
    except (OSError, ImportError):
        pass


def hold_lock(path, message):
    """Take a lock file, or exit with message if another run holds it. The OS releases the lock by itself when
    this run exits, crashes or the PC shuts down, so it never needs clearing. Keep the returned file open."""
    f = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(message)
    return f


SETTINGS_KEYS = ("arch", "compact_feat", "compact_convs", "patch", "crops", "batch", "lr", "iters", "perceptual",
                 "gan", "disc", "rotate", "pretrained", "min_patch_match")


def note_settings(a, run_dir, resuming):
    """Remember the training settings; when resuming, say which ones differ from the earlier run."""
    now = {k: str(getattr(a, k)) for k in SETTINGS_KEYS}
    f = Path(run_dir) / "settings.json"
    if resuming:
        try:
            old = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        diff = [f"--{k.replace('_', '-')} {old[k]} -> {now[k]}" for k in SETTINGS_KEYS if k in old and old[k] != now[k]]
        if diff:
            say("NOTE: resuming with different settings than the earlier run: " + ";  ".join(diff))
    write_json_durably(f, now)


# =============================== the kind of movie and the upscaler's filters, from dvd_upscale.py ===============================
# dvd_upscale.py decides what a movie is (anime, cgi, live, vhs) and how its frames are prepared before the AI
# (inverse telecine or deinterlacing, square pixels, the denoise of that type, the VHS clean-up). The pairs are
# made from DVD frames prepared exactly that way, so the model learns from what it will be given; and the
# model is named after the kind, which dvd_upscale.py --trained picks by the same detection.
TYPE_NAMES = {"anime": "anime / drawn animation", "cgi": "3D animation (CGI)", "live": "live action",
              "vhs": "VHS tape"}
TYPE_DENOISE = {"anime": "2:1.5:3:2.5", "live": "2:1.5:6:5", "cgi": "2:1.5:6:5", "vhs": "4:8:9:14"}   # (its presets)


def default_work(a):
    """Each movie's own work folder: <DVD name>_training next to it (movies sharing a folder, like
    Movies\\cgi, must not share one). A training started by an older version in the shared
    upscale_training_work folder next to it carries on there, unless that one is another movie's."""
    if a.work:
        return Path(a.work)
    dvd = Path(a.dvd).resolve()
    old = dvd.parent / "upscale_training_work"
    if old.is_dir():
        try:
            mine = json.loads((old / "upscaler.json").read_text(encoding="utf-8"))["key"]["dvd"] == dvd.name
        except (OSError, ValueError, KeyError, TypeError):
            mine = not (dvd.parent / f"{dvd.stem}_training").exists()      # (older versions: no record of the movie)
        if mine:
            return old
    return dvd.parent / f"{dvd.stem}_training"


TYPE_DIRS = {"live": "live", "live action": "live", "live-action": "live", "anime": "anime",
             "cgi": "cgi", "3d": "cgi", "vhs": "vhs"}            # (dvd_upscale.py's type folders)


def type_from_place(path):
    """The type a movie's folder (Movies\\cgi\\...) or, failing that, its name's tag (cgi_Shrek_dvd.mkv)
    gives it, as dvd_upscale.py reads them; None if neither."""
    p = Path(path).resolve()
    if p.parent.name.lower() in TYPE_DIRS:
        return TYPE_DIRS[p.parent.name.lower()]
    m = re.match(r"(anime|live|cgi|3d|vhs)[_-]", p.name, re.I)
    return TYPE_DIRS[m.group(1).lower()] if m else None


def find_upscaler():
    """dvd_upscale.py: next to this script, or a folder or two up (training/ inside the upscaler's folder)."""
    here = Path(__file__).resolve().parent
    for d in [here, *here.parents][:3]:
        if (d / "dvd_upscale.py").is_file():
            return d / "dvd_upscale.py"
    return None


def upscaler_settings(a, work):
    """{type, mode, filters, ...} as dvd_upscale.py --analyze finds them for --dvd, kept in work/upscaler.json
    (the same movie keeps them, so resumed pairs are made the same way). None if it can't be asked."""
    saved = Path(work) / "upscaler.json"
    dvd = Path(a.dvd)
    key = dict(dvd=dvd.name, folder=dvd.resolve().parent.name.lower(), size=dvd.stat().st_size, type=a.type, dar=a.dar)
    try:
        old = json.loads(saved.read_text(encoding="utf-8"))
        if old.get("key") == key and old.get("filters"):
            return old
    except (OSError, ValueError):
        pass
    up = find_upscaler()
    if up is None:
        return None
    report = Path(work) / "upscaler_report.json"
    report.unlink(missing_ok=True)
    cmd = [sys.executable, str(up), str(dvd), "--analyze", "--work", str(Path(work) / "upscaler_check")]
    cmd += (["--type", a.type] if a.type != "auto" else []) + (["--dar", a.dar] if a.dar else [])
    say(f"asking {up.name} what this movie is and how it prepares its frames (--analyze, a minute or two)...")
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace",
                       env=dict(os.environ, DVD_UPSCALE_REPORT=str(report)))
    for line in r.stdout.splitlines():
        if re.match(r"\s*(Type|Tape check|Detected|Mode|  If)", line):
            say("  " + line.strip())                      # (its verdict and reasons, as it prints them)
    try:
        got = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        say(f"WARNING: {up.name} --analyze didn't report (exit code {r.returncode}): " + " | ".join(tail))
        return None
    if not got.get("filters"):
        say(f"WARNING: this {up.name} doesn't report its filters (an older version): update it")
        return None
    got["key"] = key
    write_json_durably(saved, got)
    return got


def setup_kind(a, work, need_filters):
    """a.kind (anime/cgi/live/vhs or None) and a.dvd_filters (the upscaler's, or None)."""
    a.kind, a.dvd_filters = (None if a.type == "auto" else a.type), None
    how = "--type" if a.kind else "found earlier"
    if a.dvd and Path(a.dvd).exists():
        got = upscaler_settings(a, work)
        if got:
            a.kind, a.dvd_filters = got["type"], got["filters"]
            say(f"movie type: {TYPE_NAMES.get(a.kind, a.kind)} ({got['mode']}); the pairs use dvd_upscale.py's filters: "
                + ",".join(a.dvd_filters))
            if a.kind == "vhs":
                say("NOTE: a tape and a Blu-ray rarely show the same picture area (tapes are mostly 4:3 pan and "
                    "scan): if most pairs are rejected, that's why")
            return
    else:
        try:                                              # (train/export without --dvd: as found before)
            a.kind = a.kind or json.loads((Path(work) / "upscaler.json").read_text(encoding="utf-8"))["type"]
        except (OSError, ValueError, KeyError):
            pass
    if a.kind is None and a.dvd and Path(a.dvd).exists():
        a.kind = type_from_place(a.dvd)                   # (dvd_upscale.py didn't answer: its folder / tag rules)
        how = "from its folder or name tag"
    if need_filters:
        if a.kind is None:
            sys.exit("dvd_upscale.py wasn't found (or couldn't say what the movie is) next to this script or a folder "
                     "or two up, so the DVD frames can't be prepared the way it prepares them. Put this script "
                     "in the upscaler's folder (or a 'training' folder inside it), or pass --type anime/cgi/live/vhs.")
        say(f"WARNING: making the pairs without dvd_upscale.py's own filters (it wasn't found or didn't answer): "
            f"a plain deinterlace and the {a.kind} denoise. Film DVDs (inverse telecine) and tapes will differ "
            "a little from what the upscaler gives the model.")
    if a.kind:
        say(f"movie type: {TYPE_NAMES.get(a.kind, a.kind)} ({how})")


def model_name(a, kind):
    return a.name or (f"ai-{kind}-x2" if kind else "upscale-training-x2")


# =============================== step 1: the DVD / Blu-ray pairs ===============================
THUMB = (64, 36)
PRE_ROLL = 1.0              # seconds decoded before each DVD frame so the denoiser has warmed up (see make_pair)
PROG = None                # the live progress page (--web), when asked for


LIVE = None                 # the status line in the console
HW = None                   # the hardware monitor (GPU watts, load, heat, CPU)
LOCK = None                 # the open lock file of the work folder (kept open while running)


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
    """The filters dvd_upscale.py applies before the upscaler, for one frame: its own list for this movie
    (setup_kind), with --denoise put in when given; else a plain stand-in with the type's denoise."""
    if a.dvd_filters:
        f = list(a.dvd_filters)
        if a.denoise:
            f = [f"hqdn3d={a.denoise}" if x.startswith("hqdn3d=") else x for x in f]
        return f
    f = ["yadif=deint=interlaced"]                  # (only frames the decoder flags as combed)
    if a.dar:
        n, d = (float(x) for x in a.dar.replace("/", ":").split(":"))
        f.append(f"scale=trunc(ih*{n / d}/2)*2:ih:flags=lanczos")
    else:
        sn, sd = (float(x) for x in (sar if ":" in sar and sar != "0:1" else "1:1").split(":"))
        f.append(f"scale=trunc(iw*{sn / sd}/2)*2:ih:flags=lanczos")
    f += ["setsar=1", f"hqdn3d={a.denoise or TYPE_DENOISE.get(a.kind, TYPE_DENOISE['live'])}"]
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
    i, t_start = 0, time.time()
    while t < end:
        i += 1
        off, sc = anchor_offset(a, t, guess, span, dvd_pre, bd_pre)
        ok = off is not None and sc >= a.min_anchor
        say(f"  offset {i}/{n_total} at {t / 60:5.1f} min: "
            + (f"Blu-ray {off:+7.2f} s, match {sc:.3f}" if ok else f"no clear match ({sc:.2f})"))
        el = time.time() - t_start
        eta = el / i * (n_total - i)
        LIVE.fields([f"{bar(i / n_total, 16)} offsets {i}/{n_total}", f"ETA {hms(eta)} (~{clock_in(eta)})",
                     f"elapsed {hms(el)}"], (HW.line() if HW else [])[:1] + queue_parts(eta) + (HW.line() if HW else [])[1:])
        if PROG:
            PROG.set(phase="measuring the offset between the discs", step=i, total=n_total,
                     unit="offset measurements", eta=eta, elapsed=el, speed_text=f"{el / i:.1f} s each")
            PROG.metric(**(HW.metrics() if HW else {}))
        if ok:
            anchors.append((t, off)); guess, span = off, 40
        else:
            span = min(span * 2, a.search)              # lost: look wider next time
        t += a.anchor_every
    LIVE.clear()
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
    # hqdn3d (and yadif) remember the frames before: on the first frame after a seek they do nothing, while
    # dvd_upscale.py plays the movie straight through. So start a moment early and keep the frame at dvd_t,
    # so the DVD frame is denoised exactly as it will be when upscaling.
    warm = min(PRE_ROLL, dvd_t)
    lr = read_frames(["ffmpeg", "-v", "error", "-ss", f"{dvd_t - warm:.3f}", "-i", str(a.dvd),
                      "-vf", ",".join(dvd_pre + c(dvd_crop) + [f"trim=start={warm:.3f}",
                                                               f"scale={lrw}:{lrh}:flags=lanczos", "format=bgr24"]),
                      "-frames:v", "1", "-f", "rawvideo", "-"], lrw, lrh, 3)
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
        (out / "measured.json").unlink(missing_ok=True)
        for sub in ("lr", "hr"):
            for f in (out / sub).glob("*.png"):
                f.unlink()
        if csv_path.exists():
            csv_path.unlink()
        (out / "rejected.txt").unlink(missing_ok=True)
    done_times, have = set(), 0
    if csv_path.exists():
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                done_times.add(round(float(r["t_dvd"]), 3))
                have = max(have, int(r["name"]))
    dw, dh, dsar, ddur, _ = probe(a.dvd)
    dvd_pre = dvd_chain(a, dw, dh, dsar)
    # the DVD frames of the pairs already made must have been prepared the same way as the new ones
    made_with = out / "dvd_filters.json"
    try:
        before = json.loads(made_with.read_text(encoding="utf-8"))["filters"]
    except (OSError, ValueError, KeyError):
        before = None
    if have and before is None:
        say("WARNING: these pairs were made by an older version of this script, whose DVD frames weren't prepared "
            "the way dvd_upscale.py prepares them (no denoise, no inverse telecine). Run with --fresh to make "
            "them again: the model will match the upscaler much better.")
    elif have and before != dvd_pre:
        say("WARNING: these pairs were made with other DVD filters (" + ",".join(before) + ") than now ("
            + ",".join(dvd_pre) + "). Run with --fresh so they all match the upscaler.")
    if not have:
        write_json_durably(made_with, {"filters": dvd_pre})
    if have:
        say(f"found {have} pairs already in {out}: carrying on from there (--fresh starts over)")
        if have >= a.count:
            say(f"already {have} pairs, which is --count {a.count} or more: nothing to do")
            return True
    bw, bh, _, bdur, bd_fps = probe(a.bluray)
    print(f"DVD {dw}x{dh} sar {dsar}, {ddur / 60:.1f} min;  Blu-ray {bw}x{bh}, {bdur / 60:.1f} min")
    # the picture areas and the offsets between the discs are measured once (minutes) and kept for a
    # resumed run, as long as it's the same two files with the same settings
    measured_file = out / "measured.json"
    key = dict(dvd=Path(a.dvd).name, dvd_size=Path(a.dvd).stat().st_size, bd=Path(a.bluray).name,
               bd_size=Path(a.bluray).stat().st_size, filters=dvd_pre, speed=a.speed, offset=a.offset, skip=a.skip,
               search=a.search, anchor_every=a.anchor_every, min_anchor=a.min_anchor)
    try:
        measured = json.loads(measured_file.read_text(encoding="utf-8"))
        if measured.get("key") != key:
            measured = {}
    except (OSError, ValueError):
        measured = {}
    if measured.get("dvd_crop"):
        dvd_crop, bd_crop = tuple(measured["dvd_crop"]), measured.get("bd_crop") and tuple(measured["bd_crop"])
    else:
        t_probe = ddur * 0.3
        dvd_crop = find_crop(a.dvd, dvd_pre, t_probe)
        bd_crop = find_crop(a.bluray, [], t_probe * a.speed)
    print(f"picture area: DVD (after square pixels) {dvd_crop}, Blu-ray {bd_crop}"
          + (" (measured before)" if measured.get("dvd_crop") else ""))
    if not dvd_crop:
        sys.exit("cropdetect found nothing on the DVD")
    lrw, lrh = dvd_crop[0], dvd_crop[1]
    if bd_crop and abs(bd_crop[0] / bd_crop[1] / (lrw / lrh) - 1) > 0.02:
        print(f"WARNING: picture aspect differs: DVD {lrw / lrh:.3f}, Blu-ray {bd_crop[0] / bd_crop[1]:.3f} "
              "(a re-framed transfer: pairs will be rejected or distorted)")
    lrw -= lrw % 2
    lrh -= lrh % 2
    check_disk(out, int((a.count - have) * lrw * lrh * 3 * 5 * 0.55 * 1.2), f"{a.count} pairs")

    cropped_bd_pre = ([f"crop={bd_crop[0]}:{bd_crop[1]}:{bd_crop[2]}:{bd_crop[3]}"] if bd_crop else [])
    cropped_dvd_pre = dvd_pre + [f"crop={dvd_crop[0]}:{dvd_crop[1]}:{dvd_crop[2]}:{dvd_crop[3]}"]
    if measured.get("anchors"):
        anchors = [tuple(x) for x in measured["anchors"]]
        say(f"offsets between the discs: measured before ({len(anchors)} measurements), not again")
    else:
        print("measuring the offset between the discs along the movie (it can change: extra logos, scenes)...")
        anchors = build_offsets(a, cropped_dvd_pre, cropped_bd_pre, ddur)
        write_json_durably(measured_file, dict(key=key, dvd_crop=list(dvd_crop),
                                               bd_crop=list(bd_crop) if bd_crop else None, anchors=anchors))
    offs = [o for _, o in anchors]
    print(f"offsets found: {min(offs):+.1f} to {max(offs):+.1f} s ({len(anchors)} measurements)")

    lo, hi = ddur * a.skip, ddur * (1 - a.skip)
    # more candidates than wanted: some are rejected. Jittered, evenly spread over the movie. If a movie has many
    # unusable frames (dark, flat), more rounds of candidates at other spots, until --count is reached
    n_try = int(a.count * 1.6)
    times = []
    for rnd in range(4):
        rng = random.Random(a.seed if rnd == 0 else a.seed * 1000 + rnd)     # (round 0 as in earlier versions)
        batch = [lo + (hi - lo) * (i + rng.random()) / n_try for i in range(n_try)]
        rng.shuffle(batch)
        times += batch
    # frames rejected before (in an earlier, stopped run) aren't tried again: same frame, same verdict
    rejected_file = out / "rejected.txt"
    tried = set(done_times)
    # (a verdict is only valid for the settings and filters it was made with: the first line records them)
    rej_key = json.dumps(dict(filters=dvd_pre, min_thumb=a.min_thumb, min_ecc=a.min_ecc, min_ncc=a.min_ncc,
                              max_warp=a.max_warp, window=a.window, speed=a.speed, color_match=a.color_match),
                         sort_keys=True)
    same_rej = False
    try:
        rej_lines = rejected_file.read_text(encoding="utf-8").splitlines()
        same_rej = bool(rej_lines) and rej_lines[0] == "# " + rej_key
        if same_rej:
            tried |= {round(float(line.split()[0]), 3) for line in rej_lines[1:] if line.strip()}
    except (OSError, ValueError, IndexError):
        same_rej = False
    skipped_before = len([t for t in times if round(t, 3) in tried and round(t, 3) not in done_times])
    if skipped_before:
        say(f"  {skipped_before} frames were rejected before (dark, flat, no match): not tried again")
    times = [t for t in times if round(t, 3) not in tried]
    rej_fh = open(rejected_file, "a" if same_rej else "w", encoding="utf-8")
    if not same_rej:
        rej_fh.write("# " + rej_key + "\n")
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
        futs_t = dict(zip(futs, times))
        try:
            for f in as_completed(futs):
                lr, hr, info = f.result()
                if lr is None:
                    rejects[info.split()[0]] += 1
                    if not info.startswith("error"):           # (an ffmpeg hiccup may work next time)
                        rej_fh.write(f"{futs_t[f]:.3f} {info.split()[0]}\n")
                    continue
                done += 1
                name = f"{done:06d}"
                write_png_durably(out / "lr" / f"{name}.png", lr)
                write_png_durably(out / "hr" / f"{name}.png", hr)
                wr.writerow([name, info["t_dvd"], info["t_bd"], info["ecc"], info["ncc"], info["sharp_ratio"]])
                fh.flush()
                os.fsync(fh.fileno())
                if done % 25 == 0:
                    say(f"  {done}/{a.count} pairs  (rejected: {dict(rejects)})")
                el = time.time() - t_start
                new = done - have                       # (made in this run: the speed and ETA are from these)
                eta = el / new * (a.count - done)
                LIVE.fields([f"{bar(done / a.count, 16)} {100 * done / a.count:5.1f}%  {done}/{a.count} pairs",
                             f"ETA {hms(eta)} (~{clock_in(eta)})", f"elapsed {hms(el)}",
                             f"{new / el * 60:.1f} pairs/min", f"rejected {sum(rejects.values())}"],
                            (HW.line() if HW else [])[:1] + queue_parts(eta) + (HW.line() if HW else [])[1:])
                if PROG:
                    PROG.set(step=done, eta=eta, speed_text=f"{new / el * 60:.1f} pairs/min", pairs_per_min=new / el * 60,
                             elapsed=el)
                    PROG.metric(rejected=sum(rejects.values()), **(HW.metrics() if HW else {}))
                if done >= a.count:
                    break
        except KeyboardInterrupt:
            stopped = True
            LIVE.clear()
            print("\nStopping (finishing the frames in progress)...", flush=True)
        finally:
            for f in futs:
                f.cancel()
    rej_fh.close()
    LIVE.clear()
    if stopped:
        a.stopped = True
        say(f"Stopped with {done} pairs saved in {out}. Run the same command again to carry on from here.")
        if PROG:
            PROG.set(phase=f"stopped at {done} pairs (run the command again to resume)")
        return False
    say(f"done: {done} pairs in {out}  (rejected: {dict(rejects)})")
    if PROG:
        PROG.set(phase="finished", step=done, eta=0, finished=True)
    if done < a.count // 2:
        say("few pairs survived: check the picture areas above, or lower --min-thumb/--min-ecc/--min-ncc")
    elif done < a.count:
        say(f"({done} of {a.count}: every spot tried; the rest of this movie is too dark or flat to learn from. "
            "That's enough for the training)")
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


class SRVGGNetCompactX2(nn.Module):
    """Real-ESRGAN's small, fast 'compact' network (realesr-animevideov3 / general-x4v3 style) as a 2x:
    plain convs + PReLU, a pixel-shuffle at the end and a nearest-neighbour copy of the input added back.
    Several times faster than RRDBNetX2, but there is no 2x starting model: it trains from random."""

    def __init__(s, nf=64, nconv=16):
        super().__init__()
        s.body = nn.ModuleList([nn.Conv2d(3, nf, 3, 1, 1), nn.PReLU(num_parameters=nf)])
        for _ in range(nconv):
            s.body += [nn.Conv2d(nf, nf, 3, 1, 1), nn.PReLU(num_parameters=nf)]
        s.body.append(nn.Conv2d(nf, 3 * 4, 3, 1, 1))
        s.use_checkpoint = False                          # (kept so both networks take the same switch)

    def forward(s, x):
        out = x
        for layer in s.body:
            out = layer(out)
        return F.pixel_shuffle(out, 2) + F.interpolate(x, scale_factor=2, mode="nearest")


def arch_of(sd):
    """('rrdb', {}) or ('compact', {nf, nconv}) from the names in a state dict."""
    if "conv_first.weight" in sd:
        return "rrdb", {}
    if "body.0.weight" in sd:
        last = max(int(k.split(".")[1]) for k in sd if k.startswith("body.") and k.endswith(".weight"))
        return "compact", dict(nf=sd["body.0.weight"].shape[0], nconv=(last - 2) // 2)
    raise ValueError("unknown network: neither RRDBNetX2 nor SRVGGNetCompactX2 weights")


def make_net(arch, nf=64, nconv=16):
    return SRVGGNetCompactX2(nf, nconv) if arch == "compact" else RRDBNetX2()


def load_weights(net, path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(sd.get("params_ema", sd.get("params", sd)), strict=True)


# ---- data ----
class Pairs(torch.utils.data.Dataset):
    """Each item: `crops` random patches of one random pair -> (lr [n,3,p,p], hr [n,3,2p,2p]).
    items: (pairs folder, name), from one movie or several."""

    def __init__(s, items, patch, crops, rotate=False, min_match=0.0, focus=(), focus_share=0.5):
        s.items, s.p, s.n, s.rotate, s.min_match = items, patch, crops, rotate, min_match
        # (focus: the new movies' pairs, drawn `focus_share` of the time, the rest from all of them: among many
        #  movies a new one would otherwise get only its small share of the steps and hardly be learned)
        s.focus = list(focus) if focus and len(focus) < len(items) * focus_share else []
        s.focus_share = focus_share

    def __len__(s):
        return 10 ** 7

    def __getitem__(s, _):
        for _ in range(10):
            root, name = random.choice(s.focus if s.focus and random.random() < s.focus_share else s.items)
            lr = cv2.imread(str(Path(root) / "lr" / f"{name}.png"))
            hr = cv2.imread(str(Path(root) / "hr" / f"{name}.png"))
            if lr is not None and hr is not None and hr.shape[0] == 2 * lr.shape[0] and hr.shape[1] == 2 * lr.shape[1]:
                break
        else:
            raise RuntimeError(f"can't read a valid pair from {root} (10 tries)")
        h, w = lr.shape[:2]
        p, lrs, hrs = s.p, [], []
        for _ in range(s.n):
            y, x = pick_spot(lr, hr, p, s.min_match)
            a, b = lr[y:y + p, x:x + p], hr[2 * y:2 * (y + p), 2 * x:2 * (x + p)]
            # flips are safe. 90-degree turns are off by default: a DVD is stretched sideways (anamorphic),
            # so its blur and artifacts have a direction the model should learn the right way round (--rotate)
            if random.random() < 0.5:
                a, b = a[:, ::-1], b[:, ::-1]
            if random.random() < 0.5:
                a, b = a[::-1], b[::-1]
            if s.rotate:
                k = random.randrange(4)
                if k:
                    a, b = np.rot90(a, k), np.rot90(b, k)
            lrs.append(a); hrs.append(b)
        t = lambda L: torch.from_numpy(np.ascontiguousarray(np.stack(L)[..., ::-1])).permute(0, 3, 1, 2).float() / 255
        return t(lrs), t(hrs)


def patch_match(lr_patch, hr_patch):
    """How well a Blu-ray patch, shrunk to DVD size, lines up with the DVD patch (correlation, 1 = same)."""
    p = lr_patch.shape[0]
    g = lambda im: cv2.GaussianBlur(cv2.cvtColor(np.ascontiguousarray(im), cv2.COLOR_BGR2GRAY),
                                    (0, 0), 1.0).astype(np.float32)
    u = g(lr_patch)
    v = g(cv2.resize(np.ascontiguousarray(hr_patch), (p, p), interpolation=cv2.INTER_AREA))
    u -= u.mean(); v -= v.mean()
    return float((u * v).sum() / (math.sqrt(float((u * u).sum()) * float((v * v).sum())) + 1e-6))


def pick_spot(lr, hr, p, min_match, want=3, tries=8):
    """(y, x) of a patch: of `want` random spots that line up between the discs (a whole frame can pass the
    pairs step while someone in it moved between the two frames), the one with the most going on.
    min_match 0 skips the check; if no spot lines up, the best-matching one is used."""
    h, w = lr.shape[:2]
    good, fallback = [], None
    for _ in range(tries if min_match > 0 else want):
        y, x = random.randint(0, h - p), random.randint(0, w - p)
        a = lr[y:y + p, x:x + p]
        v = float(a.std())
        if min_match > 0:
            m = patch_match(a, hr[2 * y:2 * (y + p), 2 * x:2 * (x + p)])
            if fallback is None or m > fallback[0]:
                fallback = (m, y, x)
            if m < min_match:
                continue
        good.append((v, y, x))
        if len(good) >= want:
            break
    if not good:
        return fallback[1], fallback[2]
    _, y, x = max(good)
    return y, x


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


class UNetDiscriminatorSN(nn.Module):
    """Real-ESRGAN's U-Net discriminator with spectral norm: a real/fake score for every pixel, so it
    judges both the whole picture and the fine detail. Needs --patch a multiple of 4."""

    def __init__(s, nf=64):
        super().__init__()
        sn = nn.utils.spectral_norm
        s.conv0 = nn.Conv2d(3, nf, 3, 1, 1)
        s.conv1 = sn(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False))
        s.conv2 = sn(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False))
        s.conv3 = sn(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False))
        s.conv4 = sn(nn.Conv2d(nf * 8, nf * 4, 3, 1, 1, bias=False))
        s.conv5 = sn(nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=False))
        s.conv6 = sn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=False))
        s.conv7 = sn(nn.Conv2d(nf, nf, 3, 1, 1, bias=False))
        s.conv8 = sn(nn.Conv2d(nf, nf, 3, 1, 1, bias=False))
        s.conv9 = nn.Conv2d(nf, 1, 3, 1, 1)

    def forward(s, x):
        lr = lambda t: F.leaky_relu(t, 0.2)
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        x0 = lr(s.conv0(x)); x1 = lr(s.conv1(x0)); x2 = lr(s.conv2(x1)); x3 = lr(s.conv3(x2))
        x4 = lr(s.conv4(up(x3))) + x2
        x5 = lr(s.conv5(up(x4))) + x1
        x6 = lr(s.conv6(up(x5))) + x0
        return s.conv9(lr(s.conv8(lr(s.conv7(x6)))))


def discriminator(kind="unet"):
    if kind == "unet":
        return UNetDiscriminatorSN()
    sn = nn.utils.spectral_norm
    c = lambda i, o, k, st: sn(nn.Conv2d(i, o, k, st, k // 2))
    return nn.Sequential(c(3, 64, 3, 1), nn.LeakyReLU(0.2), c(64, 64, 4, 2), nn.LeakyReLU(0.2),
                         c(64, 128, 3, 1), nn.LeakyReLU(0.2), c(128, 128, 4, 2), nn.LeakyReLU(0.2),
                         c(128, 256, 3, 1), nn.LeakyReLU(0.2), c(256, 256, 4, 2), nn.LeakyReLU(0.2),
                         c(256, 1, 3, 1))                        # PatchGAN: a score per patch


# ---- evaluation ----
_SSIM_WIN = {}


def ssim(a, b):
    """Structural similarity of two [1,3,H,W] images in 0..1 (11x11 gaussian window, sigma 1.5)."""
    key = (a.device, a.dtype)
    if key not in _SSIM_WIN:
        g = torch.exp(-((torch.arange(11, dtype=torch.float32) - 5) ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        _SSIM_WIN[key] = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous().to(a.device, a.dtype)
    w = _SSIM_WIN[key]
    f = lambda t: F.conv2d(t, w, groups=3)
    mu_a, mu_b = f(a), f(b)
    va, vb, cov = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean().item()


def make_lpips(dev):
    """LPIPS (pip install lpips) if asked for and available, else None."""
    try:
        import lpips
        return lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    except Exception as e:
        print(f"WARNING: --lpips needs 'pip install lpips' and a first-time download ({str(e).splitlines()[0][:80]}): skipped")
        return None


def scores(out, hr, lp):
    out = out.float().clamp(0, 1)
    mse = F.mse_loss(out, hr).item()
    r = {"psnr": -10 * math.log10(max(mse, 1e-10)), "ssim": ssim(out, hr)}
    if lp is not None:
        r["lpips"] = lp(out * 2 - 1, hr * 2 - 1).mean().item()
    return r


def mean_scores(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


@torch.no_grad()
def eval_on(net, dev, items, crop, lp=None):
    """Mean PSNR / SSIM (/ LPIPS) of the network on the held-out pairs."""
    net.eval()
    rows = []
    for root, n in items:
        lr, hr = load_frame(root, n, crop)
        with torch.autocast(dev.type, torch.float16, enabled=dev.type == "cuda"):
            out = net(lr.to(dev))
        rows.append(scores(out, hr.to(dev), lp))
    return mean_scores(rows)


@torch.no_grad()
def eval_plain(dev, items, crop, lp=None):
    """The same for a plain bicubic 2x of the DVD frame: the floor any model must beat."""
    rows = []
    for root, n in items:
        lr, hr = load_frame(root, n, crop)
        up = F.interpolate(lr.to(dev), scale_factor=2, mode="bicubic", align_corners=False)
        rows.append(scores(up, hr.to(dev), lp))
    return mean_scores(rows)


def fmt_scores(r):
    return f"{r['psnr']:.2f} dB, SSIM {r['ssim']:.4f}" + (f", LPIPS {r['lpips']:.4f}" if "lpips" in r else "")


def auto_batch(net, dev, patch, per, frac, ids=None):
    """The largest batch (a multiple of --crops, up to 256) whose forward + backward pass stays within `frac`
    of the graphics card's memory (less when --gan / --perceptual need room for a second network)."""
    ids = ids or [dev.index or 0]
    total = min(torch.cuda.get_device_properties(i).total_memory for i in ids)       # (the smallest card decides)
    fixed = 3 * 4 * sum(q.numel() for q in net.parameters())     # (the averaged copy and Adam's two states; the pass counts the rest)
    best = max(per, 8 - 8 % per)
    net.train()
    for b in (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        b = max(per, b - b % per)
        if b <= best:
            continue
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            x = torch.rand(b, 3, patch, patch, device=dev)
            with torch.autocast(dev.type, torch.float16):
                loss = net(x).float().mean()
            loss.backward()
            peak = torch.cuda.max_memory_allocated(dev)
        except torch.cuda.OutOfMemoryError:
            break
        finally:
            net.zero_grad(set_to_none=True)
            x = loss = None
            torch.cuda.empty_cache()
        if peak + fixed > frac * total:
            break
        best = b
    return best * len(ids)                                        # (each card takes a full batch of its own)


def split_pairs(root, names, block=60.0, every=20, gap=5.0):
    """(train, held_out, how). Pairs are ~1.5 s apart, so every 20th pair has near-copies from the same shot
    in the training set and the held-out score mostly measures memory. Instead whole minutes of the movie
    (one in every 20) are held out, and training pairs within `gap` seconds of them are left out too.
    Falls back to every 20th pair when pairs.csv doesn't give the times."""
    times = {}
    try:
        with open(Path(root) / "pairs.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                times[r["name"]] = float(r["t_dvd"])
    except (OSError, KeyError, ValueError):
        pass
    if all(n in times for n in names):
        held = lambda t: int(t // block) % every == every // 2
        val = [n for n in names if held(times[n])]
        train = [n for n in names if not (held(times[n]) or held(times[n] - gap) or held(times[n] + gap))]
        if len(val) >= 20:
            return train, val, f"minute {every // 2} of every {every}, never trained on"
    val = names[::every]
    held_set = set(val)
    return [n for n in names if n not in held_set], val, f"every {every}th pair"


def stage_train(a, roots, out):
    """Step 2: fine-tune on the pairs in the folders `roots` (one movie's, or several movies' of one kind).
    Carries on from out/train_state.pt. True when finished, False when stopped with Ctrl+C."""
    prog, live = PROG, LIVE
    if a.pretrained != "none" and not Path(a.pretrained).exists():
        raise SystemExit(f"starting model not found: {a.pretrained}\nDownload RealESRGAN_x2plus.pth from\n"
                         f"{PRETRAINED_URL}\n"
                         "and put it in the training folder (or pass its path with --pretrained).")
    ids = pick_gpus(a.gpus)
    dev = torch.device(f"cuda:{ids[0]}") if ids else torch.device("cpu")
    if ids:
        torch.cuda.set_device(dev)
        HW.focus = ids
        names_ = [torch.cuda.get_device_name(i) for i in ids]
        say(f"training on {len(ids)} GPU(s): " + ", ".join(f"{i}: {n}" for i, n in zip(ids, names_)))
        if len(set(names_)) > 1:
            say("WARNING: the GPUs are different models: every step waits for the slowest one (use --gpus to pick matching ones)")
    if dev.type == "cpu":
        print("WARNING: no CUDA GPU found, training on the processor (very slow)")
    out.mkdir(parents=True, exist_ok=True)
    train, val, split, small, split_id = [], [], "", [], []
    for root in map(Path, roots):
        names = sorted(f.stem for f in (root / "lr").glob("*.png") if (root / "hr" / f.name).exists())
        if not names:
            say(f"WARNING: no pairs in {root}: left out")
            continue
        t, v, split = split_pairs(root, names)
        split_id.append(f"{root.resolve()}:{split}")
        train += [(root, n) for n in t]
        val += [(root, n) for n in v]
        probe_img = cv2.imread(str(root / "lr" / f"{names[0]}.png"))
        if probe_img is None or min(probe_img.shape[:2]) < a.patch:
            small.append(str(root / "lr"))
    split_id = "|".join(sorted(split_id))                # (which movies and how each is split: saved with the training)
    if len(train) < 8:
        raise SystemExit(f"only {len(train) + len(val)} pairs in {', '.join(map(str, roots))}: make more in the pairs step")
    if small:
        raise SystemExit(f"the DVD pictures in {', '.join(small)} are smaller than --patch {a.patch}: lower --patch")
    say(f"{len(train)} training pairs, {len(val)} held out ({split}"
        + (f", in each of {len(roots)} movies" if len(roots) > 1 else "") + f"); device {dev}")
    focus_dirs = {str(Path(d).resolve()) for d in (a.focus_from or [])}
    focus = [(r, n) for r, n in train if str(Path(r).resolve()) in focus_dirs]
    if focus and len(focus) < len(train) * 0.5:
        say(f"new movies: {len(focus_dirs)} ({len(focus)} pairs): half of each step's frames come from them, "
            "the other half from all the movies, so the new ones are learned and the earlier ones kept")
    if prog:
        prog.set(title=f"Training: {out.name}", total=a.iters, phase="starting")

    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True            # (every step has the same shape: the fastest kernels are picked once)
    if a.gan > 0 and a.disc == "unet" and a.patch % 4:
        raise SystemExit(f"--patch {a.patch}: the U-Net discriminator needs a multiple of 4 (or use --disc patch)")
    net = make_net(a.arch, a.compact_feat, a.compact_convs).to(dev)
    if a.pretrained != "none":
        load_weights(net, a.pretrained)
    elif a.arch == "compact":
        print("note: the compact network has no 2x starting model, so it learns from zero: it needs far more "
              "steps than the default (try --iters 100000 or more) and more pairs (--count 6000+)")
    else:
        print("WARNING: random start (--pretrained none): only for testing the scripts")
    net.use_checkpoint = a.checkpoint
    cl = dev.type == "cuda" and not a.no_channels_last
    if cl:
        net = net.to(memory_format=torch.channels_last)          # (the layout tensor cores like: faster fp16 convolutions)
    if not 0.1 <= a.gpu_memory <= 1.0:
        raise SystemExit("--gpu-memory is a share of the card's memory between 0.1 and 1.0 (e.g. 0.9)")
    if a.batch == "auto":
        a.batch = auto_batch(net, dev, a.patch, max(1, a.crops), a.gpu_memory * (0.8 if (a.gan or a.perceptual) else 1.0), ids) if dev.type == "cuda" else 8
        say(f"--batch auto: {a.batch} patches per step")
    if a.batch % max(1, a.crops):
        c = max(1, a.crops)
        fixed_batch = max(c, round(a.batch / c) * c)
        say(f"--batch {a.batch} is not a multiple of --crops {c} (each loaded pair gives {c} patches): using {fixed_batch}")
        a.batch = fixed_batch
    ema = copy.deepcopy(net).eval()
    model = nn.DataParallel(net, device_ids=ids) if len(ids) > 1 else net      # (what the training steps run; net holds the weights)
    if len(ids) > 1:
        a.batch = max(len(ids), a.batch - a.batch % len(ids))                 # (a whole number of patches per GPU)
        say(f"{a.batch} patches per step, {a.batch // len(ids)} on each GPU")
    for q in ema.parameters():
        q.requires_grad = False
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    vgg = VGGLoss().to(dev) if a.perceptual > 0 else None
    disc = discriminator(a.disc).to(dev) if a.gan > 0 else None
    lpips_net = make_lpips(dev) if a.lpips else None
    d_opt = torch.optim.Adam(disc.parameters(), lr=a.lr, betas=(0.9, 0.99)) if disc else None
    d_scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    step, best = 0, -1e9
    state_path = out / "train_state.pt"
    note_settings(a, out, state_path.exists() or state_path.with_name(state_path.name + ".prev").exists())
    check_disk(out, int(10 * 4 * sum(q.numel() for q in net.parameters()) * 1.2), "the saved training (and its backup)")
    if state_path.exists() or state_path.with_name(state_path.name + ".prev").exists():
        st, used = load_state(state_path, dev)
        if st.get("arch", "rrdb") != a.arch:
            raise SystemExit(f"{state_path} is a '{st.get('arch', 'rrdb')}' training, not '{a.arch}': "
                             "use --arch to match it, or another --work folder for the new network")
        net.load_state_dict(st["net"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        if disc and "disc" in st:
            try:
                disc.load_state_dict(st["disc"]); d_opt.load_state_dict(st["d_opt"])
            except (RuntimeError, ValueError):
                say("the saved discriminator is a different kind (--disc): starting a new one")
                disc = discriminator(a.disc).to(dev)
                d_opt = torch.optim.Adam(disc.parameters(), lr=a.lr, betas=(0.9, 0.99))
        step, best = st["step"], st["best"]
        if st.get("split") != split_id:
            best = -1e9                                  # (scores on other held-out frames can't be compared)
            say("the held-out frames changed since the earlier run: the best score starts over")
        if "scaler" in st:
            scaler.load_state_dict(st["scaler"])
        say(f"resuming at step {step}")

    # where we start from, on frames never trained on
    floor = eval_plain(dev, val, a.val_crop, lpips_net)
    base = eval_on(ema, dev, val, a.val_crop, lpips_net) if step == 0 else None
    if base is not None:
        say(f"held-out: plain bicubic {fmt_scores(floor)};  starting model {fmt_scores(base)}")
    if prog:
        for key, name in (("psnr", "held-out PSNR (dB)"), ("ssim", "held-out SSIM"), ("lpips", "held-out LPIPS (lower is better)")):
            if key in floor:
                prog.ref(name, "bicubic", floor[key])
                if base is not None:
                    prog.ref(name, "start", base[key])
        prog.set(step=step, phase="training")

    per = max(1, a.crops)
    ds = Pairs(train, a.patch, per, a.rotate, a.min_patch_match, focus)
    workers = a.train_workers * max(1, len(ids))              # (more GPUs eat data faster)
    def make_loader():
        return torch.utils.data.DataLoader(ds, batch_size=max(1, a.batch // per), num_workers=workers,
                                           collate_fn=collate, persistent_workers=workers > 0,
                                           pin_memory=dev.type == "cuda", prefetch_factor=4 if workers > 0 else None)
    dl = make_loader()
    oom_steps = 0                                        # times the batch was halved after running out of GPU memory
    ema_params, net_params = list(ema.parameters()), list(net.parameters())
    bad = 0                                              # checks in a row (every 10 steps) with a loss that is not a number
    def save_state():
        save_durably({"params_ema": ema.state_dict()}, out / "upscale_training_latest.pth")
        save_durably({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                      "step": step, "best": best, "scaler": scaler.state_dict(), "arch": a.arch, "split": split_id,
                      **({"disc": disc.state_dict(), "d_opt": d_opt.state_dict()} if disc else {})}, state_path, keep_prev=True)

    it = iter(dl)
    t0, run, n_run = time.time(), {}, 0
    run_t0, start_step, last_l1 = time.time(), step, float("nan")
    wait_t, loop_t, tick = 0.0, 0.0, time.time()      # data-loader wait vs whole step, for the verdict
    net.train()
    try:
        while True:
            try:
                while step < a.iters:
                    w0 = time.time()
                    batch = next(it)                       # (blocks while the data loader is behind)
                    wait_t += time.time() - w0
                    lr_img, hr_img = (x.to(dev, non_blocking=True) for x in batch)
                    if cl:
                        lr_img = lr_img.contiguous(memory_format=torch.channels_last)
                    lr_now = a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.iters)))     # cosine to 10%
                    for g in opt.param_groups:
                        g["lr"] = lr_now
                    if d_opt:
                        for g in d_opt.param_groups:
                            g["lr"] = lr_now
                    # (the losses stay on the GPU: reading one with .item() makes the processor wait for the GPU to
                    #  finish the step, so it can't queue the next. They are read every 10 steps instead.)
                    with torch.autocast(dev.type, torch.float16, enabled=dev.type == "cuda"):
                        pred = model(lr_img)
                        loss = F.l1_loss(pred, hr_img)
                        logs = {"l1": loss.detach()}
                        if vgg:
                            lp = vgg(pred.clamp(0, 1), hr_img)
                            loss = loss + a.perceptual * lp; logs["vgg"] = lp.detach()
                        if disc:
                            disc.requires_grad_(False)                    # (only the generator learns from this score)
                            lg = F.softplus(-disc(pred)).mean()           # fool the discriminator
                            disc.requires_grad_(True)
                            loss = loss + a.gan * lg; logs["g"] = lg.detach()
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
                        logs["d"] = ld.detach()
                    step += 1
                    decay = min(0.999, (1 + step) / (10 + step))
                    with torch.no_grad():
                        torch._foreach_mul_(ema_params, decay)               # (one fused call, not a python loop over ~500 tensors)
                        torch._foreach_add_(ema_params, net_params, alpha=1 - decay)
                    for k, v in logs.items():
                        run[k] = run[k] + v.float() if k in run else v.float()
                    n_run += 1
                    loop_t += time.time() - tick
                    tick = time.time()
                    if step % 10 == 0:
                        last_l1 = logs["l1"].item()
                        if not math.isfinite(last_l1):
                            bad += 1
                            if bad >= 5:
                                raise SystemExit("the loss has not been a number for 50 steps: training has blown up. "
                                                 "Run again with a lower --lr (e.g. half), or fewer --gan / --perceptual.")
                        else:
                            bad = 0
                        el = time.time() - run_t0
                        sps = el / max(step - start_step, 1)          # (average over this run, saves included)
                        eta = (a.iters - step) * sps
                        live.fields([f"{bar(step / a.iters, 16)} {100 * step / a.iters:5.1f}%  {step}/{a.iters} steps",
                                     f"ETA {hms(eta)} (~{clock_in(eta)})", f"elapsed {hms(el)}",
                                     f"{sps:.2f} s/step", f"l1 {last_l1:.4f}"] + ([f"best {best:.2f} dB"] if best > -1e8 else []),
                                    HW.line()[:1] + queue_parts(eta) + HW.line()[1:])
                        if prog:
                            prog.set(step=step, eta=eta, speed_text=f"{sps:.2f} s/step", sec_per_step=sps, elapsed=el)
                            prog.metric(**HW.metrics())
                    if step % 100 == 0 and n_run:
                        el = time.time() - t0
                        avg = {k: v.item() / n_run for k, v in run.items()}     # (n_run: fewer than 100 after a resume)
                        say(f"step {step}/{a.iters}  " + "  ".join(f"{k} {v:.4f}" for k, v in avg.items())
                            + f"  lr {lr_now:.2e}  {el / n_run:.2f} s/step")
                        if prog:
                            prog.point("loss (L1)", step, avg["l1"])
                            prog.metric(**{k: f"{v:.4f}" for k, v in avg.items()})
                        wf = wait_t / loop_t if loop_t else 0.0
                        why = verdict(HW, wf, el / n_run)
                        say(f"   hardware: " + "  ".join(f"{k} {v}" for k, v in HW.metrics().items()) + f"  | data wait {100 * wf:.0f}%")
                        say(f"   -> {why}")
                        if prog:
                            prog.metric(**{"data wait": f"{100 * wf:.0f} %", "what limits it": why})
                            if HW.gpu and HW.gpu["watts"] is not None:
                                prog.point("GPU power (W)", step, HW.gpu["watts"])
                            if HW.gpu and HW.gpu["busy"] is not None:
                                prog.point("GPU busy (%)", step, HW.gpu["busy"])
                            if HW.cpu is not None:
                                prog.point("CPU (%)", step, HW.cpu)
                        run, n_run, t0, wait_t, loop_t = {}, 0, time.time(), 0.0, 0.0
                    if step % a.save_every == 0 or step == a.iters:
                        res = eval_on(ema, dev, val, a.val_crop, lpips_net)
                        score = res["psnr"]
                        net.train()
                        note = ""
                        if score > best:
                            best = score; note = " (best)"
                            save_durably({"params_ema": ema.state_dict()}, out / "upscale_training_best.pth")
                        save_state()
                        say(f"== step {step}: held-out {fmt_scores(res)}{note}   (bicubic {fmt_scores(floor)}"
                            + (f"; start {fmt_scores(base)}" if base is not None else "") + ")")
                        if prog:
                            prog.point("held-out PSNR (dB)", step, score)
                            prog.point("held-out SSIM", step, res["ssim"])
                            if "lpips" in res:
                                prog.point("held-out LPIPS (lower is better)", step, res["lpips"])
                            prog.metric(**{"best PSNR": f"{best:.2f} dB"})
                break
            except torch.cuda.OutOfMemoryError:
                if oom_steps >= 4 or max(1, a.batch // per) <= 1:
                    raise                               # (nothing smaller left to try: handled below, progress saved)
            # (outside the except block, so the failed step's tensors can really be freed)
            oom_steps += 1
            batch = loss = pred = lr_img = hr_img = None
            opt.zero_grad(set_to_none=True)
            if disc:
                d_opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            a.batch = max(1, a.batch // per // 2) * per
            live.clear()
            say(f"the graphics card ran out of memory: carrying on with half the batch ({a.batch} patches per step)")
            del it
            dl = make_loader()
            it = iter(dl)
            wait_t, loop_t, tick = 0.0, 0.0, time.time()
    except torch.cuda.OutOfMemoryError:
        live.clear()
        torch.cuda.empty_cache()
        save_state()
        print(f"\nThe graphics card ran out of memory at step {step}; progress is saved. Run the same command again with a "
              f"smaller --batch (now {a.batch}) or --gpu-memory 0.9, or add --checkpoint.", flush=True)
        if prog:
            prog.set(phase=f"out of GPU memory at step {step}: run again with a smaller --batch")
        return False
    except KeyboardInterrupt:
        a.stopped = True
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
    arch, kw = arch_of(sd)
    make_net(arch, **kw).load_state_dict(sd, strict=True)
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

    if arch == "compact":
        # conv + PReLU ..., conv to 12 channels, PixelShuffle, plus the input enlarged by nearest-neighbour
        def prelu(name, inp, outp, key):
            slope = sd[key + ".weight"]
            ops.append(["PReLU", name, [inp], [outp], f"0={slope.numel()}"])
            binbuf.extend(slope.detach().cpu().numpy().astype("<f4").tobytes())

        ops.append(["Input", "data", [], ["data"], ""])
        x, last = "data", 2 * kw["nconv"] + 2
        for i in range(0, last, 2):
            conv(f"conv{i}", x, f"c{i}", f"body.{i}")
            prelu(f"prelu{i}", f"c{i}", f"p{i}", f"body.{i + 1}")
            x = f"p{i}"
        conv(f"conv{last}", x, "c_last", f"body.{last}")
        ops.append(["PixelShuffle", "shuffle", ["c_last"], ["shuffled"], "0=2"])
        up2("base", "data", "base_up")
        add("sum", "shuffled", "base_up", "output")
        return finish_ncnn(ops, binbuf, dest_dir, model_name)

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
    return finish_ncnn(ops, binbuf, dest_dir, model_name)


def finish_ncnn(ops, binbuf, dest_dir, model_name):
    """Write the layer list: a blob read by several layers goes through a Split, one copy per reader."""
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


def find_pretrained(name="RealESRGAN_x2plus.pth"):
    """The starting model: next to this script, in a Training / training folder next to it, in the
    upscaler's models folder, or a folder up. The first found; else the path next to this script."""
    here = Path(__file__).resolve().parent
    places = [here, here / "Training", here / "training", here / "models", here.parent, here.parent / "Training"]
    models = find_models_dir()
    if models:
        places.append(models)
    for d in places:
        if (d / name).is_file():
            return d / name
    return here / name


def find_models_dir():
    """The models folder next to realesrgan-ncnn-vulkan, looked for from here upwards."""
    here = Path(__file__).resolve().parent
    for d in [here, *here.parents][:6]:
        if (d / "models").is_dir() and any((d / n).exists() for n in ("realesrgan-ncnn-vulkan.exe", "realesrgan-ncnn-vulkan")):
            return d / "models"
    return None


def held_out_items(roots):
    """The held-out pairs of these movies' pairs folders (the same minutes every training leaves out)."""
    val = []
    for root in map(Path, roots):
        names = sorted(f.stem for f in (root / "lr").glob("*.png") if (root / "hr" / f.name).exists())
        if names:
            val += [(root, n) for n in split_pairs(root, names)[1]]
    return val


def judge(a, roots, new_pth, old_pth):
    """(new model's scores, old model's scores) on the held-out minutes of all the movies in `roots`: frames
    neither model trained on. None when it can't be done (no pairs, an unreadable model)."""
    val = held_out_items(roots)
    if not val:
        return None
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    out = []
    for p in (new_pth, old_pth):
        try:
            sd = torch.load(p, map_location="cpu", weights_only=True)
            sd = sd.get("params_ema", sd.get("params", sd))
            arch, kw = arch_of(sd)
            net = make_net(arch, **kw)
            net.load_state_dict(sd, strict=True)
        except (OSError, RuntimeError, ValueError, KeyError) as e:
            say(f"can't read {p} to compare ({e})")
            return None
        net = net.to(dev)
        out.append(eval_on(net, dev, val, a.val_crop))
        del net
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    say(f"judged on {len(val)} held-out frames of {len(roots)} movie{'s' * (len(roots) > 1)}")
    return out


def stage_export(a, run_dir, work, roots=()):
    """Step 3: the trained model as .param/.bin where dvd_upscale.py finds it. When a model of that name is
    there already, the new one replaces it only if it scores higher on the held-out frames of all the movies."""
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
    name = model_name(a, getattr(a, "kind", None))
    gate = work / "gate.json"
    gate.unlink(missing_ok=True)
    old_pth = models / f"{name}.pth"
    import filecmp
    same_weights = old_pth.exists() and filecmp.cmp(pth, old_pth, shallow=False)     # (converting the model in use again)
    if same_weights:
        say(f"the {name} in use has these very weights: converting it again (no comparison, no -before copy)")
    if old_pth.exists() and roots and not a.keep_any and not same_weights:
        if a.gan or a.perceptual:
            say("--gan / --perceptual: held-out PSNR can't tell which model looks better, so the new one is "
                "used without comparing (the one before is kept as -before)")
        else:
            got = judge(a, roots, pth, old_pth)
            if got:
                new_s, old_s = got
                say(f"new {name}: {fmt_scores(new_s)};  the {name} in use: {fmt_scores(old_s)}")
                if new_s["psnr"] <= old_s["psnr"]:
                    write_json_durably(gate, {"accepted": False, "new": new_s, "old": old_s, "at": time.time()})
                    say(f"the new model isn't better: the {name} in use stays (the new one is {pth}). Its movies "
                        "are learned again when the next movie of this kind is added.")
                    return True
                say(f"the new model is better by {new_s['psnr'] - old_s['psnr']:.3f} dB: it replaces the one in use")
                write_json_durably(gate, {"accepted": True, "new": new_s, "old": old_s, "at": time.time()})
            else:
                say("couldn't compare with the model in use: the new one is used (the one before is kept as -before)")
    if roots:
        write_json_durably(work / "accepted.json", sorted(str(Path(r).resolve()) for r in roots))
    if (models / f"{name}.param").exists() and (models / f"{name}.bin").exists() and not a.name \
            and not same_weights:
        # (an earlier movie of the same kind made this model: keep it, it may be the better one)
        for ext in ("param", "bin", "pth"):
            if (models / f"{name}.{ext}").exists():
                replace_file(models / f"{name}.{ext}", models / f"{name}-before.{ext}")
        say(f"the {name} model already there is kept as {name}-before (use --model {name}-before to compare)")
    n_layers, n_bytes = export_ncnn(pth, models, name)
    # (its weights too: the upscaler doesn't use them, but a later training of this kind starts from them when
    #  the movies it learned from are gone, --delete-originals)
    tmp = models / f"{name}.pth.tmp"
    shutil.copyfile(pth, tmp)
    fsync_file(tmp)
    replace_file(tmp, models / f"{name}.pth")
    say(f"wrote {models / (name + '.param')}, {name}.bin and {name}.pth ({n_layers} layers, {n_bytes / 1e6:.0f} MB, "
        f"from {pth.name})")
    say(f"use it:  {upscale_command(a)}")
    return True


def upscale_command(a):
    """--trained picks ai-<type>-x2 by the type dvd_upscale.py detects (the same detection as here)."""
    if a.name:
        return f"python dvd_upscale.py <movie> --model {a.name} --scale 2"
    kind = getattr(a, "kind", None)
    return "python dvd_upscale.py <movie> --trained" + (f"   (it picks {model_name(a, kind)} for movies it detects "
                                                         f"as {TYPE_NAMES[kind]})" if kind in TYPE_NAMES else "")


def analyze(a):
    """--analyze: what dvd_upscale.py makes of --dvd, the filters the pairs would use, and the model's name."""
    if not a.dvd or not Path(a.dvd).exists():
        sys.exit("--analyze needs --dvd and a movie file that exists")
    work = default_work(a)
    work.mkdir(parents=True, exist_ok=True)
    setup_kind(a, work, need_filters=True)
    print(f"the trained model would be named {model_name(a, a.kind)};  then upscale with:  {upscale_command(a)}")


def pick_gpus(spec):
    """CUDA device numbers to train on: 'all' (default) or a list like '0,1'. [] without CUDA."""
    n = torch.cuda.device_count()
    if n == 0:
        return []
    if str(spec).strip().lower() == "all":
        return list(range(n))
    try:
        ids = [int(x) for x in str(spec).split(",") if x.strip() != ""]
    except ValueError:
        raise SystemExit(f"--gpus: 'all' or a list of numbers like 0,1 (got {spec!r})")
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= n for i in ids):
        raise SystemExit(f"--gpus {spec}: this PC has {n} CUDA GPU(s), numbered 0..{n - 1} (as nvidia-smi lists them), each used once")
    return ids


def bench(a, devs, steps=12, warm=3):
    """Patches per second for training steps on random data across `devs` (one card, or several split)."""
    dev = devs[0]
    net = make_net(a.arch, a.compact_feat, a.compact_convs).to(dev)
    net.train()
    model = nn.DataParallel(net, device_ids=[d.index for d in devs]) if len(devs) > 1 else net
    batch = 8 * len(devs)                                       # (8 patches per card, so the cards are compared fairly)
    opt = torch.optim.Adam(net.parameters(), lr=1e-5)
    cuda = dev.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=cuda)
    x = torch.rand(batch, 3, a.patch, a.patch)
    y = torch.rand(batch, 3, 2 * a.patch, 2 * a.patch, device=dev)
    for i in range(steps):
        if i == warm:
            if cuda:
                torch.cuda.synchronize(dev)
            t0 = time.time()
        with torch.autocast(dev.type, torch.float16, enabled=cuda):
            loss = F.l1_loss(model(x.to(dev)), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    if cuda:
        torch.cuda.synchronize(dev)
    return batch * (steps - warm) / (time.time() - t0)


def pcie_links():
    """{gpu number: 'Gen3 x4 (card supports Gen4 x16)'} from nvidia-smi, {} if it can't be read."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {}
    try:
        r = subprocess.run([smi, "--query-gpu=index,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,"
                            "pcie.link.width.max", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=8, stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return {}
    out = {}
    for row in r.stdout.splitlines():
        v = [x.strip() for x in row.split(",")]
        if len(v) == 5 and v[0].isdigit():
            out[int(v[0])] = f"Gen{v[1]} x{v[2]} (card supports Gen{v[3]} x{v[4]})"
    return out


def gpu_test(a):
    """--gpu-test: lists the NVIDIA GPUs, times each alone and all together, and says what to use."""
    n = torch.cuda.device_count()
    if not n:
        print("no NVIDIA (CUDA) GPU found by PyTorch: check the driver (nvidia-smi) and that torch was installed with CUDA.")
        return
    print(f"{n} CUDA GPU(s):")
    for i in range(n):
        pr = torch.cuda.get_device_properties(i)
        print(f"  {i}: {pr.name}, {pr.total_memory / 2 ** 30:.1f} GB")
    if n < 2:
        print("only one GPU: nothing to compare. A second card will show up here (and in nvidia-smi) once it is installed.")
        return
    torch.backends.cudnn.benchmark = True
    results = {}
    for ids in [[i] for i in range(n)] + [list(range(n))]:
        try:
            results[tuple(ids)] = bench(a, [torch.device(f"cuda:{i}") for i in ids])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results[tuple(ids)] = None
        r = results[tuple(ids)]
        print(f"  --gpus {','.join(map(str, ids)):<6} " + ("ran out of memory" if r is None else f"{r:6.1f} patches/s"))
    links = pcie_links()                                       # (read now, while the cards are busy: an idle card drops its link speed)
    if links:
        print("PCIe link of each card (an external M.2 / OCuLink case is usually x4):")
        for i in range(n):
            print(f"  {i}: {links.get(i, 'unknown')}")
    singles = {k: v for k, v in results.items() if len(k) == 1 and v}
    both = results[tuple(range(n))]
    if not singles:
        return
    best = max(singles, key=singles.get)
    if both and both > 1.15 * singles[best]:
        print(f"-> use all of them (the default, --gpus all): {both / singles[best]:.2f}x the best single card.")
    else:
        print(f"-> a second card does not speed this up (together: {both or 0:.1f} vs {singles[best]:.1f} patches/s alone). "
              f"Use --gpus {best[0]}. Common causes: a card on a slow link (Thunderbolt/eGPU or a x1 slot), "
              "or two very different cards.")
    if len({torch.cuda.get_device_name(i) for i in range(n)}) > 1:
        print("note: the cards are different models, so a joint run is paced by the slower one.")


def batch_arg(v):
    return v if v == "auto" else int(v)


# =============================== --all / --queue: several movies, one model per kind of movie ===============================
# First the pairs of every movie, one after another (each in its own <DVD name>_training folder, resumable), then
# one training per kind of movie on the pairs of all the movies of that kind: ai-cgi-x2 from every CGI movie,
# ai-anime-x2 from every anime... (training them one by one would leave only the last movie's model).
# Each step runs as a run of this script of its own, so a failure is logged and the queue goes on.
VIDEO_EXT = {".mkv", ".mp4", ".m4v", ".avi", ".mpg", ".mpeg", ".ts", ".m2ts", ".vob", ".wmv", ".mov"}
NOT_IN_QUEUE = ("--dvd", "--bluray", "--work", "--stages", "--name", "--pairs-from", "--focus-from", "--analyze", "--gpu-test")
STEPS_PER_MOVIE = 10000         # (the queue's training steps: per movie for a new model, per new movie when it carries on)
STOPPED_CODES = (130, -2, 3221225786)        # (Ctrl+C: our own exit code, a signal, Windows' STATUS_CONTROL_C_EXIT)


def queue_tokens(line):
    """A queue line as arguments: words and "quoted names"; # starts a comment (outside quotes)."""
    toks, cur, inq, quoted = [], [], False, False
    for ch in line.replace("\u201c", '"').replace("\u201d", '"'):
        if ch == '"':
            inq, quoted = not inq, True
        elif ch == "#" and not inq:
            break
        elif ch.isspace() and not inq:
            if cur or quoted:
                toks.append("".join(cur))
            cur, quoted = [], False
        else:
            cur.append(ch)
    if cur or quoted:
        toks.append("".join(cur))
    return toks


def passed_on(argv):
    """The command line without --all / --queue / --shutdown: options for every movie (--count, --iters ...)."""
    out, skip = [], False
    for i, t in enumerate(argv):
        if skip:
            skip = False
            continue
        name = t.split("=", 1)[0]
        if name in ("--all", "--queue"):
            skip = "=" not in t and i + 1 < len(argv) and not argv[i + 1].startswith("-")
            continue
        if name not in ("--shutdown", "--list", "--delete-originals"):
            out.append(t)
    return out


def find_movie_pairs(folder):
    """[(dvd, bluray)] in folder and its anime/live/cgi/3d/vhs folders, by name: Shrek_dvd.mkv + Shrek_bd.mkv
    (also "Shrek DVD.mkv" + "Shrek Blu-ray.mkv"; spaces, dashes, dots and capitals don't matter);
    and [(dvd, why)] for DVDs left out."""
    dirs = [folder] + [d for d in sorted(folder.iterdir()) if d.is_dir() and d.name.lower() in TYPE_DIRS]
    found, left = [], []
    for d in dirs:
        vids = [f for f in sorted(d.iterdir()) if f.is_file() and f.suffix.lower() in VIDEO_EXT]
        # (only letters and digits count: "Kingsman- The Golden Circle" and "Kingsman The Golden Circle",
        #  "Despicable Me" and "Despicable me" are the same movie)
        key = lambda m: re.sub(r"[^0-9a-z]", "", m.group(1).lower())
        bds = {}
        for f in vids:
            m = re.match(r"(.*?)[ _.-]*(bd|blu-?ray)$", f.stem, re.I)
            if m and key(m):
                bds.setdefault(key(m), f)
        for f in vids:
            m = re.match(r"(.*?)[ _.-]*dvd$", f.stem, re.I)
            if not m or not key(m):
                continue
            if abs(time.time() - f.stat().st_mtime) < 120:
                left.append((f, "changed in the last 2 minutes (still being copied?): next time"))
            elif key(m) in bds:
                found.append((f, bds[key(m)]))
            else:
                left.append((f, f"no Blu-ray next to it (name it {m.group(1).strip(' _.-')}_bd{f.suffix})"))
    return found, left


class QueueMovie:
    def __init__(s, label, args, base, parser, extras):
        """args: this movie's own options (its queue line, or --dvd/--bluray found by --all)."""
        s.label, s.error, s.args = label, None, list(args)
        try:
            a = parser.parse_args(extras + args)
        except SystemExit:
            s.error = "can't read the options on this line"
            return
        if not a.dvd or not a.bluray:
            s.error = "needs --dvd and --bluray"
            return
        absolute = lambda x: x if x is None or Path(x).is_absolute() else str(base / x)
        a.dvd, a.bluray, a.work = absolute(a.dvd), absolute(a.bluray), absolute(a.work)
        for f in (a.dvd, a.bluray):
            if not Path(f).is_file():
                s.error = f"file not found: {f}"
                return
        s.dvd, s.work, s.solo, s.name = Path(a.dvd), default_work(a), a.name is not None, a.name   # (--name: a model of its own)
        s.bluray = Path(a.bluray)
        s.count, s.given = a.count, (a.type if a.type != "auto" else None)
        own = type_from_place(a.dvd)              # (its folder, else its name's tag: wins over a --type for all)
        s.own = own
        if own and "--type" not in args:
            s.args = ["--type", own] + s.args

    def pairs_made(s):
        try:
            with open(s.work / "pairs" / "pairs.csv", newline="") as fh:
                return max(0, sum(1 for _ in fh) - 1)
        except OSError:
            return 0

    def kind_and_why(s):
        """(kind or None, where it comes from) as far as it is known before the movie's turn."""
        if "--type" in s.args:
            k = s.args[s.args.index("--type") + 1]
            return k, ("its folder" if s.dvd.parent.name.lower() in TYPE_DIRS else "its name") if k == s.own else "--type"
        if s.given:
            return s.given, "--type"
        if s.kind():
            return s.kind(), "detected before"
        return None, "detected when its turn comes"

    def kind(s):
        try:
            return json.loads((s.work / "upscaler.json").read_text(encoding="utf-8"))["type"]
        except (OSError, ValueError, KeyError):
            return None


def queue_movies(a, base, parser, extras):
    if a.all is not None:
        found, left = find_movie_pairs(base)
        rel = lambda f: str(f.relative_to(base))
        movies = [QueueMovie(rel(d), ["--dvd", str(d), "--bluray", str(b)], base, parser, extras) for d, b in found]
        return movies, [(rel(d), why) for d, why in left]
    qfile = Path(a.queue).resolve()
    try:
        lines = qfile.read_text(encoding="utf-8-sig").splitlines()
    except OSError as e:
        sys.exit(f"can't read the queue file {qfile}: {e}\nIt lists one movie per line, e.g.\n"
                 '  --dvd "Shrek_dvd.mkv" --bluray "Shrek_bd.mkv"\n'
                 '  --dvd "C:\\Movies\\DBZ_dvd.mkv" --bluray "C:\\Movies\\DBZ_bd.mkv" --type anime   # a comment')
    movies = []
    for n, line in enumerate(lines, 1):
        toks = queue_tokens(line)
        if toks:
            m = QueueMovie(f"line {n}", toks, base, parser, extras)
            if m.error is None:
                m.label = f"line {n} ({m.dvd.name})"
            movies.append(m)
    return movies, []


def gate_rejected(work):
    """True when the last export of this work folder kept the model in use because the new one wasn't better:
    the movies of that training are then NOT learned by the model in use."""
    try:
        return json.loads((Path(work) / "gate.json").read_text(encoding="utf-8")).get("accepted") is False
    except (OSError, ValueError, AttributeError):
        return False


def finished(work, pairs_from=None):
    try:
        f = json.loads((Path(work) / "finished.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return pairs_from is None or f.get("pairs_from") == pairs_from


class QueueStatus(Progress):
    """--all / --queue: the whole queue on one page that stays up from start to end (each movie and model, done
    / now / waiting, the queue's time left and elapsed), with the current run's own progress, GPU readings and
    charts under it (that run writes them to child_file). Also the time the jobs still waiting need, for the
    console line of each run: guessed at first, then from the speeds measured on this PC."""

    def __init__(s, a, base, parser, extras, results):
        super().__init__("Training queue", a.web)
        s.a, s.base, s.parser, s.extras, s.results = a, base, parser, extras, results
        s.t0, s.current, s.models, s.q = time.time(), None, {}, None
        s.child_file = base / ".training_queue_status.json"
        s.pairs_s, s.step_s = 3000 / (3 * 3600), 0.36       # (pairs per second, seconds per training step: guesses)
        s.speeds_file = base / ".training_speeds.json"     # (...replaced by what this PC measured, also in earlier runs)
        try:
            known = json.loads(s.speeds_file.read_text(encoding="utf-8"))
            s.pairs_s = float(known.get("pairs_s") or s.pairs_s)
            s.step_s = float(known.get("step_s") or s.step_s)
        except (OSError, ValueError, TypeError):
            pass
        try:
            s.iters = parser.parse_args(extras + ["--dvd", "x"]).iters
        except SystemExit:
            s.iters = 20000

    def refresh(s):
        """The plan from what is on disk now: -> (seconds the waiting jobs need, seconds the current one needs)."""
        movies, _ = queue_movies(s.a, s.base, s.parser, s.extras)
        items, rest, cur, kinds, unknown, rest_pairs = [], 0.0, 0.0, {}, 0, 0.0
        for m in [m for m in movies if m.error is None]:
            key, r = str(m.work), s.results.get(str(m.work))
            k, _ = m.kind_and_why()
            made = m.pairs_made()
            need = max(0, m.count - made) / s.pairs_s + (s.iters * s.step_s if m.solo else 0)
            pairs = "pairs done" if made >= m.count else f"pairs {made} / {m.count}"
            if key == s.current:
                state, cur = "running", need
            elif r is None:
                state, rest = "waiting", rest + need
                rest_pairs += max(0, m.count - made) / s.pairs_s
            else:
                state = "failed" if r.startswith("failed") else "done"
            items.append(dict(label=m.dvd.name, state=state, detail="failed" if state == "failed" else
                              f"{k or 'kind ?'} · {pairs}" + (" · own model" if m.solo else "")))
            if not m.solo and state != "failed":
                if k:
                    kinds.setdefault(k, []).append(str((m.work / "pairs").resolve()))
                else:
                    unknown += 1
        for k in sorted(kinds):
            name = f"ai-{k}-x2"
            st = s.models.get(name, "")
            twork = s.base / f"{name}_training"
            try:                                  # (trained before on all these movies: nothing to do)
                trained = (twork / "finished.json").exists() and set(kinds[k]) <= set(
                    json.loads((twork / "movies.json").read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                trained = False
            if name == s.current:
                state, cur = "running", s.iters * s.step_s
            elif st.startswith(("ok", "done")) or (trained and not st):
                state = "done"
            elif st.startswith("failed"):
                state = "failed"
            else:
                state, rest = "waiting", rest + s.iters * s.step_s
            try:
                earlier = sum(1 for d in json.loads((twork / "movies.json").read_text(encoding="utf-8"))
                              if d not in kinds[k] and (Path(d) / "lr").is_dir())
            except (OSError, ValueError, TypeError):
                earlier = 0
            n = len(kinds[k]) + earlier
            items.append(dict(label=f"model {name}", state=state, detail=f"from {n} movie{'s' * (n > 1)}"
                              + (f" ({earlier} from earlier runs)" if earlier else "")))
        if unknown:
            items.append(dict(label="model for the movies of a kind not known yet", state="waiting",
                              detail=f"{unknown} movie{'s' * (unknown > 1)}"))
            rest += s.iters * s.step_s
        with s.lock:
            s.q = dict(items=items, done=sum(i["state"] == "done" for i in items), total=len(items),
                       rest=rest, cur=cur, rest_pairs=rest_pairs, pairs_s=s.pairs_s)
        return rest, cur

    def start_job(s, key):
        s.current = key
        shutdown_reason(f"upscale_training.py is working on {key if not os.sep in key else Path(key).name} "
                        "(its queue isn't finished): shutting down now stops it; it carries on when run again")
        s.child_file.unlink(missing_ok=True)
        rest, cur = s.refresh()
        el = time.time() - s.t0
        print(f"queue: {s.q['done']} of {s.q['total']} done, about {hms(rest + cur)} left (~{clock_in(rest + cur)}), "
              f"elapsed {hms(el)}", flush=True)
        return rest

    def end_job(s):
        """The speeds the run measured (better than the guesses for the time left)."""
        try:
            st = json.loads(s.child_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = {}
        if (st.get("pairs_per_min") or 0) > 0 and st.get("step", 0) >= 30:
            s.pairs_s = st["pairs_per_min"] / 60
        if (st.get("sec_per_step") or 0) > 0:
            s.step_s = st["sec_per_step"]
        try:
            write_json_durably(s.speeds_file, dict(pairs_s=s.pairs_s, step_s=s.step_s))
        except OSError:
            pass
        s.current = None
        s.child_file.unlink(missing_ok=True)
        s.set(title="Training queue", phase="starting the next one", step=0, total=0, eta=None, speed_text="",
              elapsed=None, metrics={}, series={}, refs={})
        s.refresh()

    def snapshot(s):
        st = None
        if s.current:
            try:
                st = json.loads(s.child_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                st = None                       # (not written yet, or being replaced just then)
        with s.lock:
            out = json.loads(json.dumps(st or s.s))
            q = json.loads(json.dumps(s.q)) if s.q else None
        if q:
            for i in q["items"]:
                if i["state"] == "running" and st and st.get("total"):
                    i["detail"] = ((i["detail"] + " · " if i["label"].startswith("model ") else "")
                                   + f"{st.get('step', 0)} / {st['total']} {st.get('unit', '')}")
            rest = q["rest"]
            if st and (st.get("pairs_per_min") or 0) > 0 and st.get("step", 0) >= 30:
                # (the movies still waiting, at the speed this one is going now)
                rest += q["rest_pairs"] * (q["pairs_s"] / (st["pairs_per_min"] / 60) - 1)
            eta = rest + (st["eta"] if st and st.get("eta") is not None else q["cur"])
            el = time.time() - s.t0
            out["queue"] = dict(items=q["items"], done=q["done"], total=q["total"], eta=eta, elapsed=el,
                                eta_clock=clock_in(eta), frac=el / (el + eta) if el + eta > 0 else 0)
        return json.dumps(out).encode()


def recycle(path):
    """Windows: a file or folder to the Recycle Bin (Windows deletes it for good if it's too big for the bin,
    or on a USB stick or network drive), as dvd_upscale.py --delete-originals does. Elsewhere: deleted.
    True when it's gone."""
    path = Path(path)
    if not path.exists():
        return True
    if os.name != "nt":
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
        return not path.exists()
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
                    ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
                    ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]
    FO_DELETE, FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 3, 4, 16, 64, 1024
    op = SHFILEOPSTRUCTW(None, FO_DELETE, os.path.abspath(path) + "\0", None,   # (the list ends \0\0)
                         FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI, False, None, None)
    rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    return rc == 0 and not op.fAnyOperationsAborted and not path.exists()


def run_child(cmd, cwd, pos, env=None):
    """One run of this script; its exit code (130: stopped with Ctrl+C)."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL,
                            env=dict(os.environ, TRAINING_QUEUE_POS=pos, **(env or {})))
    try:
        return proc.wait()
    except KeyboardInterrupt:
        # (the Ctrl+C reached that run too: it saves where it is and stops; wait for that)
        while True:
            try:
                proc.wait()
                return 130
            except KeyboardInterrupt:
                pass


def shutdown_pc():
    if os.name == "nt":
        subprocess.run(["shutdown", "/s", "/t", "120", "/c", "upscale_training.py: all done"])
        print("The PC turns off in 2 minutes (to cancel: shutdown /a)", flush=True)
    else:
        subprocess.run(["shutdown", "-h", "+2"])


def overview(a, base, parser, extras):
    """What the queue found, before it starts: every movie, its kind, how far its pairs are, and the models."""
    movies, left = queue_movies(a, base, parser, extras)
    ok = [m for m in movies if m.error is None]
    print(f"\nFound {len(ok)} movie{'s' * (len(ok) != 1)} to train on in {base}"
          + (f" ({len(left) + len(movies) - len(ok)} left out, see below)" if left or len(ok) < len(movies) else "") + ":")
    width = max([len(m.label) for m in ok] + [10])
    kinds, unknown, pairs_left = {}, 0, 0
    for n, m in enumerate(ok, 1):
        k, why = m.kind_and_why()
        made = m.pairs_made()
        pairs_left += max(0, m.count - made)
        state = ("pairs done" if made >= m.count else f"pairs {made} of {m.count}" if made else "not started")
        if m.solo:
            state += ", " + ("its own model: done" if finished(m.work) else "then its own model")
        elif k:
            kinds.setdefault(k, []).append(m)
        else:
            unknown += 1
        print(f"  {n:>2}. {m.label:<{width}}  {(k or '?'):<6} ({why})  {state}")
    for label, why in left + [(m.label, m.error) for m in movies if m.error]:
        print(f"   -  {label}: left out: {why}")
    def with_earlier(k, ms):
        # (a model keeps the movies it learned from before: see step 2 of queue_main)
        now = {str((m.work / "pairs").resolve()) for m in ms}
        try:
            before = json.loads((base / f"ai-{k}-x2_training" / "movies.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            before = []
        extra = len({d for d in before if d not in now and (Path(d) / "lr").is_dir()})
        total = len(now) + extra
        return (f"ai-{k}-x2 from {total} movie{'s' * (total > 1)}"
                + (f" ({extra} from earlier runs)" if extra else ""))

    if kinds or unknown:
        print("Models it will make: " + ", ".join(with_earlier(k, v) for k, v in sorted(kinds.items()))
              + (f"{', and ' if kinds else ''}the kinds of {unknown} more movie{'s' * (unknown > 1)} once detected"
                 if unknown else ""))
    if a.delete_originals:
        print("--delete-originals: once each model is made, the DVD and Blu-ray files of its movies and their pairs "
              "go to the Recycle Bin; only the models stay.")
    if ok:
        hours = pairs_left / 3000 * 3 + 2 * (len(kinds) + (1 if unknown else 0))
        print(f"Rough time: {hours:.0f} hours or more (about 3 h of pairs per movie, 2 h of training per model on an "
              "RTX 3060). Ctrl+C stops it at any point; the same command carries on.")
    print(flush=True)


def queue_main(a, parser):
    if a.all is not None and a.queue is not None:
        parser.error("--all or --queue, not both")
    extras = passed_on(sys.argv[1:])
    bad = [t for t in extras if t.split("=", 1)[0] in NOT_IN_QUEUE]
    if bad:
        parser.error(f"{', '.join(bad)}: not with --all / --queue (put --dvd/--bluray/--work in the queue lines; "
                     "the movies' folders and _dvd/_bd names do it for --all)")
    base = Path(a.all).resolve() if a.all is not None else Path(a.queue).resolve().parent
    if not base.is_dir():
        sys.exit(f"folder not found: {base}")
    log_file = base / "training_queue.log"

    def log(text):
        print(text, flush=True)
        try:
            with open(log_file, "a", encoding="utf-8") as fh:
                fh.write(f"{datetime.datetime.now():%Y-%m-%d %H:%M}  {text}\n")
        except OSError:
            pass

    overview(a, base, parser, extras)
    if a.list:
        return 0
    if a.pretrained != "none" and not Path(a.pretrained).exists():
        sys.exit(f"starting model not found: {a.pretrained}\n(also looked in a Training folder next to this script "
                 f"and in the models folder)\nDownload RealESRGAN_x2plus.pth from\n{PRETRAINED_URL}\nand put it in "
                 "the same folder as upscale_training.py (or pass its path with --pretrained). Checked now, before "
                 "the hours of pairs, so the training at the end doesn't fail.")
    global LOCK
    LOCK = hold_lock(base / ".training_queue.lock", f"Another --all / --queue is already running for {base}.")
    keep_awake()
    protect_run(a, "working through its list of movies")
    me = [sys.executable, str(Path(__file__).resolve())]
    results = {}                       # what happened to each movie / model in this run
    status = QueueStatus(a, base, parser, extras, results)
    status.refresh()
    if a.web:
        status.start()                 # (one page for the whole queue; the runs write their state for it)

    def job_env(rest):
        env = dict(TRAINING_QUEUE_REST=f"{rest:.0f}", TRAINING_QUEUE_T0=f"{status.t0:.0f}")
        if a.web:
            env["TRAINING_QUEUE_STATE"] = str(status.child_file)
        return env

    mdir = Path(a.models) if a.models else find_models_dir()

    def clean_up(name, ms, extra_dirs=(), own=None):
        """--delete-originals, once the model `name` is in the models folder: the movies it learned from
        (their DVD, Blu-ray and _training folder) and its own training folder go to the Recycle Bin."""
        if not a.delete_originals:
            return
        if mdir is None or not all((mdir / f"{name}.{e}").exists() for e in ("param", "bin")):
            log(f"{name}: not deleting its movies: the model isn't in the models folder")
            return
        for path in ([x for m in ms for x in (m.dvd, m.bluray, m.work)] + [Path(d).parent for d in extra_dirs]
                     + ([own] if own else [])):
            if path.exists():
                log(f"  {'to the Recycle Bin' if os.name == 'nt' else 'deleted'}: {path}" if recycle(path)
                    else f"  couldn't delete {path} (in use?): delete it by hand")

    def finish(phase, code):
        status.current = None
        status.refresh()
        status.set(title="Training queue", phase=phase, finished=code == 0, eta=0)
        if a.web and code != 130:
            status.hold()
        return code
    log(f"=== {'--all ' + str(base) if a.all is not None else '--queue ' + str(Path(a.queue).resolve())}")

    # 1. the pairs, movie by movie (the list is read again before each: movies can be added while it runs)
    shown = set()
    while True:
        movies, left = queue_movies(a, base, parser, extras)
        for label, why in left + [(m.label, m.error) for m in movies if m.error]:
            if (label, why) not in shown:
                shown.add((label, why))
                log(f"skipped {label}: {why}")
        ok = [m for m in movies if m.error is None]
        todo = [m for m in ok if str(m.work) not in results]
        if not todo:
            break
        m = todo[0]
        n = ok.index(m) + 1
        if m.solo and finished(m.work):
            results[str(m.work)] = "done before"
            log(f"[{n}/{len(ok)}] {m.label}: done before")
            if gate_rejected(m.work):
                log(f"{m.name}: not deleting its movie: the model in use didn't take the new training (it wasn't better)")
            else:
                clean_up(m.name, [m])
            continue
        stages = "pairs,train,export" if m.solo else "pairs"
        log(f"[{n}/{len(ok)}] {m.label}: " + ("pairs, training and its own model" if m.solo else "pairs"))
        rest = status.start_job(str(m.work))
        rc = run_child(me + extras + m.args + ["--stages", stages], base, f"movie {n} of {len(ok)}: {m.label}", job_env(rest))
        status.end_job()
        if rc in STOPPED_CODES:
            log(f"stopped during {m.label}. Run the same command again to carry on.")
            return finish("stopped (run the same command again to carry on)", 130)
        results[str(m.work)] = "ok" if rc == 0 else f"failed (exit code {rc}; its messages are above)"
        log(f"[{n}/{len(ok)}] {m.label}: {results[str(m.work)]}")
        if m.solo and rc == 0:
            if gate_rejected(m.work):
                log(f"{m.name}: not deleting its movie: the model in use didn't take the new training (it wasn't better)")
            else:
                clean_up(m.name, [m])

    # 2. one model per kind of movie, from the pairs of all the movies of that kind
    movies, _ = queue_movies(a, base, parser, extras)
    groups = {}
    for m in movies:
        if m.error is None and not m.solo and results.get(str(m.work)) == "ok":
            kind = m.kind() or m.kind_and_why()[0]     # (upscaler.json, else --type / its folder / name tag)
            if kind:
                groups.setdefault(kind, []).append(m)
            else:
                log(f"{m.label}: its kind of movie isn't known: left out of the training")
    for i, (kind, ms) in enumerate(sorted(groups.items()), 1):
        name = f"ai-{kind}-x2"
        twork = base / f"{name}_training"
        try:
            before = json.loads((twork / "movies.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            before = None
        # a model only gains movies: those it learned from before stay in (while their pairs are there), also
        # when this run lists fewer (a queue of just the new movie); delete its _training folder to start over
        now = {str((m.work / "pairs").resolve()) for m in ms}
        earlier = {d for d in (before or []) if d not in now and (Path(d) / "lr").is_dir()}
        dirs = sorted(now | earlier)
        what = (f"{name} from {len(dirs)} {TYPE_NAMES.get(kind, kind)} movie{'s' * (len(dirs) > 1)}"
                + (f" ({len(earlier)} of them from earlier runs)" if earlier else ""))
        if finished(twork, dirs):
            results[name] = status.models[name] = "done before (same movies)"
            if gate_rejected(twork):
                results[name] = status.models[name] = "done before (not better than the model in use)"
                log(f"model {i}/{len(groups)} {what}: done before, but the model in use stays (the new one wasn't better): "
                    "not deleting its movies")
                continue
            log(f"model {i}/{len(groups)} {what}: done before, from the same movies")
            clean_up(name, ms, earlier, twork)
            continue
        # it carries on from the model in use (its .pth, kept in the models folder), so each run builds on all the
        # training before it instead of starting from x2plus again (which also covers movies it learned from that
        # are gone, --delete-originals). The movies the model in use hasn't learned yet are the "new" ones.
        start_from = mdir / f"{name}.pth" if mdir and (mdir / f"{name}.pth").exists() else None
        try:
            accepted = json.loads((twork / "accepted.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            try:                                   # (made before this was kept: the movies of its last training)
                accepted = json.loads((twork / "finished.json").read_text(encoding="utf-8")).get("pairs_from") or []
            except (OSError, ValueError, AttributeError):
                accepted = []
        if not start_from:
            accepted = []
        new = [d for d in dirs if d not in accepted]
        given = {t.split("=", 1)[0] for t in extras}
        steps = (max(STEPS_PER_MOVIE, STEPS_PER_MOVIE * len(new)) if start_from
                 else max(2 * STEPS_PER_MOVIE, STEPS_PER_MOVIE * len(dirs)))
        more = [] if "--iters" in given else ["--iters", str(steps)]
        if start_from and "--lr" not in given:
            more += ["--lr", f"{a.lr / 2:g}"]      # (a gentler start: it is already a trained model)
        if start_from and new and len(new) < len(dirs):
            more += ["--focus-from", *new]
        twork.mkdir(parents=True, exist_ok=True)
        (twork / "gate.json").unlink(missing_ok=True)
        if before is not None and before != dirs and (twork / "run").exists():
            # (other movies than the training there was started with: train again from the start, on all of them)
            old = twork / "run_old"
            if old.exists():
                shutil.rmtree(old, ignore_errors=True)
            os.replace(twork / "run", old)
            (twork / "finished.json").unlink(missing_ok=True)
            log(f"{name}: the movies changed since its last training ({len(before)} -> {len(dirs)}): training it "
                "again on all of them (the earlier one is in run_old)")
        write_json_durably(twork / "movies.json", dirs)
        log(f"model {i}/{len(groups)}: {what}: " + ", ".join(Path(d).parent.name.removesuffix("_training") for d in dirs))
        rest = status.start_job(name)
        if start_from:
            log(f"{name}: carrying on from the {name} in use, {len(new)} movie{'s' * (len(new) != 1)} new to it"
                + (f", {steps} steps" if not "--iters" in given else ""))
        else:
            log(f"{name}: a new model from x2plus" + (f", {steps} steps" if not "--iters" in given else ""))
        rc = run_child(me + extras + ["--type", kind, "--work", str(twork), "--stages", "train,export",
                                      "--pairs-from", *dirs] + more
                       + (["--pretrained", str(start_from)] if start_from else []),
                       base, f"model {i} of {len(groups)}: {what}", job_env(rest))
        status.end_job()
        if rc in STOPPED_CODES:
            log(f"stopped while training {name}. Run the same command again to carry on.")
            return finish("stopped (run the same command again to carry on)", 130)
        kept = False
        try:
            kept = rc == 0 and json.loads((twork / "gate.json").read_text(encoding="utf-8")).get("accepted") is False
        except (OSError, ValueError, AttributeError):
            pass
        results[name] = status.models[name] = (("ok, but not better than the model in use: that one stays" if kept
                                                else "ok") if rc == 0 else f"failed (exit code {rc}; its messages are above)")
        log(f"model {i}/{len(groups)} {name}: {results[name]}")
        if rc == 0 and kept:
            if a.delete_originals:
                log(f"{name}: not deleting its movies: the model in use hasn't learned the new ones yet")
        elif rc == 0:
            clean_up(name, ms, earlier, twork)

    failed = [k for k, v in results.items() if v.startswith("failed")]
    log(f"=== finished: {len(groups)} model(s) for {sum(len(v) for v in groups.values())} movie(s)"
        + (f"; {len(failed)} failed (see {log_file.name})" if failed else "")
        + ".  Upscale with:  python dvd_upscale.py <movie> --trained  (it picks ai-<kind>-x2 for each movie)")
    if a.shutdown:
        allow_shutdown()                    # (this run's own shutdown block goes first)
        shutdown_pc()
    return finish("all done" + (f", {len(failed)} failed (see {log_file.name})" if failed else ""), 1 if failed else 0)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("several movies, one after another (one model per kind of movie)")
    g.add_argument("--all", nargs="?", const=".", metavar="FOLDER",
                   help="every DVD + Blu-ray pair in FOLDER (default: the current one) and its anime/live/cgi/3d/vhs "
                        "folders, found by name: Shrek_dvd.mkv + Shrek_bd.mkv")
    g.add_argument("--queue", nargs="?", const="training_queue.txt", metavar="FILE",
                   help="the movies listed in FILE (default training_queue.txt), one per line: --dvd ... --bluray ...")
    g.add_argument("--shutdown", action="store_true", help="--all / --queue: turn the PC off when everything is finished")
    p.add_argument("--no-guard", action="store_true",
                   help="Windows: don't block shutdowns/restarts or turn off QuickEdit while it runs (sleep is still prevented)")
    g.add_argument("--delete-originals", action="store_true",
                   help="--all / --queue: once a model is made, its movies' DVD and Blu-ray files and their pairs "
                        "(the _training folders) go to the Recycle Bin; only the models stay. A later movie of the "
                        "same kind then continues training from the model")
    g.add_argument("--list", action="store_true",
                   help="--all / --queue: just show the movies it found, their kind and how far they are, then stop")
    g.add_argument("--pairs-from", nargs="+", metavar="FOLDER", help=argparse.SUPPRESS)   # (the queue: several movies' pairs)
    g.add_argument("--focus-from", nargs="+", metavar="FOLDER", help=argparse.SUPPRESS)   # (the queue: the new movies' pairs)
    p.add_argument("--dvd", help="the DVD movie file")
    p.add_argument("--bluray", help="the Blu-ray movie file (the same film)")
    p.add_argument("--work", help="folder for the pairs and the training (default: <DVD name>_training next to the DVD file)")
    p.add_argument("--stages", default="pairs,train,export", help="which steps to run (default: pairs,train,export)")
    p.add_argument("--web", type=int, default=8643, metavar="PORT",
                   help="live progress page for a browser or phone (default port 8643; 0 turns it off)")
    g = p.add_argument_group("step 1: pairs")
    g.add_argument("--count", type=int, default=3000, help="pairs wanted (default 3000)")
    g.add_argument("--fresh", action="store_true", help="start the pairs over (deletes those already made)")
    g.add_argument("--denoise", default=None,
                   help="hqdn3d for the DVD frames (default: what dvd_upscale.py uses for this movie's type)")
    g.add_argument("--dar", default=None, help="force the DVD picture aspect, e.g. 16:9 (passed on to dvd_upscale.py --dar)")
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
    g.add_argument("--pair-workers", type=int, default=None, help="frames made at the same time (default: from the number of processor cores, 2 to 6)")
    g.add_argument("--seed", type=int, default=1)
    g = p.add_argument_group("step 2: train")
    g.add_argument("--arch", choices=["rrdb", "compact"], default="rrdb",
                   help="rrdb: Real-ESRGAN x2plus, best quality, slow (default). compact: a small fast network "
                        "(upscales several times quicker) that learns from zero, so it needs --iters 100000+")
    g.add_argument("--compact-feat", type=int, default=64, help="compact: channels (default 64)")
    g.add_argument("--compact-convs", type=int, default=16, help="compact: conv layers (default 16; fewer = faster)")
    g.add_argument("--pretrained", default=None,
                   help="the starting model (default: RealESRGAN_x2plus.pth next to this script for rrdb, none for compact)")
    g.add_argument("--iters", type=int, default=20000)
    g.add_argument("--batch", type=batch_arg, default=8,
                   help="patches per step (default 8). 'auto' tries bigger and bigger batches and keeps the largest "
                        "that fits in the graphics card's memory: use it when the monitor shows the GPU under-used")
    g.add_argument("--patch", type=int, default=96, help="DVD patch size in pixels (default 96)")
    g.add_argument("--lr", type=float, default=5e-5)
    g.add_argument("--perceptual", type=float, default=0.0)
    g.add_argument("--gan", type=float, default=0.0)
    g.add_argument("--disc", choices=["unet", "patch"], default="unet", help="discriminator for --gan (default: Real-ESRGAN's U-Net)")
    g.add_argument("--crops", type=int, default=4, help="patches cut from each pair per step (default 4: --batch 8 loads 2 pairs per step)")
    g.add_argument("--rotate", action="store_true", help="also turn patches by 90 degrees (off: a DVD's blur has a direction)")
    g.add_argument("--min-patch-match", type=float, default=0.8,
                   help="skip training patches where the two discs don't line up locally (something moved between the "
                        "frames), which would teach the model to blur. 0 turns the check off (default 0.8)")
    g.add_argument("--lpips", action="store_true", help="also judge held-out frames with LPIPS (pip install lpips)")
    g.add_argument("--checkpoint", action="store_true", help="trade speed for much less GPU memory")
    g.add_argument("--train-workers", type=int, default=None, help="data loading processes (default: from the number of processor cores, 2 to 8)")
    g.add_argument("--gpu-memory", type=float, default=1.0, metavar="FRACTION",
                   help="--batch auto grows the batch until it no longer fits, up to this share of the graphics card's memory "
                        "(default 1.0: everything that fits; lower it to leave room for other programs)")
    p.add_argument("--gpu-test", action="store_true",
                   help="just list the NVIDIA GPUs and time each alone and together (to check a new card), then exit")
    g.add_argument("--gpus", default="all", metavar="LIST",
                   help="NVIDIA GPUs to train on: 'all' (default) or numbers like 0,1 (as nvidia-smi lists them). "
                        "With 2 or more, each step is split across them (needs matching cards; the slowest sets the pace)")
    g.add_argument("--no-channels-last", action="store_true", help="turn off the GPU-friendly memory layout (on by default with CUDA)")
    g.add_argument("--save-every", type=int, default=1000)
    g.add_argument("--val-crop", type=int, default=384, help="held-out frames are judged on this centre square (DVD px)")
    g = p.add_argument_group("the kind of movie (as dvd_upscale.py --type)")
    g.add_argument("--type", choices=["auto", *TYPE_NAMES], default="auto",
                   help="anime, cgi (3D animation), live (live action) or vhs (a tape). Default auto: dvd_upscale.py "
                        "detects it, as it does when upscaling, and the model is named after it (ai-anime-x2, ai-cgi-x2...)")
    g.add_argument("--analyze", action="store_true",
                   help="just ask dvd_upscale.py what --dvd is and how it prepares its frames, then exit "
                        "(a minute or two; no Blu-ray needed)")
    g = p.add_argument_group("step 3: export")
    g.add_argument("--name", default=None,
                   help="model name for dvd_upscale.py --model (default: from the kind of movie, e.g. ai-anime-x2)")
    g.add_argument("--models", help="the upscaler's models folder (default: found next to realesrgan-ncnn-vulkan)")
    g.add_argument("--keep-any", action="store_true",
                   help="use the new model even when it scores lower than the one in use (default: it replaces it "
                        "only when it scores higher on the held-out frames of all its movies)")
    return p


def main():
    p = build_parser()
    a = p.parse_args()
    a.stopped = False
    cores = os.cpu_count() or 4
    if a.train_workers is None:
        a.train_workers = max(2, min(8, cores // 2))
    if a.pair_workers is None:
        a.pair_workers = max(2, min(6, cores // 3))
    if a.pretrained is None:
        a.pretrained = str(find_pretrained()) if a.arch == "rrdb" else "none"

    if a.gpu_test:
        return gpu_test(a)
    if a.analyze:
        return analyze(a)
    if a.all is not None or a.queue is not None:
        return queue_main(a, p)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    if not stages or set(stages) - {"pairs", "train", "export"}:
        p.error("--stages is a list of: pairs, train, export")
    if "pairs" in stages and not (a.dvd and a.bluray):
        p.error("--dvd and --bluray are needed to make the pairs")
    if not a.work and not a.dvd:
        p.error("--work (or --dvd, to put it next to the DVD) is needed")
    if a.pairs_from and "pairs" in stages:
        p.error("--pairs-from is for training on pairs already made: --stages train,export")
    work = default_work(a)
    pairs_dir, run_dir = work / "pairs", work / "run"
    roots = [Path(x) for x in a.pairs_from] if a.pairs_from else [pairs_dir]
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
    pos = os.environ.get("TRAINING_QUEUE_POS")             # ("2 of 5: Shrek", when run by --all / --queue)
    title = "DVD to Blu-ray training" + (f" ({pos})" if pos else "")
    if os.environ.get("TRAINING_QUEUE_STATE"):            # (the queue's page shows this run)
        PROG = Progress(title, 0).write_to(os.environ["TRAINING_QUEUE_STATE"])
    else:
        PROG = Progress(title, a.web).start() if a.web else None
    if "train" in stages and not torch.cuda.is_available():
        print("WARNING: no CUDA GPU found: training would take weeks on the processor")
    work.mkdir(parents=True, exist_ok=True)
    global LOCK
    LOCK = hold_lock(work / ".lock", f"Another run is already using the work folder '{work}'. Wait for it to finish (or stop it) first.")
    if "pairs" in stages or "train" in stages:
        keep_awake()
        protect_run(a, "training an upscaling model")
    say(f"working folder: {work}")
    setup_kind(a, work, need_filters="pairs" in stages)
    titles = {"pairs": "making the DVD / Blu-ray pairs", "train": "training", "export": "making the model for the upscaler"}
    for n, stage in enumerate(stages, 1):
        say(f"=== step {n} of {len(stages)}: {titles[stage]} ===")
        if PROG:
            PROG.reset(f"Step {n} of {len(stages)}: {titles[stage]}")
        if stage == "pairs":
            ok = stage_pairs(a, pairs_dir)
        elif stage == "train":
            ok = stage_train(a, roots, run_dir)
        else:
            ok = stage_export(a, run_dir, work, roots)
        if not ok:
            return 130 if a.stopped else 1                 # (stopped with Ctrl+C / failed: for --all and --queue)
    if PROG:
        PROG.set(phase="all done", finished=True, eta=0)
    if "export" in stages:
        write_json_durably(work / "finished.json", {"model": model_name(a, a.kind), "stages": stages,
                                                     "pairs_from": sorted(map(str, roots)), "at": time.time()})
        say(f"All done. Upscale with:  {upscale_command(a)}")
    if PROG and not os.environ.get("TRAINING_QUEUE_POS"):
        PROG.hold()
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):      # a name the console can't show becomes '?', not a crash
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    # ffmpeg/ffprobe next to this script work even when the movies are elsewhere (as with dvd_upscale.py)
    _here = str(Path(__file__).resolve().parent)
    if any(shutil.which(t, path=_here) for t in ("ffmpeg", "ffprobe")):
        os.environ["PATH"] = _here + os.pathsep + os.environ.get("PATH", "")
    try:
        rc = main()
        sys.exit(rc if isinstance(rc, int) else 0)
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to carry on.", file=sys.stderr)
        sys.exit(130)
    except (subprocess.CalledProcessError, RuntimeError) as e:
        sys.exit(f"\nFailed: {e}\nRun the same command again to carry on from where it stopped.")
    except OSError as e:                          # disk full, file in use, missing folder...
        sys.exit(f"\nFailed: {e}\nFix that and run the same command again: the saved pairs and training are kept.")
