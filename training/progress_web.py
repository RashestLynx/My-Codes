"""A tiny live progress page for make_pairs.py and train.py (--web PORT).

Open  http://<this PC's address>:PORT  in a browser on this PC or on a phone on the same Wi-Fi
(or through Tailscale, if you use it). Read-only: it only shows numbers. No extra packages.
If Windows asks whether to let Python through the firewall, allow it on private networks.
"""
import datetime, http.server, json, socket, sys, threading, time

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

    def hold(self, seconds=300):
        """Keep the page up after the work ends, so a phone can still see 'finished'."""
        if not getattr(self, "started", False):
            return
        print(f"(the progress page stays up for {seconds // 60} more minutes; Ctrl+C closes it)", flush=True)
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
