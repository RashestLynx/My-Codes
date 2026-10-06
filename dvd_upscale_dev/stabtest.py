"""Shaky test clips and measurements for --stabilize.

make(path, secs, fps, telecine=False): a textured scene with hand-held shake (fast jitter plus a
  slow pan), a magenta marker that moves with the scene, and a frame number in a fixed band at the
  top (like a camcorder date stamp): 12 cells of 60x40 across a 720x480 frame, white = bit set.
measure(path, zoom): per output frame (frame number, marker x, marker y), positions in the
  640x480 square-pixel picture; zoom: the stabilizer's zoom in percent (to find the band).
jitter(rows): RMS of the marker's fast movement (position minus its 31-frame moving average).
"""
import math, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import barcode

W, H = 640, 480


def make(path, secs=12, fps="30000/1001", telecine=False):
    canvas = HERE / "canvas.png"
    if not canvas.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                        "mandelbrot=s=880x620:start_scale=1.5:end_scale=1.5:maxiter=200",
                        "-frames:v", "1", str(canvas)], check=True)
    rate = "24000/1001" if telecine else fps
    band = f"color=c=black:s=720x40:r={rate},format=yuv420p,geq=lum='{barcode.lum_expr()}':cb=128:cr=128"
    shake = ("crop=720:480:x='60+0.6*n+14*sin(n*0.9)+6*sin(n*2.3+1)'"
             ":y='60+11*sin(n*1.3+0.5)+5*cos(n*3.1)'")
    graph = (f"[0:v]format=yuv420p[c];[c][2:v]overlay=430:200:shortest=1,{shake}[s];"
             f"[s][1:v]overlay=0:10:shortest=1,setsar=8/9,format=yuv420p"
             + (",telecine=first_field=t:pattern=23" if telecine else "") + "[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-loop", "1", "-framerate", rate,
                    "-i", str(canvas), "-f", "lavfi", "-i", band, "-f", "lavfi", "-i",
                    f"color=c=0xFF00FF:s=24x24:r={rate}", "-f", "lavfi", "-i",
                    f"sine=f=440:d={secs}", "-filter_complex", graph, "-map", "[v]", "-map", "3:a",
                    "-t", str(secs), "-c:v", "mpeg2video", "-b:v", "8M", "-g", "15", "-bf", "2",
                    *(["-flags", "+ilme+ildct", "-top", "1"] if telecine else []),
                    "-c:a", "ac3", "-f", "dvd", str(path)], check=True)


def measure(path, zoom=0.0):
    raw = subprocess.run(["ffmpeg", "-v", "info", "-nostdin", "-copyts", "-i", str(path),
                          "-map", "0:v:0", "-vf", f"scale={W}:{H}:flags=area,showinfo",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True)
    pts = [float(x.split(b"pts_time:")[1].split()[0])
           for x in raw.stderr.splitlines() if b"pts_time:" in x]
    z = 1 + zoom / 100
    fs = W * H * 3
    rows = []
    for i in range(len(raw.stdout) // fs):
        f = raw.stdout[i * fs:(i + 1) * fs]
        n = 0
        for k in range(12):                   # band cells: 720/12 = 60 px, 53.3 at 640 wide
            x = W / 2 + ((k + 0.5) * W / 12 - W / 2) * z
            y = H / 2 + (30 - H / 2) * z
            s = sum(f[(int(y) + dy) * W * 3 + (int(x) + dx) * 3 + 1]
                    for dy in range(-4, 5) for dx in range(-4, 5)) / 81
            n |= (s > 125) << k
        sx = sy = c = 0
        for yy in range(60, H, 2):            # the magenta marker (below the band)
            row = f[yy * W * 3:(yy + 1) * W * 3]
            for xx in range(0, W, 2):
                r, g, b = row[3 * xx], row[3 * xx + 1], row[3 * xx + 2]
                if r > 150 and b > 150 and g < 90:
                    sx, sy, c = sx + xx, sy + yy, c + 1
        rows.append((pts[i] if i < len(pts) else None, n,
                     sx / c if c else None, sy / c if c else None))
    return rows


def jitter(rows):
    xs = [r[2] for r in rows]
    ys = [r[3] for r in rows]
    out = []
    for v in (xs, ys):
        d = []
        for i in range(15, len(v) - 15):
            win = [u for u in v[i - 15:i + 16] if u is not None]
            if v[i] is not None and win:
                d.append(v[i] - sum(win) / len(win))
        out.append(math.sqrt(sum(e * e for e in d) / len(d)) if d else float("nan"))
    return out


def order(rows):
    seq = [r[1] for r in rows]
    return seq, [(i, seq[i - 1], seq[i]) for i in range(1, len(seq)) if seq[i] != seq[i - 1] + 1]


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make":
        make(sys.argv[2], telecine=len(sys.argv) > 3 and sys.argv[3] == "telecine")
    else:
        rows = measure(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.0)
        seq, breaks = order(rows)
        jx, jy = jitter(rows)
        print(f"{len(rows)} frames, first {seq[:4]}, breaks {len(breaks)} {breaks[:4]}, "
              f"jitter x {jx:.2f} px, y {jy:.2f} px")
