"""Frame-number barcodes: 12 vertical bars (bit 0 leftmost), white = 1. decode(file) -> [(pts, n)]."""
import subprocess, sys
BITS = 12
def lum_expr():
    # bar k covers X in [k*W/12, (k+1)*W/12): white if bit k of N is set
    return f"if(mod(floor(N/pow(2,floor(X*{BITS}/W))),2),235,16)"
def decode(path, w=48, h=8):
    """Each output frame shrunk to 48x8 grey: bar k is columns 4k..4k+3 (use the middle two)."""
    raw = subprocess.run(["ffmpeg", "-v", "info", "-nostdin", "-copyts", "-i", path, "-map", "0:v:0",
                          "-vf", f"scale={w}:{h}:flags=area,format=gray,showinfo", "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-"], capture_output=True)
    pts = [float(x.split(b"pts_time:")[1].split()[0]) for x in raw.stderr.splitlines() if b"pts_time:" in x]
    fs = w * h
    out = []
    for i in range(len(raw.stdout) // fs):
        f = raw.stdout[i * fs:(i + 1) * fs]
        row = f[(h // 2) * w:(h // 2 + 1) * w]
        n = sum((1 << k) for k in range(BITS) if (row[4 * k + 1] + row[4 * k + 2]) / 2 > 125)
        out.append((pts[i] if i < len(pts) else None, n))
    return out
if __name__ == "__main__":
    for p, n in decode(sys.argv[1])[:int(sys.argv[2]) if len(sys.argv) > 2 else 10**9]:
        print(p, n)
