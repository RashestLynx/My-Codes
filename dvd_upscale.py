#!/usr/bin/env python3
r"""
DVD -> near-Blu-ray upscaler (ffmpeg + Real-ESRGAN ncnn Vulkan).

Per chunk (1440 output frames for anime, 480 for live/cgi/vhs; cuts land on exact frames):
  auto-detect film/video -> deinterlace/IVTC -> square pixels -> light denoise
  -> AI upscale -> resize to 1080p -> BT.601->BT.709 color fix -> sharpen
  -> H.264 (HEVC with --hevc)
Chunks are joined, then original audio/subs/chapters are muxed back in.

Needs: ffmpeg + ffprobe 5.1 or newer, and realesrgan-ncnn-vulkan (unless --fast).
Live action and 3D CGI work best with realesrgan-x2plus.param/.bin added to the upscaler's models
folder.

What's in the file is detected first (--type auto, the default), from frames sampled across the
whole movie, and printed with the reasons:
- anime: drawn animation (anime, cartoons, hand-drawn Disney) -> realesr-animevideov3
- live: live action (and stop-motion) -> realesrgan-x2plus
- cgi: 3D animation (Pixar, DreamWorks...) -> the live-action settings (they keep fur, hair and
  fabric textures that the anime model would flatten)
- vhs: a VHS capture or a VHS-to-DVD transfer -> the VHS handling below
If it isn't sure it says so and takes the safe choice (live; DVD rather than VHS). --type
anime/live/cgi/vhs overrides it, and so does a folder of that name, or (not in such a folder) a
name starting with it and _ or -: cgi_Shrek.mkv. A resumed movie keeps the
type it started with.
Output: .mkv (keeps everything) or .mp4 (for Apple devices/browsers; DVD subtitles can't be kept).
Video is H.264 8-bit by default so it plays on any player/TV/phone; --hevc gives smaller
HEVC 10-bit files for newer devices.

  python dvd_upscale.py movie.mkv --analyze              # just report what it detects
  python dvd_upscale.py movie.mkv --test 60              # 60s preview ("... 1080p test.mkv")
  python dvd_upscale.py movie.mkv                        # full run (resumable), saved as
                                                         # "1080p Upscale\movie 1080p.mkv"
  python dvd_upscale.py movie.mkv out.mkv                # ...or under a name of your own
  python dvd_upscale.py --clip movie.mkv 15:00 30        # cut a 30 s sample (no re-encoding)
  python dvd_upscale.py --clip movie.mkv 15:00 10 --upscale    # ...and upscale just that piece

VHS captures (--type vhs, normally detected by itself): a capture card's 720x480 29.97i recording
(.mpg/.avi/.mkv/.mp4/.ts) or a VHS-to-DVD transfer, NTSC or PAL:
  python dvd_upscale.py tape.mpg "tape 1080p.mkv" --type vhs --analyze   # what it finds
  python dvd_upscale.py tape.mpg "tape 1080p.mkv" --type vhs
- It tells movie tapes from camcorder/TV tapes by itself. Movie tapes (film with 3:2 pulldown)
  become 23.976 fps, upscaled with realesrgan-x2plus. Camcorder tapes become 59.94 fps (one frame
  per field, so motion stays as smooth as on the tape) with realesr-general-dn50-x4v3, which isn't
  in the official download either: unzip realesr-general-dn50-x4v3.zip into the upscaler's models
  folder (without it, realesrgan-x2plus is used: slower).
- Output is 1440x1080 (4:3) unless --height / --dar say otherwise. Before the upscale it crops to the 4:3 picture, moves the
  colour back onto the picture (VHS colour sits a few pixels right and a line low; measured per
  tape, or --chroma-delay), denoises strongly and blacks out the ragged left/right edges and the
  head-switching noise at the bottom (--mask).
- A very still sample (a long title card) can hide the film cadence: if a movie tape is reported
  as video, pass --mode telecine. --mode interlaced forces the camcorder handling.
- Cartoon tapes get the movie-tape settings too: on simulated Dragon Ball Z tapes they kept line
  weight and colours closer than the anime model (which thickened outlines, shifted flat colours).
- --stabilize steadies a shaky camcorder tape: the camera shake is measured once over the whole
  video (ffmpeg's vid.stab; on Windows the gyan.dev "full" build has it), then smoothed out with
  the picture zoomed in 4% (--stabilize strong: 8%) so no moving edges show. Slow pans stay.
- Faces are restored after the upscale (on by itself for live action and VHS/home video once
  its extras are installed; --faces for 3D animation too, --no-faces for none; --faces [strength]):
  GFPGAN 1.4 (or --face-model codeformer, licensed for non-commercial use only) redraws each
  face it finds, steadied from frame to frame, as 0.6 of the final picture's face (or the
  strength given, up to the AI frames' share, --ai-blend: 0.75 for live and VHS). Needs pip
  install -U onnxruntime-directml opencv-python-headless numpy (onnxruntime-gpu for NVIDIA with
  CUDA and cuDNN; plain onnxruntime runs on the processor, very slowly) and the model files in a
  face_models folder next to this script; the run says what is missing and where to get it.

Every movie in a folder, one after another:
  python dvd_upscale.py --all                  # all movies in the current folder
  python dvd_upscale.py --all "D:\Movies" --shutdown
- Movies in a subfolder named "anime", "live", "cgi" or "vhs" get that --type, whatever else you
  pass; so do loose movies whose name starts with that tag and _ or - (cgi_Shrek.mkv,
  anime-DBZ.mkv). Other loose movies are detected one by one (or get the --type you give, e.g.
  --all --type anime). Other options (--hevc ...) are passed on to every movie.
- Finished movies go into a "1080p Upscale" folder inside the movies folder (movies from the
  "anime"/"live"/"cgi"/"vhs" subfolders into "1080p Upscale\anime", "...\live" ...), so they
  are never picked up again: Movie.mkv becomes "1080p Upscale\Movie 1080p.mkv". The work folder is
  "Movie_work" next to the original. A finished "Movie 1080p.mkv" right next to the
  original (made earlier by hand) also counts as done.
- Skipped: its own "... 1080p" outputs, half-written files, files under 5 minutes (test
  previews, trailers, extras; not in "vhs", home videos are often captured as short clips),
  files that are already HD, .VOB pieces of a DVD folder (rip the disc with MakeMKV) and files
  still being copied (picked up once the copy is done).
- The folder is checked again before each movie, so you can drop more movies in while it runs.
Everything below about the queue (resuming, Ctrl+C, sleep, log) applies to --all as well.
Add --delete-originals (to --all or --queue) to free space as you go: after each movie is
finished and checked (full length, all audio/subtitle tracks, plays cleanly, the picture matches
the original's at points through the movie), the original goes to the Recycle Bin and the
movie's work folder is deleted. A movie finished in an earlier run has its original deleted
too, once it is checked. (On USB sticks, network drives, or
if it's bigger than the Recycle Bin, Windows deletes it for good.) If the check fails, the
original is kept and the log says why. Never done for --test runs.

Queue (a list of movies with their own options):
  python dvd_upscale.py --queue                # runs queue.txt in the current folder
  python dvd_upscale.py --queue --shutdown     # ...and turns the PC off when it's done
  python dvd_upscale.py --queue other.txt      # a different list
queue.txt has one movie per line, exactly what you'd type after "python dvd_upscale.py":
  "DRAGONBALL_Z_FUSION_REBORN.mkv" "DRAGONBALL_Z_FUSION_REBORN 1080p.mkv" --work dbz
  "C:\Movies\Spider-Man.mkv" "Spider-Man 1080p.mkv" --type live --work spiderman   # comment
- A movie counts as done when its output exists and is as long as the source, so running the
  queue again continues where it stopped (the movie in progress resumes from its last
  finished chunk). A short --test preview with the same name isn't mistaken for the movie.
- If a movie fails, the queue logs it and moves on; the reason is shown in the window.
- The list is re-read before each movie, so you can add lines while it runs.
- Ctrl+C stops the whole queue; run it again to continue.
- Lines starting with # are ignored, and so is anything after a # outside quotes. Use straight
  quotes "like this" around names with spaces. Paths are relative to the queue file's folder.
- Each movie needs its own --work folder; if a line has none, "<output name>_work" next to the
  output is used (or, if a run of that same movie was started by hand or with --all, the folder
  it was started in, so it continues instead of starting over).
- Two lines with the same output file: the second is skipped. A finished movie records which
  file it was made from, and is never replaced by the upscale of a different file.
- Each movie starts with the processor's and NVIDIA GPU's model, and every chunk's progress line
  is followed by their load during that chunk: CPU busy %, GPU busy %, clock, temperature, power
  and video memory, each as average and peak. When the movie's chunks are done, the same for the
  whole run. The --phone page shows them too (last chunk and whole run, next to the live values).
- While it runs the PC is kept from going to sleep (single movies too). Keep it plugged in, and if you
  close the lid, set the lid action to "Do nothing" for when it's plugged in.
- Also while it runs (Windows; --no-guard turns this off): a click in the window can't freeze the
  run (QuickEdit is off until it ends), keys typed meanwhile are thrown away instead of being run
  as a command afterwards, and a shutdown or restart stops at Windows' "This app is preventing
  shutdown" screen with the progress (Shut down anyway still works; --shutdown is let through).
  At the start it warns if the laptop is on battery, if closing the lid would sleep it, or if
  Windows Update is waiting to restart. Ctrl+C still stops the run on purpose.
Progress is logged to queue_log.txt next to the queue file.

Progress on your phone (iPhone or Android, any browser): on by itself for every run
(--no-phone turns it off). It prints an address like http://192.168.1.20:8642 : open it on a
phone on the same Wi-Fi (add it to the home screen to keep it handy). It shows the movie, % done, time left for this movie
and all of them, the current step and the latest output, refreshed every 3 s, and the NVIDIA
graphics card's load, temperature, power, memory and clock (from nvidia-smi). --phone 8650
uses another port. The first time, Windows asks whether Python may use the network: allow it
on private networks (the Wi-Fi must be set as a Private network in Windows).

After changing this script: python dvd_upscale.py --self-test checks its own logic in a few
seconds (the queue file, the folder scan, the detection decision, face tracking, GPU steps).
"""
import argparse, json, math, operator, os, re, shutil, signal, statistics, struct, subprocess, sys
import atexit, tempfile
import threading, time, types
import http.server, socket
from bisect import bisect_right as _bisect
from collections import Counter, deque
from operator import add, sub
from fractions import Fraction
from pathlib import Path

OUTPUT_TAG = "Upscaled by dvd_upscale.py"


def nostdin(cmd):
    """ffmpeg reads the console keyboard ('q' quits, 'c'/'d' wait for input): never let it."""
    cmd = [str(c) for c in cmd]
    return cmd[:1] + ["-nostdin"] + cmd[1:] if Path(cmd[0]).stem.lower() == "ffmpeg" else cmd


def run(cmd, cwd=None):
    subprocess.run(nostdin(cmd), check=True, stdin=subprocess.DEVNULL, cwd=cwd)


def capture(cmd):
    # ffmpeg writes UTF-8 (file names): Windows' default code page would choke on some letters
    p = subprocess.run(nostdin(cmd), capture_output=True, stdin=subprocess.DEVNULL)
    return p.returncode, (p.stdout + p.stderr).decode("utf-8", "replace")


def eta_text(secs):
    """'1 h 22 min left, done ~4:51 PM', with the weekday when it ends on another day."""
    m = max(1, round(secs / 60))
    left = f"{m // 60} h {m % 60} min" if m >= 60 else f"{m} min"
    end = time.localtime(time.time() + secs)
    clock = time.strftime("%I:%M %p", end).lstrip("0")
    if time.strftime("%Y%m%d", end) != time.strftime("%Y%m%d"):
        clock = time.strftime("%a ", end) + clock
    return f"{left} left, done ~{clock}"


def status_line(msg=""):
    """A progress line that keeps being overwritten, in a console window only ("" clears it)."""
    phone_line(msg)
    if not sys.stdout.isatty():
        return
    width = max(20, shutil.get_terminal_size((80, 24)).columns - 1)  # a full line would wrap
    sys.stdout.write("\r" + msg[:width].ljust(width) + "\r")
    sys.stdout.flush()


def say(msg):
    status_line()
    print(msg, flush=True)
    if msg.startswith("["):                 # ([3/40] ...: the progress, for the shutdown screen)
        shutdown_reason(f"dvd_upscale.py is upscaling {_GUARD['movie'] or 'a movie'} "
                        f"({msg.split(' - ')[-1].strip()}): shutting down now loses the chunk "
                        "in progress (finished chunks are kept)")


# ---- --phone: the progress on a web page that a phone on the same Wi-Fi opens ----
# The run that was given --phone serves the page. Each movie's run (the same process for a single
# movie, a new one per movie with --all/--queue) copies what it prints, and its progress line,
# into a small status file the page reads; the --all/--queue run itself adds its own lines
# (Movie 2 of 5, DONE, FAILED) from memory.
PHONE_PORT = 8642
PHONE = None                # what the page shows, once phone_setup turned it on
PHONE_LOCK = threading.Lock()
PHONE_FILE = None           # the status file this process writes (a movie's run)
PHONE_PATH = None           # the status file the page reads (the process serving the page)
PHONE_QUEUE = [False]       # the page is served by an --all/--queue run
PHONE_DIRTY = [False]
PHONE_CACHE = [None]
PHONE_ETA = re.compile(r"((?:\d+ h )?\d+ min left, done ~[^,()]*?\d{1,2}:\d\d(?: ?[AP]M)?)")


def phone_line(msg):
    if PHONE is None:
        return
    with PHONE_LOCK:
        PHONE["line"] = msg.strip()
        PHONE["t"] = time.time()
        PHONE_DIRTY[0] = True


def phone_log(text):
    text = text.strip()
    if PHONE is None or not text:
        return
    with PHONE_LOCK:
        now = time.time()
        PHONE["log"].append([now, text])
        PHONE["t"] = now
        m = re.match(r"\[(\d+)/(\d+)\]", text)
        if m:
            PHONE["chunk"] = [int(m[1]), int(m[2])]
            PHONE["eta"] = "finishing the file" if "all chunks done" in text else ""
        eta = PHONE_ETA.search(text)
        if eta:
            PHONE["eta_all" if text.startswith("all ") and " movies:" in text else "eta"] = eta[1]
        m = re.match(r"=== Movie (\d+) of (\d+): (.*) ===$", text)
        if m:
            PHONE["n"], PHONE["of"], PHONE["movie"] = int(m[1]), int(m[2]), m[3]
        PHONE_DIRTY[0] = True


class PhoneTee:
    """sys.stdout/sys.stderr that also hands each finished line to the page (the progress
    line's own writes, between carriage returns, are left out: phone_line has those)."""
    def __init__(self, stream):
        self._stream, self._buf = stream, ""

    def write(self, text):
        n = self._stream.write(text)
        try:
            buf = self._buf + text
            *lines, buf = buf.split("\n")
            for line in lines:
                phone_log(line.rstrip("\r").split("\r")[-1])
            self._buf = buf[buf.rfind("\r") + 1:][-2000:]
        except Exception:           # (the page is a nicety: never let it stop a run)
            pass
        return n

    def __getattr__(self, name):
        return getattr(self._stream, name)


def phone_movie(name):
    if PHONE is not None:
        with PHONE_LOCK:
            PHONE["movie"] = str(name)
            PHONE_DIRTY[0] = True


PHONE_WRITE_LOCK = threading.Lock()


def phone_write():
    if not PHONE_FILE or not PHONE_DIRTY[0]:
        return
    with PHONE_WRITE_LOCK:      # (the writer thread and the exit hook never write together)
        with PHONE_LOCK:
            data = dict(PHONE, log=list(PHONE["log"]))
            PHONE_DIRTY[0] = False
        tmp = PHONE_FILE + f".{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, PHONE_FILE)
        except OSError:             # (Windows: the page was reading it right then) next time
            PHONE_DIRTY[0] = True


def phone_writer():
    while True:
        time.sleep(1)
        phone_write()


def phone_setup(queue_parent):
    """Turn the page's copy of the output on, in a run started by --phone (or by a queue that
    was): a movie's run writes the status file, an --all/--queue run keeps its lines in memory."""
    global PHONE, PHONE_FILE
    path = os.environ.get("DVD_UPSCALE_PHONE")
    if not path:
        return
    PHONE = dict(movie="", n=0, of=0, line="", chunk=None, eta="", eta_all="",
                 log=deque(maxlen=80), t=time.time(), started=time.time())
    try:
        q = json.loads(os.environ.get("DVD_UPSCALE_QUEUE") or "null")
        if q:
            PHONE["n"], PHONE["of"] = int(q["n"]), int(q["of"])
    except (ValueError, TypeError, KeyError):
        pass
    sys.stdout, sys.stderr = PhoneTee(sys.stdout), PhoneTee(sys.stderr)
    if not queue_parent:
        PHONE_FILE = path
        threading.Thread(target=phone_writer, daemon=True).start()
        atexit.register(phone_write)        # (the last lines, "Failed: ..." among them)


PHONE_GPU = dict(t=0.0, gpus=[])
PHONE_GPU_LOCK = threading.Lock()
PHONE_CPU = dict(t=0.0, last=None, busy=None, name=None, threads=0)


def phone_load(chunk, run):
    """The CPU and GPU load (average and peak) of the last chunk and of the whole run so far,
    for the page (LoadWatch.stats(); chunk None at the end of the movie)."""
    if PHONE is not None:
        with PHONE_LOCK:
            PHONE["load"] = dict(chunk=chunk, run=run)
            PHONE_DIRTY[0] = True


def phone_cpu():
    """The processor right now (its load between two of the page's polls, at most every
    2.5 s): name, threads, busy % (None until there are two readings)."""
    with PHONE_GPU_LOCK:
        if PHONE_CPU["name"] is None:
            PHONE_CPU["name"], PHONE_CPU["threads"] = cpu_name()
        if time.time() - PHONE_CPU["t"] >= 2.5:
            now, last = cpu_times(), PHONE_CPU["last"]
            busy = (100.0 * (now[0] - last[0]) / (now[1] - last[1])
                    if now and last and now[1] > last[1] else None)
            PHONE_CPU.update(t=time.time(), last=now, busy=busy)
        return dict(name=PHONE_CPU["name"], threads=PHONE_CPU["threads"], busy=PHONE_CPU["busy"])
PHONE_GPU_FIELDS = ("index,name,temperature.gpu,power.draw,power.limit,utilization.gpu,"
                    "memory.used,memory.total,clocks.sm,clocks.max.sm,fan.speed,"
                    "clocks_throttle_reasons.active")


def phone_gpus():
    """The NVIDIA GPUs right now (nvidia-smi; at most every 2.5 s, only while the page is open),
    [] without nvidia-smi. A value the GPU doesn't report (a laptop's fan) is None."""
    with PHONE_GPU_LOCK:
        if time.time() - PHONE_GPU["t"] < 2.5 or not shutil.which("nvidia-smi"):
            return PHONE_GPU["gpus"]
        PHONE_GPU["t"] = time.time()
        PHONE_GPU["gpus"] = gpus = query_gpus()
        return gpus


def query_gpus():
    """The NVIDIA GPUs right now, one nvidia-smi call ([] without it): a dict each, a value the
    GPU doesn't report (a laptop's fan) is None."""
    if not shutil.which("nvidia-smi"):
        return []
    gpus = []
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=" + PHONE_GPU_FIELDS,
                            "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=8, stdin=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for row in r.stdout.splitlines():
            v = [x.strip() for x in row.split(",")]
            if len(v) != 12:
                continue
            num = lambda x: (float(x) if re.fullmatch(r"[\d.]+", x) else None)
            try:
                reasons = int(v[11], 16)
            except ValueError:
                reasons = 0
            why = [name for bit, name in THROTTLE_BITS if reasons & bit]
            gpus.append(dict(index=v[0], name=v[1], temp=num(v[2]), power=num(v[3]),
                             power_max=num(v[4]), busy=num(v[5]), mem=num(v[6]),
                             mem_max=num(v[7]), clock=num(v[8]), clock_max=num(v[9]),
                             fan=num(v[10]), slowed=why))
    except (OSError, subprocess.SubprocessError):
        pass
    return gpus


def cpu_times():
    """The processor's (busy, total) time so far, all its threads together, None if this system
    doesn't tell: two calls give the load in between."""
    try:
        if os.name == "nt":
            import ctypes
            idle, kernel, user = (ctypes.c_ulonglong() for _ in range(3))
            if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel),
                                                         ctypes.byref(user)):
                return None
            total = kernel.value + user.value               # (kernel time includes idle time)
            return total - idle.value, total
        with open("/proc/stat", encoding="ascii") as f:
            v = [int(x) for x in f.readline().split()[1:9]]
        return sum(v) - v[3] - v[4], sum(v)                 # (idle and iowait aren't busy)
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def cpu_name():
    """The processor's model name ("" if unknown) and its thread count."""
    name = ""
    try:
        if os.name == "nt":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                name = winreg.QueryValueEx(k, "ProcessorNameString")[0]
        elif sys.platform == "darwin":
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5,
                                  stdin=subprocess.DEVNULL).stdout
        else:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                name = next((ln.split(":", 1)[1] for ln in f
                             if ln.lower().startswith("model name")), "")
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return " ".join(str(name).split()), os.cpu_count() or 0


def hardware_text():
    """One line per processor and NVIDIA GPU: what they are, and their limits."""
    name, threads = cpu_name()
    lines = [f"CPU: {name or 'unknown model'}" + (f", {threads} threads" if threads else "")]
    for g in query_gpus():
        extra = [f"{g['mem_max'] / 1024:.1f} GB" if g["mem_max"] else "",
                 f"up to {g['clock_max']:.0f} MHz" if g["clock_max"] else "",
                 f"{g['power_max']:.0f} W limit" if g["power_max"] else ""]
        lines.append(f"GPU {g['index']}: {g['name']}"
                     + "".join(f", {x}" for x in extra if x))
    return lines


class LoadWatch:
    """The processor's and the NVIDIA GPUs' load while a movie is made, sampled every few
    seconds in the background: lines() gives the average and the peak of each, over the samples
    since the last lines(since_last=True) or over the whole run."""

    def __init__(self, every=3.0, start=True):
        self.samples, self.mark, self.every = [], 0, every  # [(cpu % or None, [gpu dict])]
        self.stop_event = threading.Event()
        self.thread = None
        if start:
            self.thread = threading.Thread(target=self.loop, daemon=True)
            self.thread.start()

    def loop(self):
        last = cpu_times()
        while not self.stop_event.wait(self.every):
            now = cpu_times()
            cpu = None
            if last and now and now[1] > last[1]:
                cpu = 100.0 * (now[0] - last[0]) / (now[1] - last[1])
            last = now
            self.add(cpu, query_gpus())

    def add(self, cpu, gpus):
        self.samples.append((cpu, gpus))

    def stop(self):
        self.stop_event.set()

    def stats(self, since_last=False):
        """The [average, peak] of each value, over the samples since the last
        stats(since_last=True) or over the whole run: {"cpu": [..] or None, "gpus": [{...}]}."""
        rows = self.samples[self.mark:] if since_last else self.samples
        if since_last:
            self.mark = len(self.samples)
        ap = lambda xs: [sum(xs) / len(xs), max(xs)] if xs else None
        by_gpu = {}
        for _, gpus in rows:
            for g in gpus:
                by_gpu.setdefault(g["index"], []).append(g)
        out = dict(cpu=ap([c for c, _ in rows if c is not None]), gpus=[])
        for idx, gs in by_gpu.items():
            vals = lambda k: [g[k] for g in gs if g.get(k) is not None]
            out["gpus"].append(dict(index=idx, name=gs[-1].get("name", ""),
                                    mem_max=max(vals("mem_max"), default=None),
                                    **{k: ap(vals(k)) for k in
                                       ("busy", "clock", "temp", "power", "mem")}))
        return out

    def lines(self, since_last=False):
        return self.format(self.stats(since_last))

    @staticmethod
    def format(st):
        """stats() as text lines: CPU, then each GPU."""
        out = []
        avg_peak = lambda x, fmt, unit: (f"{fmt.format(x[0])}{unit} avg, "
                                         f"{fmt.format(x[1])}{unit} peak")
        if st["cpu"]:
            out.append(f"CPU: {avg_peak(st['cpu'], '{:.0f}', '%')}")
        for g in st["gpus"]:
            parts = [avg_peak(g[key], "{:.0f}", unit)
                     for key, unit in (("busy", "%"), ("clock", " MHz"), ("temp", " C"),
                                       ("power", " W")) if g[key]]
            if g["mem"]:
                parts.append(f"VRAM {g['mem'][0] / 1024:.1f} GB avg, "
                             f"{g['mem'][1] / 1024:.1f} GB peak"
                             + (f" of {g['mem_max'] / 1024:.1f}" if g["mem_max"] else ""))
            if parts:
                label = "GPU" if len(st["gpus"]) == 1 else f"GPU {g['index']}"
                out.append(f"{label}: " + " | ".join(parts))
        return out


def phone_snapshot():
    try:
        with open(PHONE_PATH, encoding="utf-8") as f:
            PHONE_CACHE[0] = json.load(f)
    except (OSError, ValueError, TypeError):
        pass                            # (not written yet, or being replaced: the last one)
    out = dict(now=time.time(), movie=PHONE_CACHE[0], queue=None, gpus=phone_gpus(),
               cpu=phone_cpu())
    if PHONE_QUEUE[0] and PHONE is not None:
        with PHONE_LOCK:
            out["queue"] = dict(PHONE, log=list(PHONE["log"]))
    return out


def phone_host_ok(host):
    """The Host a request was made to: an address (the phone typed one), localhost, this PC's
    name, or a Tailscale name. A web page on the internet that points its own name at this PC
    (DNS rebinding) arrives with ITS name and is refused."""
    host = str(host or "").strip().lower()
    if host.startswith("["):                     # an IPv6 address in brackets
        return True
    host = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host) or host in ("localhost", ""):
        return True
    mine = {socket.gethostname().lower(), socket.gethostname().lower().split(".")[0]}
    return host in mine or host.endswith(".ts.net") or host.endswith(".local")


class PhoneHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if not phone_host_ok(self.headers.get("Host")):
            self.send_error(403)
            return
        if self.path.startswith("/status"):
            body, kind = json.dumps(phone_snapshot()).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, kind = PHONE_PAGE.encode("utf-8"), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):       # (each request would print a line in the window)
        pass

    def handle(self):
        try:
            super().handle()
        except OSError:                 # (a phone that locked its screen or left the Wi-Fi)
            pass


def is_tailscale_ip(ip):
    """100.64.0.0/10: the addresses Tailscale gives its devices."""
    try:
        a, b = (int(x) for x in str(ip).split(".")[:2])
    except ValueError:
        return False
    return a == 100 and 64 <= b <= 127


class PhoneServer(http.server.ThreadingHTTPServer):
    """The page's server. Windows lets a second program take a port that is in use when it asks
    to reuse addresses (Python's servers do), so two runs at once would share 8642 and the
    phone would see either: there the port is taken exclusively instead, and the second run
    says the port is in use."""
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        # (HTTPServer.server_bind asks the network for this PC's full name, which can take
        # seconds on a PC with no DNS: the page doesn't use it)
        import socketserver
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "dvd_upscale", self.server_address[1]

    def handle_error(self, request, client_address):
        pass                            # (a dropped connection is not an error worth a traceback)


def lan_ip():
    """This PC's address on the home network (no packet is sent: it only picks the route; a
    Tailscale route to that range is passed over)."""
    for probe in ("10.255.255.255", "192.168.255.255", "172.31.255.255"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((probe, 1))
            ip = s.getsockname()[0]
            if not is_tailscale_ip(ip) and not ip.startswith("127."):
                return ip
        except OSError:
            pass
        finally:
            s.close()
    return ""


def tailscale_addresses():
    """This PC's Tailscale name and address, if Tailscale is running here: the phone can open
    the page through it from anywhere (mobile data, another Wi-Fi), with Tailscale on the phone
    too. From the Tailscale program (its MagicDNS name, e.g. laptop.tail1234.ts.net, and its
    100.x.y.z address); without it, from this PC's own addresses. [] if none."""
    found = []
    exe = shutil.which("tailscale")
    for path in ([os.path.join(os.environ[v], "Tailscale", "tailscale.exe")
                  for v in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")
                  if os.environ.get(v)]       # (never a path relative to the current folder)
                 + ["/Applications/Tailscale.app/Contents/MacOS/Tailscale"]):
        if not exe and os.path.isabs(path) and os.path.isfile(path):
            exe = path
    if exe:
        try:
            r = subprocess.run([exe, "status", "--json"], capture_output=True, timeout=8,
                               stdin=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            j = json.loads(r.stdout.decode("utf-8", "replace") or "{}")
            me = j.get("Self") or {}
            if j.get("BackendState", "Running") == "Running":
                name = str(me.get("DNSName") or "").rstrip(".")
                found += [name] if name else []
                found += [ip for ip in me.get("TailscaleIPs") or [] if is_tailscale_ip(ip)][:1]
        except (OSError, ValueError, subprocess.SubprocessError, AttributeError):
            pass
    if not found:
        try:
            found = sorted({i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None,
                                                                 socket.AF_INET)
                            if is_tailscale_ip(i[4][0])})[:1]
        except OSError:
            pass
    return found


PHONE_ASKED = [False]       # --phone was typed: the firewall is opened for this run (Windows)
PHONE_RULE = "dvd_upscale phone page"


def phone_firewall_command(port, pid):
    """PowerShell that opens the phone page's port in Windows Firewall, waits until the run with
    that process id ends, then closes it again: one admin prompt covers the whole run, and
    nothing stays open. Two rules, named for the port (a second run on another port never
    deletes this one's): the local subnet on PRIVATE and DOMAIN networks only (not a café's
    or hotel's Wi-Fi), and Tailscale's own addresses on any."""
    n = f"{PHONE_RULE} {int(port)}"
    return (f"$n = '{n}'; $t = '{n} (Tailscale)'; "
            "Remove-NetFirewallRule -DisplayName $n,$t -ErrorAction SilentlyContinue; "
            f"New-NetFirewallRule -DisplayName $n -Direction Inbound -Action Allow -Protocol TCP "
            f"-LocalPort {int(port)} -Profile Private,Domain -RemoteAddress LocalSubnet "
            "| Out-Null; "
            f"New-NetFirewallRule -DisplayName $t -Direction Inbound -Action Allow -Protocol TCP "
            f"-LocalPort {int(port)} -Profile Any -RemoteAddress 100.64.0.0/10 | Out-Null; "
            f"Wait-Process -Id {int(pid)} -ErrorAction SilentlyContinue; "
            "Remove-NetFirewallRule -DisplayName $n,$t -ErrorAction SilentlyContinue")


def phone_firewall(port):
    """--phone on Windows: ask (the Windows admin prompt, UAC) to open the page's port for this
    run, so the phone reaches it through Tailscale and any home network. Returns a line to show."""
    if os.name != "nt":
        return ""
    import base64
    import ctypes
    script = phone_firewall_command(port, os.getpid())
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    try:
        # ("runas": the admin prompt; the helper runs hidden and on its own, so the run doesn't wait)
        rc = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", "powershell.exe",
            f"-NoProfile -WindowStyle Hidden -EncodedCommand {encoded}", None, 0)
    except (OSError, AttributeError):
        rc = 0
    if rc > 32:
        return (f"       firewall: port {port} opened for this run (home network and Tailscale "
                "only); it closes by itself when the run ends")
    return ("       firewall: not opened (the admin prompt was declined): the page works on the "
            "home Wi-Fi if Python is allowed there")


def phone_args(argv):
    """Take --phone [port] and --no-phone out of argv (so every parser and every movie's run
    never sees them). The page is on by itself: the port to serve it on, or None for --no-phone,
    an --analyze run, or a movie of --all/--queue (the queue's own run serves the page)."""
    off = "--no-phone" in argv
    while "--no-phone" in argv:
        argv.remove("--no-phone")
    port = PHONE_PORT
    for i, w in enumerate(argv):
        if w == "--phone" or w.startswith("--phone="):
            val = w.split("=", 1)[1] if "=" in w else None
            if val is None and i + 1 < len(argv) and argv[i + 1].isdigit():
                val = argv[i + 1]
                del argv[i + 1]
            del argv[i]
            if val and not val.isdigit():
                sys.exit(f"--phone: '{val}' isn't a port number (e.g. --phone 8642)")
            port = int(val) if val else PHONE_PORT
            if not 1 <= port <= 65535:
                sys.exit(f"--phone: {port} isn't a port number (1-65535, e.g. --phone 8642)")
            PHONE_ASKED[0] = True
            break
    if off:
        os.environ["DVD_UPSCALE_NO_PHONE"] = "1"       # (the movies of --all/--queue inherit it)
    # a run that only prints help or a list, a set-up command, or a movie's run under a queue
    # that serves the page, doesn't start the page
    quiet = ("-h", "--help", "--commands", "commands", "--gpu-detect", "--clip") + tuple(
        w for w in argv if w.startswith("--ncnn-"))
    if off or os.environ.get("DVD_UPSCALE_NO_PHONE") or "--analyze" in argv \
            or os.environ.get("DVD_UPSCALE_PHONE") or any(w in quiet for w in argv):
        return None
    return port


def phone_start(port, queue_parent):
    """The page is a nicety: whatever goes wrong here, the run goes on without it."""
    try:
        _phone_start(port, queue_parent)
    except Exception as e:           # (a full temp folder, a firewall tool that fails...)
        print(f"Phone page: couldn't start it ({e}); the run goes on without it.", flush=True)


def _phone_start(port, queue_parent):
    global PHONE_PATH
    folder = Path(tempfile.mkdtemp(prefix="dvd_upscale_phone_"))
    atexit.register(shutil.rmtree, folder, True)
    server, err = None, None
    for attempt in range(6):
        # (on Windows the port is taken exclusively, and the last run's closed connections hold
        # it for a little while: a run started right after another waits a few seconds)
        try:
            server = PhoneServer(("0.0.0.0", port), PhoneHandler)
            break
        except OSError as e:
            err = e
            time.sleep(1)
    if server is None:
        print(f"Phone page: port {port} can't be used ({err.strerror or err}; another run?): "
              f"this run goes on without it; --phone {port + 1} would serve it on another port.",
              flush=True)
        return
    server.daemon_threads = True
    PHONE_PATH = str(folder / "status.json")
    PHONE_QUEUE[0] = queue_parent
    os.environ["DVD_UPSCALE_PHONE"] = PHONE_PATH        # (each movie's run writes it)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ip = lan_ip()
    print((f"Phone: on the same Wi-Fi, open  http://{ip}:{port}  in the phone's browser\n"
           if ip else "Phone: no home-network address found for this PC (Wi-Fi off?)\n")
          + "       (if Windows asks, allow Python on private networks)", flush=True)
    if PHONE_ASKED[0]:
        note = phone_firewall(port)
        if note:
            print(note, flush=True)
    ts = tailscale_addresses()
    if ts:
        print("       anywhere, through Tailscale (on the phone too):  "
              + "  or  ".join(f"http://{x}:{port}" for x in ts), flush=True)


PHONE_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#111418">
<title>Upscale progress</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--fg:#15181d;--dim:#667080;--line:#e2e5ea;--acc:#2f6fed;
--ok:#1f9d55;--warn:#c27c0e;--bad:#d64545}
@media (prefers-color-scheme:dark){:root{--bg:#111418;--card:#1a1e24;--fg:#eef0f3;--dim:#8d96a3;
--line:#2a3039;--acc:#5b8cff;--ok:#3ccf7c;--warn:#f0b13c;--bad:#ff6b6b}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.4 -apple-system,system-ui,
"Segoe UI",Roboto,sans-serif;padding:max(16px,env(safe-area-inset-top)) 16px 32px}
.wrap{max-width:560px;margin:0 auto}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
h1{font-size:15px;font-weight:600;color:var(--dim);margin:0;letter-spacing:.02em}
.pill{font-size:13px;padding:3px 10px;border-radius:99px;border:1px solid var(--line);color:var(--dim)}
.pill::before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;
margin-right:6px;background:currentColor;vertical-align:1px}
.live{color:var(--ok)}.stale{color:var(--warn)}.off{color:var(--bad)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:12px}
.movie{font-size:19px;font-weight:650;word-break:break-word}
.sub{color:var(--dim);font-size:14px;margin-top:2px}
.pct{font-size:44px;font-weight:700;font-variant-numeric:tabular-nums;margin:12px 0 6px}
.bar{height:10px;background:var(--line);border-radius:99px;overflow:hidden}
.bar>i{display:block;height:100%;width:0;background:var(--acc);border-radius:99px;transition:width .6s}
[hidden]{display:none!important}
.row{display:flex;justify-content:space-between;gap:12px;margin-top:10px;font-size:14px}
.row b{font-weight:600;text-align:right}
.now{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;color:var(--dim);
margin-top:12px;word-break:break-word;min-height:1.4em}
h2{font-size:13px;color:var(--dim);font-weight:600;margin:0 0 8px;text-transform:uppercase;letter-spacing:.05em}
ol{list-style:none;margin:0;padding:0;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px}
li{padding:5px 0;border-top:1px solid var(--line);display:flex;gap:10px;word-break:break-word}
li:first-child{border-top:0}
li time{color:var(--dim);flex:none}
.err{color:var(--bad)}.done{color:var(--ok)}
.gpu+.gpu{margin-top:14px;padding-top:14px;border-top:1px solid var(--line)}
.gname{font-weight:600;font-size:15px;margin-bottom:10px}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.tile{background:var(--bg);border-radius:10px;padding:10px 12px}
.tile:last-child:nth-child(odd){grid-column:1/-1}
.tile small{display:block;color:var(--dim);font-size:12px}
.tile b{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}
.tile span{color:var(--dim);font-size:13px;margin-left:3px}
.mini{height:5px;background:var(--line);border-radius:9px;margin-top:7px;overflow:hidden}
.mini>i{display:block;height:100%;background:var(--acc)}
.hot b{color:var(--warn)}.vhot b{color:var(--bad)}
.slow{margin-top:10px;font-size:13px;color:var(--warn)}
.ap{width:100%;border-collapse:collapse;margin-top:12px;font-size:13px;font-variant-numeric:tabular-nums}
.ap th{color:var(--dim);font-weight:600;font-size:12px;text-align:right;padding:0 0 4px}
.ap th:first-child,.ap td:first-child{text-align:left;color:var(--dim)}
.ap td{text-align:right;padding:5px 0;border-top:1px solid var(--line);white-space:nowrap}
.ap td+td,.ap th+th{padding-left:10px}
.banner{display:none;background:var(--bad);color:#fff;border-radius:12px;padding:12px 14px;
margin-bottom:12px;font-size:14px}
footer{color:var(--dim);font-size:12px;text-align:center;margin-top:8px}
</style></head><body><div class="wrap">
<header><h1>DVD UPSCALE</h1><span id="pill" class="pill">connecting</span></header>
<div id="banner" class="banner">Can't reach the PC. The run may have finished or been stopped,
or the PC is asleep or off this Wi-Fi. Showing the last thing it reported.</div>
<div class="card">
 <div class="movie" id="movie">Waiting for the first update...</div>
 <div class="sub" id="which"></div>
 <div class="pct" id="pct">--</div>
 <div class="bar"><i id="fill"></i></div>
 <div class="row"><span>Chunks</span><b id="chunks">-</b></div>
 <div class="row"><span>This movie</span><b id="eta">-</b></div>
 <div class="row" id="allrow" hidden><span>All movies</span><b id="etaall">-</b></div>
 <div class="now" id="now"></div>
</div>
<div class="card" id="ccard" hidden><h2>Processor</h2><div id="cpu"></div></div>
<div class="card" id="gcard" hidden><h2>Graphics card</h2><div id="gpus"></div></div>
<div class="card" id="qcard" hidden><h2>Queue</h2><ol id="qlog"></ol></div>
<div class="card"><h2>Recent output</h2><ol id="log"></ol></div>
<footer id="foot"></footer>
</div>
<script>
const $=id=>document.getElementById(id);
let last=null,lastOk=0,skew=0;
function hm(t){const d=new Date((t-skew)*1000);return d.toLocaleTimeString([],{hour:'numeric',minute:'2-digit'})}
function fill(ol,items,n){ol.textContent='';items.slice(-n).reverse().forEach(([t,s])=>{
 const li=document.createElement('li'),tm=document.createElement('time'),sp=document.createElement('span');
 tm.textContent=hm(t);sp.textContent=s;
 if(/fail|error|warning|⚠/i.test(s))sp.className='err';else if(/^(✅ )?DONE|^Finished|Saved/.test(s))sp.className='done';
 li.append(tm,sp);ol.append(li)})}
function show(d){
 const m=d.movie,q=d.queue;
 if(q&&q.log.length){$('qcard').hidden=false;fill($('qlog'),q.log,8)}
 if(!m){if(q){$('movie').textContent=q.movie||'Getting ready...';
   $('which').textContent=q.of?`Movie ${q.n} of ${q.of}`:''}return}
 $('movie').textContent=m.movie||'Getting ready...';
 $('which').textContent=m.of?`Movie ${m.n} of ${m.of}`:'';
 if(m.chunk){const p=m.chunk[0]/m.chunk[1];$('pct').textContent=Math.floor(p*100)+'%';
  $('fill').style.width=(p*100)+'%';$('chunks').textContent=`${m.chunk[0]} of ${m.chunk[1]}`}
 else{$('pct').textContent='Preparing';$('fill').style.width='0';$('chunks').textContent='-'}
 $('eta').textContent=m.eta||'-';
 $('allrow').hidden=!m.eta_all;$('etaall').textContent=m.eta_all||'';
 $('now').textContent=m.line||'';
 fill($('log'),m.log,25);
}
function tile(label,val,unit,frac,cls){
 const t=document.createElement('div');t.className='tile '+(cls||'');
 const sm=document.createElement('small');sm.textContent=label;
 const b=document.createElement('b');b.textContent=val==null?'n/a':val;
 t.append(sm,b);if(unit&&val!=null){const u=document.createElement('span');u.textContent=unit;t.append(u)}
 if(frac!=null){const m=document.createElement('div');m.className='mini';const i=document.createElement('i');
  i.style.width=Math.min(100,Math.max(0,frac*100))+'%';m.append(i);t.append(m)}
 return t}
const r0=x=>x==null?null:Math.round(x);
function apTable(ld,rows){
 const cols=ld?[['Last chunk',ld.chunk],['Whole run',ld.run]].filter(c=>c[1]):[];if(!cols.length)return null;
 const t=document.createElement('table');t.className='ap';const h=t.insertRow();
 ['avg / peak',...cols.map(c=>c[0])].forEach(s=>{const c=document.createElement('th');c.textContent=s;h.append(c)});
 const ap=(x,f)=>x?`${f(x[0])} / ${f(x[1])}`:'-';
 rows.forEach(([label,get,f])=>{const v=cols.map(c=>get(c[1]));if(!v.some(x=>x))return;
  const r=t.insertRow();[label,...v.map(x=>ap(x,f))].forEach(s=>{r.insertCell().textContent=s})});
 return t.rows.length>1?t:null}
function showCpu(c,ld){
 const box=$('cpu'),has=ld&&((ld.chunk&&ld.chunk.cpu)||(ld.run&&ld.run.cpu));
 $('ccard').hidden=!c&&!has;if(!c&&!has)return;box.textContent='';
 if(c){const n=document.createElement('div');n.className='gname';
  n.textContent=(c.name||'Processor')+(c.threads?` · ${c.threads} threads`:'');box.append(n);
  const ts=document.createElement('div');ts.className='tiles';
  ts.append(tile('Busy now',r0(c.busy),'%',c.busy==null?null:c.busy/100));box.append(ts)}
 const t=apTable(ld,[['Busy %',s=>s.cpu,r0]]);if(t)box.append(t)}
function showGpus(gs,ld){
 const box=$('gpus');$('gcard').hidden=!gs||!gs.length;if(!gs||!gs.length)return;box.textContent='';
 gs.forEach(g=>{const d=document.createElement('div');d.className='gpu';
  const n=document.createElement('div');n.className='gname';n.textContent=g.name;d.append(n);
  const ts=document.createElement('div');ts.className='tiles';
  ts.append(tile('Busy',r0(g.busy),'%',g.busy==null?null:g.busy/100));
  ts.append(tile('Temperature',r0(g.temp),'°C',null,g.temp>=87?'vhot':g.temp>=78?'hot':''));
  ts.append(tile('Power',r0(g.power),g.power_max?`W of ${r0(g.power_max)}`:'W',
   g.power!=null&&g.power_max?g.power/g.power_max:null));
  ts.append(tile('Memory',g.mem==null?null:(g.mem/1024).toFixed(1),
   g.mem_max?`of ${(g.mem_max/1024).toFixed(0)} GB`:'GB',g.mem!=null&&g.mem_max?g.mem/g.mem_max:null));
  ts.append(tile('Clock',r0(g.clock),g.clock_max?`MHz (max ${r0(g.clock_max)})`:'MHz',
   g.clock!=null&&g.clock_max?g.clock/g.clock_max:null));
  if(g.fan!=null)ts.append(tile('Fan',r0(g.fan),'%',g.fan/100));
  d.append(ts);
  if(g.slowed&&g.slowed.length){const w=document.createElement('div');w.className='slow';
   w.textContent='⚠ Slowing itself down: '+g.slowed.join(', ');d.append(w)}
  const of=s=>s.gpus.find(x=>x.index===g.index)||{};
  const t=apTable(ld,[['Busy %',s=>of(s).busy,r0],['Clock MHz',s=>of(s).clock,r0],
   ['Temperature °C',s=>of(s).temp,r0],['Power W',s=>of(s).power,r0],
   ['Memory GB',s=>of(s).mem,x=>(x/1024).toFixed(1)]]);
  if(t)d.append(t);
  box.append(d)})}
function status(){
 const pill=$('pill'),now=Date.now()/1000,off=now-lastOk>12;
 $('banner').style.display=off&&lastOk?'block':'none';
 if(!last){return}
 const t=Math.max(last.movie?last.movie.t:0,last.queue?last.queue.t:0),age=last.now-t;
 if(off){pill.className='pill off';pill.textContent='offline'}
 else if(age>300){pill.className='pill stale';pill.textContent='quiet '+Math.round(age/60)+' min'}
 else{pill.className='pill live';pill.textContent='live'}
 $('foot').textContent='PC last reported '+(age<60?Math.round(age)+' s':Math.round(age/60)+' min')+' ago · refreshes every 3 s';
}
async function poll(){
 try{const r=await fetch('/status.json',{cache:'no-store'});const d=await r.json();
  last=d;lastOk=Date.now()/1000;skew=d.now-lastOk;show(d);const ld=d.movie&&d.movie.load;showCpu(d.cpu,ld);showGpus(d.gpus,ld)}catch(e){}
 status()}
poll();setInterval(poll,3000);
</script></body></html>"""


def short_time(secs):
    secs = max(1, round(secs))
    if secs < 60:
        return f"{secs} s"
    m = round(secs / 60)
    return f"{m // 60} h {m % 60} min" if m >= 60 else f"{m} min"


def ffmpeg_progress(cmd, secs, label, cwd=None):
    """Run an ffmpeg command showing label, % done and time left on the progress line; secs is
    the length of what it writes. Returns (exit code, its error output, seconds it took)."""
    cmd = nostdin(cmd)
    cmd = cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:]
    t0 = time.time()
    with tempfile.TemporaryFile() as err:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err,
                             encoding="utf-8", errors="replace", cwd=cwd)
        try:
            for line in p.stdout:
                key, _, val = line.strip().partition("=")
                # (out_time_ms is in microseconds too, in every ffmpeg version)
                if key in ("out_time_us", "out_time_ms") and val.isdigit() and secs > 0:
                    done, spent = min(1.0, int(val) / 1e6 / secs), time.time() - t0
                    left = (f", about {short_time(spent / done - spent)} left"
                            if done > 0.03 and spent > 5 else "")
                    status_line(f"  {label}: {done:.0%}{left}")
            p.wait()
        finally:
            if p.poll() is None:            # Ctrl+C: don't leave it running
                p.kill()
                p.wait()
        err.seek(0)
        text = err.read().decode("utf-8", "replace")
    status_line()
    return p.returncode, text, time.time() - t0


def run_step(cmd, secs, label, cwd=None):
    """run() for a long step: progress while it runs, how long it took when done."""
    rc, text, took = ffmpeg_progress(cmd, secs, label, cwd)
    if text.strip():
        print(text.rstrip(), file=sys.stderr, flush=True)
    if rc:
        raise subprocess.CalledProcessError(rc, cmd[0])
    say(f"  {label}: done ({short_time(took)})")


def frac(x, default=Fraction(0)):
    try:
        return Fraction(str(x).replace(":", "/"))
    except (ValueError, ZeroDivisionError):
        return default


def hms(x):
    """'01:23:45.678000000' (Matroska DURATION tag) or plain seconds -> float, else None."""
    try:
        secs = 0.0
        for part in str(x).split(":"):
            secs = secs * 60 + float(part)
        return secs if secs > 0 else None
    except ValueError:
        return None


def probe(src):
    j = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,sample_aspect_ratio,avg_frame_rate,r_frame_rate,start_time,duration"
         ":stream_tags:format=start_time,duration", "-of", "json", str(src)],
        stdin=subprocess.DEVNULL, timeout=LONG_PROBE_TIMEOUT))
    if not j.get("streams"):
        raise ValueError("no video stream found")
    s, f = j["streams"][0], j.get("format", {})
    cstart = float(frac(f.get("start_time"), Fraction(0)))
    vs = s.get("start_time")
    vstart = float(frac(vs)) if vs not in (None, "N/A") else cstart
    seek_delay, ext = 0.0, Path(src).suffix.lower()
    if ext in MPEG_EXT or ext == ".avi":
        # a recording or .vob cut mid-GOP: its first packets can't be decoded, and the picture
        # (which the sound is lined up with) starts at the first frame that can. An AVI stores
        # no picture times: with B-frames ffmpeg times its pictures 1-2 frames late (the decoder
        # delay), and -ss seeks by those times, so chunk seeks add that delay (see main)
        r = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-read_intervals",
                            "%+#200", "-show_entries", "frame=best_effort_timestamp_time", "-of",
                            "csv=p=0", str(src)], capture_output=True, encoding="utf-8",
                           errors="replace", stdin=subprocess.DEVNULL,
                           timeout=LONG_PROBE_TIMEOUT)
        try:
            first = float(r.stdout.split()[0].strip(","))
            if vstart < first < vstart + 5:
                if ext == ".avi":
                    seek_delay = first - vstart
                else:
                    vstart = first
        except (IndexError, ValueError):
            pass
    duration = hms(f.get("duration"))
    # the video's own length; the container duration also counts audio/subs that run longer
    vdur = hms(s.get("duration"))
    if vdur is None:
        # Matroska: the muxer's own DURATION tag; MakeMKV's DURATION-eng statistics tag survives
        # a stream-copy cut (--clip) and then still holds the whole movie's length
        tags = {k.upper(): v for k, v in s.get("tags", {}).items()}
        vdur = hms(tags.get("DURATION") or next((v for k, v in sorted(tags.items())
                                                 if k.startswith("DURATION")), None))
        if vdur and duration and vdur > duration + 2:
            vdur = None
    if duration is None and vdur is None:
        raise ValueError("could not determine duration")
    fps, rfps = frac(s.get("avg_frame_rate")), frac(s.get("r_frame_rate"))
    if rfps > 0 and (fps <= 0 or fps > 120 or fps > 1.5 * rfps):
        fps = rfps          # raw DV files report an average of 60000 fps
    return dict(
        w=int(s["width"]), h=int(s["height"]),
        sar=frac(s.get("sample_aspect_ratio")) or Fraction(1),
        fps=fps,
        cstart=cstart, vstart=vstart,
        duration=duration or vdur + (vstart - cstart),
        vduration=vdur, seek_delay=seek_delay)


def sample_start(info):
    return min(info["duration"] * 0.2, 600) if info["duration"] > 120 else 0


def detect_mode(a, info):
    """Sample ~60s with ffmpeg's idet filter and classify the source."""
    ss = sample_start(info)
    # (no -ss for a sample from the start: an AVI with B-frames can't seek to 0 and loses frames)
    rc, txt = capture(["ffmpeg", "-hide_banner", *(["-ss", f"{ss:.1f}"] if ss else []), "-t", "60",
                       "-i", a.input, "-an", "-sn", "-vf", "idet", "-f", "null", "-"])
    # ffmpeg can print the idet summary twice (graph re-init); the last one has the data
    ms = re.findall(r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*"
                    r"Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)", txt)
    rs = re.findall(r"Repeated Fields:\s*Neither:\s*(\d+)\s*Top:\s*(\d+)\s*Bottom:\s*(\d+)", txt)
    if not ms or not rs or sum(map(int, ms[-1])) == 0:
        why = f" (ffmpeg exit code {rc})" if rc else ""
        return "progressive", f"could not analyze{why}, assuming progressive"
    tff, bff, prog, und = map(int, ms[-1])
    nei, top, bot = map(int, rs[-1])
    inter = (tff + bff) / max(1, tff + bff + prog + und)
    rep = (top + bot) / max(1, nei + top + bot)
    detail = f"interlaced frames {inter:.0%}, repeated fields {rep:.0%}"
    if rep > 0.08 and inter > 0.15:
        return "telecine", detail
    if 0 < info["fps"] <= Fraction(241, 10):      # 23.976/24 fps streams are never interlaced video
        return "progressive", detail
    if inter > 0.40:
        return "interlaced", detail
    return "progressive", detail


def pulldown_mix(path):
    """How a DVD's film is stored, over the whole file, from the frame timestamps alone (a few
    seconds of reading, no decoding). Film with soft pulldown has frames flagged to show for 3
    fields (50 ms); hard-telecined film and video have a frame every 33 ms. A disc mixing both
    (common for anime, and some animated movies) needs inverse telecine throughout, but the 60 s
    sample idet looks at only sees one kind. Returns (share of the time in 3-field frames,
    seconds in runs of 33 ms frames lasting 1 s or more), or None if it can't be read."""
    rc, txt = capture(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                       "stream=codec_name:packet=pts_time", "-of", "csv=p=0", path])
    if not re.search(r"^mpeg2video", txt, re.M):
        return None                 # soft pulldown is a DVD (MPEG-2) thing
    ts = sorted(float(x) for x in re.findall(r"^\s*(-?\d+(?:\.\d+)?)", txt, re.M))
    if rc or len(ts) < 100 or ts[-1] <= ts[0]:
        return None
    d = [y - x for x, y in zip(ts, ts[1:])]
    soft = sum(x for x in d if 0.045 < x < 0.056)
    video = run = 0.0
    n = 0
    for x in d + [1.0]:
        if 0.030 < x < 0.037:
            run, n = run + x, n + 1
        else:
            if n >= 30:
                video += run
            run, n = 0.0, 0
    return soft / (ts[-1] - ts[0]), video


def cadence(a, info):
    """Decode a 20s sample. Returns (real frame rate in the file or None,
    fraction of frames mpdecimate keeps, how regular the drops are).
    24->30 duplicate-frame pulldown keeps ~80% and drops exactly one frame in every five;
    animation with held drawings can also keep ~80%, but its drops are irregular."""
    secs = min(20.0, info["duration"])
    start = sample_start(info)

    def decode(vf, loglevel="info"):
        rc, txt = capture(["ffmpeg", "-hide_banner", "-loglevel", loglevel,
                           *(["-ss", f"{start:.1f}"] if start else []), "-t", f"{secs:.1f}",
                           "-i", a.input, "-an", "-sn", "-vf", vf,
                           "-fps_mode", "passthrough", "-f", "null", "-"])
        if rc:
            print(f"WARNING: frame-rate analysis failed (ffmpeg exit code {rc}; "
                  "ffmpeg 5.1+ is required)")
        return txt

    m = re.findall(r"frame=\s*(\d+)", decode("null"))
    fin = int(m[-1]) if m else 0
    if fin < 30:
        return None, 1.0, 0.0
    rate = fin / secs
    snaps = [Fraction(24000, 1001), Fraction(25), Fraction(30000, 1001)]
    best = min(snaps, key=lambda x: abs(float(x) - rate))
    snapped = best if abs(float(best) - rate) / float(best) < 0.04 else None
    if info["h"] in (480, 486) and 23.0 <= rate <= 31.2:
        # NTSC has no 25 fps: a sample between 23.976 and 29.97 is film mixed with video, and
        # 25 would add a stutter to all of it
        snapped = Fraction(24000, 1001) if rate < 26.97 else Fraction(30000, 1001)

    drops = [d == "drop" for d in re.findall(r"\b(keep|drop) pts:", decode("mpdecimate", "debug"))]
    if len(drops) < 30:
        return snapped, 1.0, 0.0
    kept = 1 - sum(drops) / len(drops)
    # share of 5-frame windows (best phase) that contain exactly one dropped frame
    windows = len(drops) // 5
    regular = max(sum(sum(drops[p + 5 * i:p + 5 * i + 5]) == 1 for i in range(windows - 1))
                  for p in range(5)) / max(1, windows - 1)
    return snapped, kept, regular


# ---- VHS captures (--type vhs) -----------------------------------------------------------------
# On tape every field is played back with its own noise and jitter, so idet's repeated-field count
# and mpdecimate (detect_mode/cadence above) never see the 3:2 pulldown of a movie tape. Instead:
# per frame, the mean difference of each field against the same field of the previous frame. In
# 3:2 telecine one top and one bottom field in every 5 frames are repeats, so in each 5-frame
# window the smallest difference keeps landing on the same frame, top and bottom repeats 2 or 3
# frames apart; on camcorder video (60 fields/s) it wanders. il splits the fields by line parity,
# so a wrong field-order flag doesn't matter.
VHS_CADENCE_VF = ("crop=iw-32:ih-24:16:8,il=l=d:c=d,scale=88:60:flags=area,format=grayf32,"
                  "tblend=all_mode=difference,scale=1:2:flags=area")
# camcorder tapes: 2.5x the frames of a movie tape (59.94p), so the much cheaper general-x4v3
# model (its denoise-0.5 version), more temporal smoothing and no sharpening: measured best on
# simulated tapes (LPIPS 0.302 vs 0.329 with the live settings, same PSNR)
VHS_VIDEO = ("realesr-general-dn50-x4v3", 4, 480, "4:8:9:14", 0.75, 6, 0.0)
# neither is in the official ncnn download
VHS_MODEL_ZIPS = {"realesr-general-dn50-x4v3": "realesr-general-dn50-x4v3.zip",
                  "realesrgan-x2plus": "realesrgan-x2plus.zip"}
MPEG_EXT = (".mpg", ".mpeg", ".vob", ".ts", ".m2ts", ".mts", ".m2v")
# subtitle formats a .mkv can hold (mov_text is converted to SRT); teletext etc. can't go in
MKV_SUBS = {"dvd_subtitle", "dvb_subtitle", "hdmv_pgs_subtitle", "hdmv_text_subtitle", "subrip",
            "ass", "ssa", "text", "mov_text", "webvtt", "arib_caption"}
# what an .mp4 output keeps: audio .mp4 players take, and text subtitles (DVD ones are pictures)
MP4_AUDIO = {"ac3", "eac3", "aac", "mp3"}
MP4_SUBS = {"subrip", "ass", "ssa", "mov_text", "webvtt", "text"}


def _phase_share(d):
    """Share of 5-frame windows whose smallest value is on the most common phase, and that phase."""
    cnt = [0] * 5
    for i in range(0, len(d) - 4, 5):
        w = d[i:i + 5]
        cnt[w.index(min(w))] += 1
    used = sum(cnt)
    # at least 12 windows (2 s): on 1-2 s camcorder clips 6-11 windows voted film 1 time in 16
    return (max(cnt) / used, cnt.index(max(cnt))) if used >= 12 else (0.0, -1)


def _low_phases(d):
    """Phases that are in the low group (within a quarter of the window's range above its
    smallest value) of at least 80% of the 5-frame windows that show motion (largest > 2x the
    smallest), and how many windows showed motion. In telecine the repeated field is low in every
    such window. Cartoons hold each drawing for 2-3 film frames, so other phases are often just as
    low and the smallest value wanders between them: _phase_share then misses the cadence
    (measured on simulated cartoon tapes)."""
    cnt, moving = [0] * 5, 0
    for i in range(0, len(d) - 4, 5):
        w = d[i:i + 5]
        lo, hi = min(w), max(w)
        if hi > 2 * lo and hi - lo > 0.001:     # (a keyframe on a blue/black screen: ~0.00002)
            moving += 1
            for p in range(5):
                cnt[p] += w[p] <= lo + 0.25 * (hi - lo)
    return [p for p in range(5) if moving and cnt[p] >= 0.8 * moving], moving


def vhs_detect(a, info, start=None):
    """One 60 s decode: field order (idet's multi-frame TFF/BFF count reads the picture, so it
    works however the capture is flagged) and film or video (field cadence, voted per 5 s block so
    edits that break the cadence don't hide it). Returns (mode, field order, detail)."""
    import struct
    rate = info["fps"]
    # a capture app set to 50/60 fps (OBS) stores every tape frame twice: the 3:2 pattern then
    # spans 10 frames and can't be seen; look at every second frame (vhs_setup does the same
    # for the upscale if the tape turns out interlaced)
    dedup = ""
    if rate > 47:
        rate /= 2
        dedup = f"fps={rate.numerator}/{rate.denominator},"
    start = sample_start(info) if start is None else start
    # (no -ss for a sample from the start: an AVI with B-frames can't seek to 0)
    p = subprocess.run(["ffmpeg", "-hide_banner", "-nostdin",
                        *(["-ss", f"{start:.1f}"] if start else []),
                        "-t", "60", "-i", a.input, "-an", "-sn", "-filter_complex",
                        f"[0:v]{dedup}split=2[a][b];[a]idet[ai];[b]{VHS_CADENCE_VF}[bo]",
                        "-map", "[ai]", "-f", "null", "-",
                        "-map", "[bo]", "-f", "rawvideo", "-pix_fmt", "grayf32le", "pipe:1"],
                       capture_output=True)
    ms = re.findall(r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*"
                    r"Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)",
                    p.stderr.decode("utf-8", "replace"))
    if not ms or sum(map(int, ms[-1])) == 0:
        return "interlaced", "tff", (f"could not analyze (ffmpeg exit code {p.returncode}), "
                                     "assuming camcorder video, top field first")
    tff, bff, prog, und = map(int, ms[-1])
    parity = "bff" if bff > tff else "tff"
    n = len(p.stdout) // 8
    v = struct.unpack(f"<{2 * n}f", p.stdout[:8 * n])
    top, bot = v[2::2], v[3::2]                       # frame 0 has no previous frame
    blocks = votes = 0
    for i in range(0, max(1, len(top) - 99), 150):    # 5 s blocks (30 windows)
        st, pt = _phase_share(top[i:i + 150])
        sb, pb = _phase_share(bot[i:i + 150])
        if pt >= 0:
            blocks += 1
            film = min(st, sb) >= 0.40 and (pb - pt) % 5 in (2, 3)
            if not film:
                # held drawings: a top and a bottom phase 2-3 apart that stay low whenever
                # something moves (8+ windows; at most 3 such phases, else it is a still)
                lt, mt = _low_phases(top[i:i + 150])
                lb, mb = _low_phases(bot[i:i + 150])
                film = (min(mt, mb) >= 8 and 0 < len(lt) <= 3 and 0 < len(lb) <= 3
                        and lt != lb        # whole-frame events hit both fields alike
                        and any((q - p) % 5 in (2, 3) for p in lt for q in lb))
            votes += film
    secs = 150 / float(rate) if rate > 0 else 5
    progressive, note = False, ""
    if (tff + bff) < 0.1 * (tff + bff + prog + und):
        # no combing in the sample: either the file was deinterlaced already, or the sample is a
        # still/blank stretch of an interlaced tape (blue screen, snow, a tripod shot; measured:
        # idet calls those 0% interlaced). Treated as progressive, a tape would keep its combing
        # everywhere it moves, so only a file that says it is progressive, or has a frame rate
        # tapes don't have (23.976, 50, 59.94), counts as progressive
        fo = capture(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                      "stream=field_order", "-of", "csv=p=0", a.input])[1].strip().split(",")[0]
        tape_rate = min(abs(float(info["fps"]) - 30000 / 1001), abs(float(info["fps"]) - 25)) < 0.1
        progressive = fo == "progressive" or not tape_rate
        if not progressive:
            if tff + bff < 20:
                parity = "bff" if fo in ("bb", "tb") else "tff"     # the file's own flag, else TFF
            note = ("; the sample shows almost no interlacing (a still or blank stretch?), handled "
                    "as an interlaced tape: if the file was deinterlaced already, use --mode "
                    "progressive")
    detail = (f"{'bottom' if parity == 'bff' else 'top'} field first (idet {tff} TFF / {bff} BFF"
              f" / {prog} progressive), 3:2 film cadence in {votes} of {blocks} {secs:.0f}-second "
              f"blocks{note}")
    if progressive:
        return "progressive", parity, detail
    # no evidence counts as video: a movie taken for video only costs time (2.5x the frames),
    # video taken for film would lose 3 of every 5 motion steps
    film = blocks > 0 and votes >= (1 if blocks < 4 else max(2, 0.25 * blocks))
    return ("telecine" if film else "interlaced"), parity, detail


def vhs_chroma_delay(a, info):
    """How far the colour sits to the right of the picture (VHS colour-under), in pixels, or None
    if it can't be measured: cross-correlates the edges of the luma (blurred to the colour's
    bandwidth) and of Cb/Cr on 16 frames. ffmpeg averages every 16 rows, so Python only does 1-D
    work (1-2 s)."""
    import operator
    w = 704 if info["w"] >= 704 else info["w"] - 16
    h = (min(info["h"], 576) - 16) // 16
    start = sample_start(info) + (5.0 if info["duration"] > 30 else 0.0)
    deint = (f"bwdif=mode=send_frame:parity={a.parity}:deint=all,"
             if a.mode != "progressive" else "")
    vf = (f"{deint}select='not(mod(n\\,9))',crop={w}:{h * 16}:(iw-{w})/2:8,format=yuv444p,"
          f"gblur=sigma=3:sigmaV=0.01:planes=1,scale={w}:{h}:flags=area,format=yuv444p")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin",
                          *(["-ss", f"{start:.1f}"] if start else []), "-i", a.input,
                          "-an", "-sn", "-vf", vf, "-frames:v", "16", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    fs, seq = w * h, [[], [], []]
    for f in range(len(raw) // (3 * fs)):
        for k in range(3):
            pl = raw[(3 * f + k) * fs:(3 * f + k + 1) * fs]
            for r in range(h):
                ln = pl[r * w:(r + 1) * w]
                g = [abs(ln[i + 1] - ln[i - 1]) for i in range(1, w - 1)]
                m = sum(g) / len(g)
                seq[k] += [x - m for x in g] + [0.0] * 8      # gap: a shift never pairs two rows
    if not seq[0]:
        return None

    def lag(ys, cs, rng=8):
        n = len(cs)
        sc = [sum(map(operator.mul, ys[:n - s], cs[s:])) if s >= 0 else
              sum(map(operator.mul, ys[-s:], cs[:n + s])) for s in range(-rng, rng + 1)]
        i = max(range(len(sc)), key=sc.__getitem__)
        den = sc[i - 1] - 2 * sc[i] + sc[i + 1] if 0 < i < len(sc) - 1 else 0
        # how alike the edges are at that shift (normalized correlation)
        ncc = sc[i] / (math.sqrt(sum(x * x for x in ys) * sum(x * x for x in cs)) or 1)
        return i - rng + (0.5 * (sc[i - 1] - sc[i + 1]) / den if den else 0.0), ncc

    (db, nb), (dr, nr) = lag(seq[0], seq[1]), lag(seq[0], seq[2])
    # blank, blue or snowy sample: no colour edges to line up (measured: 0.00-0.01, pictures
    # 0.08-0.37), the peak is noise
    return (db + dr) / 2 if (nb + nr) / 2 >= 0.03 else None


def vhs_geometry(info, dar):
    """Crop to the picture (BT.601: the middle 704 of 720 samples are the 4:3 picture; 486-line
    captures: lines 4-483) and the square-pixel size it becomes, e.g. 704x480 -> 640x480."""
    w, h = info["w"], info["h"]
    ch = 480 if h == 486 else h
    cw = 704 if w >= 704 else w
    crop = (f"crop={cw}:{ch}:{(w - cw) // 2}:{4 if h == 486 else 0}"
            if (cw, ch) != (w, h) else None)
    return crop, round(ch * float(Fraction(dar)) / 2) * 2, ch


def vhs_prefilter(a):
    f = ["setpts=PTS-STARTPTS"] + ([a.vhs_dedup] if a.vhs_dedup else [])
    if a.mode == "telecine":
        # mchroma=0: tape colour is mostly noise; y0/y1: the head-switching lines at the bottom
        # are left out of the match decision; yadif: time-base jitter leaves even correctly
        # matched frames combed in places
        f += [f"fieldmatch=order={a.parity}:mchroma=0:y0={a.vhs_h - 18}:y1={a.vhs_h}",
              "yadif=deint=interlaced"]
        if a.vhs_fin > 26:
            f += ["decimate=chroma=0"]                   # (PAL film is 2:2: nothing to drop)
        # timeline from 0 again (decimate can start at +1 frame); a slight vertical blur calms
        # the line-to-line jitter of the rebuilt frames (measured: SSIM +0.004-0.017)
        f += ["setpts=PTS-STARTPTS", "gblur=sigma=0.01:sigmaV=0.5"]
    elif a.mode == "interlaced":
        f += [f"bwdif=mode=send_field:parity={a.parity}:deint=all"]    # one frame per field
    if a.vhs_crop:
        f += [a.vhs_crop]
    f += ["format=yuv444p"]                  # whole-pixel colour shifts need full-size chroma
    if a.chroma_shift:
        f += [f"chromashift={a.chroma_shift}"]
    l, r, t, b = a.mask
    # strong temporal chroma denoise: tape colour noise is streaky and different in every field
    f += [f"hqdn3d={a.denoise}", f"fillborders=left={l}:right={r}:top={t}:bottom={b}:mode=fixed",
          f"scale={a.vhs_sw}:{a.vhs_sh}:flags=lanczos", "setsar=1", f"fps={a.fps}"]
    if a.vhs_trim:
        f += [f"trim=start_frame={a.vhs_trim}", "setpts=PTS-STARTPTS"]   # chunk warm-up frames
    return ",".join(f)


def bar_rows_valid(a, ih, m):
    """m rows of black off the top and the bottom of an ih-line picture can be taken off before
    the upscale and put back after it: the output height of what is left, and the offset it
    sits at, must be whole even numbers (4:2:0 pictures)."""
    if m <= 0 or ih - 2 * m < ih // 2:
        return False
    keep, off = a.height * (ih - 2 * m), a.height * m
    return keep % ih == 0 and off % ih == 0 and (keep // ih) % 2 == 0 and (off // ih) % 2 == 0


def detect_bars(a, info, samples=24):
    """Black bars above and below a widescreen movie: the rows to cut off at the top and
    bottom (the same number, 0: none). They cost the upscaler as much as the picture does
    (2.39:1 on a DVD: a quarter of every frame), so the picture alone goes to the GPU and the
    bars are put back after it, the same picture. ffmpeg's cropdetect on `samples` spots across
    the movie: only the rows black at EVERY spot count (a scene with the picture over the whole
    frame, or a dark one that reads as bars, keeps the crop small), less a safety margin."""
    ih, dur = info["h"], float(info["duration"])
    tops, bots = [], []
    for k in range(samples):
        at = dur * (0.03 + 0.94 * k / max(1, samples - 1))
        rc, txt = capture(["ffmpeg", "-hide_banner", "-ss", f"{at:.1f}", "-i", a.input, "-an",
                           "-sn", "-vf", "cropdetect=24:2:0", "-frames:v", "25", "-f", "null",
                           "-"])
        found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", txt)
        if found:
            w, h, x, y = (int(v) for v in found[-1])
            tops.append(y)
            bots.append(ih - (y + h))
    if len(tops) < samples * 0.75:
        return 0                    # (couldn't read enough of the movie: no crop)
    m = min(min(tops), min(bots)) - 6           # (a margin: soft or noisy edges)
    for cand in range(m, 15, -1):
        if bar_rows_valid(a, ih, cand):
            return cand
    return 0


def prefilter(a):
    if a.type == "vhs":
        return vhs_prefilter(a)
    f = ["setpts=PTS-STARTPTS"]          # chunk timeline always starts at 0
    if getattr(a, "src_hd", False):
        # HD video is BT.709: an untagged file would be turned into RGB with SD colours
        f += ["setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709"]
    if a.mode == "telecine":
        # repeatfields first: frames the disc only flags to show for 3 fields (soft pulldown)
        # become those 3 fields, so the inverse telecine finds the same 3:2 pattern everywhere.
        # Without it a disc mixing soft and hard pulldown lost every 5th film frame in its soft
        # parts; on hard telecine it changes nothing (verified bit-identical).
        # Not for PAL: soft pulldown is an NTSC thing, and repeatfields drops the timestamps of
        # frames not flagged top field first unless the rate is 29.97; with no decimate after
        # it to make new ones, fps= then dropped every frame of BFF/progressive-flagged discs
        # (a disc with no soft pulldown at all goes without it: on a bottom-field-first disc
        #  repeatfields turned the movie's first frame into a copy of the second)
        if getattr(a, "pal", False):
            f += ["fieldmatch", "yadif=deint=interlaced"]   # PAL film is 2:2: nothing to drop
        else:
            f += ([] if getattr(a, "no_soft_pulldown", False) else ["repeatfields"]) \
                 + ["fieldmatch", "yadif=deint=interlaced", "decimate"]
    elif a.mode == "interlaced":
        f += ["bwdif=mode=send_frame:deint=all"]
    elif getattr(a, "combed", False):
        # mostly progressive, with interlaced stretches: idet marks each frame, and only the
        # ones it finds combed are deinterlaced (measured on an Avatar DVD clip: the 145 combed
        # frames of a camera move cleaned up, the other 764 frames left bit-identical)
        f += ["idet", "bwdif=mode=send_frame:deint=interlaced"]
    if a.dar:
        f += [f"scale=trunc(ih*{a.dar}/2)*2:ih:flags=lanczos"]
    else:
        f += ["scale=trunc(iw*sar/2)*2:ih:flags=lanczos"]      # anamorphic -> square pixels
    f += ["setsar=1", f"hqdn3d={a.denoise}"]
    if getattr(a, "crop_rows", 0):
        f += [f"crop=iw:ih-{2 * a.crop_rows}:0:{a.crop_rows}"]    # (the bars are put back after)
    if getattr(a, "dvd_trim", 0):
        f += [f"trim=start_frame={a.dvd_trim}", "setpts=PTS-STARTPTS"]   # chunk warm-up frames
    f += [f"fps={a.fps}"]
    return ",".join(f)


def training_filters(a):
    """The filters before the upscaler for this movie, without the chunk timing (start time, warm-up
    trim, output rate): upscale_training.py makes its DVD frames with them, so the model learns from
    frames exactly as the upscaler hands them over. None if they can't be worked out."""
    try:
        b = types.SimpleNamespace(**vars(a))
        b.fps, b.crop_rows, b.dvd_trim, b.vhs_trim = "1", 0, 0, 0
        return [f for f in prefilter(b).split(",") if f not in ("setpts=PTS-STARTPTS", "fps=1")]
    except Exception:
        return None


def postfilter(a):
    # exact width (multiple of 8, e.g. 1920 or 1440): some TVs and hardware decoders reject
    # odd sizes like 1918x1080
    m, ih = getattr(a, "crop_rows", 0), getattr(a, "crop_src_h", 0)
    if m and ih:
        # the picture without its bars, scaled to the same size it has in the full frame, then
        # the bars put back (black) where they were
        keep, off = a.height * (ih - 2 * m) // ih, a.height * m // ih
        f = [f"scale={a.out_w}:{keep}:flags=lanczos+accurate_rnd",
             f"pad={a.out_w}:{a.height}:0:{off}:black"]
    else:
        f = [f"scale={a.out_w}:{a.height}:flags=lanczos+accurate_rnd"]
    f += ["format=yuv420p10le",
         # SD (601) -> HD (709) colours. HD sources are 709 already; through the AI the frames
         # come back from RGB as 601 (the PNG step's default), so only --fast keeps their 709
         "colorspace=all=bt709:iall="
         + ("bt709" if a.fast and getattr(a, "src_hd", False) else a.matrix) + ":fast=1"]
    if a.smooth > 0 and not a.fast:
        # temporal-only denoise: calms detail the AI invents differently on each frame
        # (shimmering/pulsing backgrounds); hqdn3d backs off where things move.
        # Spatial must be 0.001, not 0: hqdn3d replaces 0 with its default (4:3 spatial blur)
        f += [f"hqdn3d=0.001:0.001:{a.smooth}:{a.smooth}"]
    f += [f"unsharp=5:5:{a.sharpen}:5:5:0.0", "setsar=1"]
    if not a.hevc:
        f += ["format=yuv420p"]          # H.264 for compatibility is 8-bit
    return ",".join(f)


# --ai-blend where the picture is busy: crowds of small people, grass, gravel, foliage. The models
# paint such dense small detail flat (people turn waxy and run together) and it shimmers from frame
# to frame, so there the plain upscale gets more say. "Busy" is the fine detail (Laplacian) of
# the source frame, averaged over SIGMA pixels: a single strong outline (a close-up face, a
# wall's edge) averages out low, a crowd or a lawn stays high. Below LO the AI keeps all of
# --ai-blend, from HI on it gets LOW_BLEND (at most --ai-blend), linearly in between. Measured on
# a real pedestrian video shrunk to DVD size, upscaled with realesrgan-x2plus and compared with
# the real full-size frames (in the people: SSIM 0.860 -> 0.879, gradient error 0.155 -> 0.140,
# shimmer of the still background 2.6x -> 2.2x the real video's; the same as --ai-blend 0.4
# everywhere, but smooth scenes keep the AI as they were)
DETAIL_BLEND = dict(sigma=6, lo=8, hi=25, low_blend=0.4)


def detail_blend_mask(a):
    """ffmpeg filters turning a source frame into the maskedmerge mask (0 = the AI frame,
    255 = the plain upscale) at the source frame's size."""
    d, high = DETAIL_BLEND, a.ai_blend
    low = min(high, d["low_blend"] if getattr(a, "busy_blend", None) is None else a.busy_blend)
    share = (f"{high}-({high}-{low})*clip((val-{d['lo']})/({d['hi']}-{d['lo']})\\,0\\,1)")
    return (f"format=gray,convolution=0m='0 -1 0 -1 4 -1 0 -1 0':0rdiv=1:0bias=128,"
            f"lut=y='abs(val-128)',gblur=sigma={d['sigma']},lut=y='255*(1-({share}))'")


def nvenc_args(a):
    """NVENC argument lists, best first; the first one this GPU accepts is used (nvenc errors
    out on unsupported options instead of ignoring them).
    No B-frames (-bf 0): NVENC gives B-frames ~3 QP worse quality than P-frames and has no
    setting to even that out in VBR mode, so fine detail pulsed sharp/soft every 4th frame.
    Keyframe every ~4 s (-g): quick, exact skipping on TVs and weaker players."""
    if a.hevc:
        codec, cq, fmt = ["-c:v", "hevc_nvenc"], "20", ["-pix_fmt", "p010le"]
    else:
        # H.264 High@4.1 8-bit: what Blu-rays use, plays on practically every device
        codec, cq = ["-c:v", "h264_nvenc"], "19"
        fmt = ["-pix_fmt", "yuv420p", "-profile:v", "high", "-level:v", a.level,
               "-maxrate", "40M", "-bufsize", "40M"]
    base = [*codec, "-rc", "vbr", "-cq", cq, "-b:v", "0", "-bf", "0", "-g", str(a.gop)]
    return [
        [*base, "-preset", "p7", "-tune", "hq", "-rc-lookahead", "20", "-spatial-aq", "1",
         "-multipass", "fullres", *fmt],
        [*base, "-preset", "p6", "-tune", "hq", "-spatial-aq", "1", *fmt],
        [*base, "-preset", "p6", *fmt],
    ]


def cpu_args(a):
    # tune grain keeps quality even across frame types (the defaults pulse on B-frames too)
    if a.hevc:
        return ["-c:v", "libx265", "-preset", "medium", "-tune", "grain", "-crf", "21",
                "-g", str(a.gop), "-pix_fmt", "yuv420p10le",
                "-x265-params", "log-level=error:open-gop=0"]   # every keyframe a clean seek point
    return ["-c:v", "libx264", "-preset", "slow", "-tune", "grain", "-crf", "19",
            "-g", str(a.gop), "-pix_fmt", "yuv420p", "-profile:v", "high", "-level:v", a.level,
            "-maxrate", "40M", "-bufsize", "40M"]


def encode_args(a):
    tags = ["-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-color_range", "tv"]
    return [*a.enc, *tags]


def pick_nvenc(a):
    """First NVENC argument list that actually encodes on this GPU, or None."""
    a.nvenc_error = ""
    # at the real size; VHS also at the real frame rate (59.94p needs level 4.2)
    rate = a.fps if a.type == "vhs" else "24000/1001"
    for args in nvenc_args(a):
        rc, txt = capture(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                         f"testsrc2=s={a.out_w}x{a.height}:r={rate}", "-frames:v", "24",
                         *args, "-f", "null", "-"])
        if rc == 0:
            return args
        lines = txt.strip().splitlines()
        # the encoder's own message (e.g. driver too old) says more than ffmpeg's last line
        a.nvenc_error = next((l for l in lines if "nvenc @" in l), lines[-1] if lines else "")
    return None


def video_start(path):
    """First video timestamp of a file (0 if unknown)."""
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=start_time",
         "-of", "csv=p=0", str(path)], encoding="utf-8", errors="replace",
        stdin=subprocess.DEVNULL, timeout=LONG_PROBE_TIMEOUT).strip()
    return float(frac(out))


def count_frames(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=READ_TIMEOUT)
    out = out.strip().split(",")[0]
    return int(out) if out.isdigit() else 0


def gpu_list(a):
    """--gpu 0,1 -> ["0", "1"]; no --gpu -> [None] (the upscaler picks the dedicated one)."""
    return str(a.gpu).replace(" ", "").split(",") if a.gpu is not None else [None]


def compact_model(a):
    """the small (SRVGGNetCompact) models: anime and VHS camcorder"""
    return any(m in a.model for m in ("animevideov3", "general"))


# what the GPU is given at once without --gpu-threads/--tile, first choice first: (frames at
# once, tile size). A chunk the upscaler fails on is tried again one step down, and the rest of
# the run keeps that (see Chunk.upscale; a movie of --all/--queue starts where the one before it
# ended, see gpu_step_file).
# - Frames at once: each frame is upscaled on its own, the same picture with any count. Lowered
#   first.
# - Tiles: the upscaler cuts each frame into tiles with a 10-pixel overlap, and the model sees
#   only its tile, so the tile size changes the picture a little all over (no visible grid):
#   x2plus on a DVD frame against the frame in one piece: 200 pixels 49 dB, 100 44 dB, 64 42 dB,
#   32 40 dB (faces a touch crisper or softer). Smaller tiles only when fewer frames didn't
#   help: the alternative is no movie at all.
# - the small (compact) anime/VHS-camcorder models: 8 frames, each in one piece (None: whole
#   frames; 21.5 frames/s on an RTX 3060 laptop)
# - the big x2plus/x4plus: 2 frames, the upscaler's own default, in its own tiles (None: 200
#   pixels on most GPUs). Windows resets a GPU whose piece of work takes over 2 seconds, and a
#   GPU can fault on a large piece ("vkQueueSubmit failed -4"). That laptop: x2plus reset it after
#   10-18 frames at 2 frames at once, 40-63 frames at 1, and in 100-pixel tiles after 163 frames
#   once and not in 1440 frames another time. A tile is one piece of work: smaller tiles, shorter
#   pieces, rarer resets; but each step down is slower too (64-pixel tiles: 20% more work than
#   100, 32: 80%), so a reset after a good stretch of frames doesn't step down (Chunk.upscale)
GPU_STEPS = {True: ((8, None), (6, None), (4, None), (2, None), (1, None), (1, 100), (1, 64)),
             False: ((2, None), (1, None), (1, 100), (1, 64), (1, 32))}  # compact?: steps


# the current ncnn (see ncnn_upscaler_main) does one frame at a time anyway
def ncnn_steps():
    """What the current ncnn is lowered to after a GPU reset, one step at a time: first the
    tile it started with (the saved one: whole frames on a big GPU, else 200), then ever
    smaller ones. Each step halves the piece of work, and costs a little speed and nothing a
    viewer can see until the very small ones: from whole frames to 100 in one go gave up much
    more than needed. Only tiles below the one it started with."""
    cap = int(ncnn_saved().get("tile") or 200)
    return ((1, None),) + tuple((1, t) for t in (512, 256, 128, 64, 32) if t < cap)


GPU_STEP = [0]          # how many steps down this run has gone


def gpu_steps(a):
    return ncnn_steps() if getattr(a, "engine", "exe") == "ncnn" else GPU_STEPS[compact_model(a)]


def gpu_load(a, step=None):
    """(frames at once, tile size or None) at a step (default: the current one); --gpu-threads
    and --tile, when given, are kept at every step"""
    steps = gpu_steps(a)
    threads, tile = steps[min(GPU_STEP[0] if step is None else step, len(steps) - 1)]
    return (a.gpu_threads if a.gpu_threads is not None else threads,
            a.tile or tile)


def gpu_load_text(a):
    threads, tile = gpu_load(a)
    return (f"{threads} frame{'s' if threads > 1 else ''} at once on the GPU"
            + (f", in {tile}-pixel tiles" if tile and str(tile) != "0" else ""))


def lower_gpu_load(a):
    """One step less on the GPU at once for the rest of the run (steps that don't change
    anything next to --gpu-threads/--tile are passed over); False if there is none left."""
    if getattr(a, "no_step_down", False):
        return False            # (--no-step-down: the settings stay as they are)
    steps = gpu_steps(a)
    now = gpu_load(a)
    for k in range(GPU_STEP[0] + 1, len(steps)):
        if gpu_load(a, k) != now:
            GPU_STEP[0] = k
            save_gpu_step(a)
            return True
    return False


def gpu_step_file():
    """Where a run leaves how far down the GPU needed to go: gpu_steps.json next to the script,
    so the next run (and the next movie of --all/--queue) starts there instead of resetting the
    GPU again on the way down. Delete it to start from the top again (after a driver update,
    say). DVD_UPSCALE_GPU_STEPS: another file (the tests)."""
    return Path(os.environ.get("DVD_UPSCALE_GPU_STEPS")
                or Path(__file__).resolve().parent / "gpu_steps.json")


def gpu_step_key(a):
    # (per model, x4plus being 4x the work; per engine; per --gpu)
    return (a.model + ("/ncnn" if getattr(a, "engine", "exe") == "ncnn" else "")
            + (f"@{a.gpu}" if a.gpu else ""))


FRAMES_OK = [0]         # frames upscaled since the last GPU reset (this model and engine)


def load_gpu_step(a):
    """The first step no bigger than the one saved: (frames, tile) is kept, not the step number,
    so a changed list of steps still reads it right. Also the frames since the last reset."""
    if getattr(a, "no_step_down", False):
        return False            # (--no-step-down: what an earlier run went down to is ignored)
    try:
        saved = json.loads(gpu_step_file().read_text(encoding="utf-8"))
        FRAMES_OK[0] = int(saved.get(gpu_step_key(a) + "#ok", 0))
        threads, tile = saved[gpu_step_key(a)]
        for k, (t2, p2) in enumerate(gpu_steps(a)):
            # (no tile: whole frames for the compact models, 200 for the big ones)
            if t2 <= int(threads) and (p2 or 10 ** 4) <= (int(tile) if tile else 10 ** 4):
                GPU_STEP[0] = k
                return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return False


def save_gpu_step(a, frames_only=False):
    """frames_only: just the frames since the last reset (after each chunk, once the file
    exists: a run with no reset ever leaves no file)"""
    path = gpu_step_file()
    try:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved = saved if isinstance(saved, dict) else {}
        except (OSError, ValueError):
            if frames_only:
                return
            saved = {}
        if not frames_only:
            saved[gpu_step_key(a)] = list(gpu_steps(a)[GPU_STEP[0]])
        saved[gpu_step_key(a) + "#ok"] = FRAMES_OK[0]
        path.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    except OSError:
        pass


def esrgan_cmd(a, src, dst, size=None, gpu=None):
    compact = compact_model(a)
    gpu = gpu if gpu is not None else gpu_list(a)[0]
    threads, tile = gpu_load(a)
    # (the current ncnn: this script as the upscaler, with the same command line)
    exe = ([sys.executable, Path(__file__).resolve(), "--ncnn-upscaler"]
           if getattr(a, "engine", "exe") == "ncnn" else [a.esrgan_path])
    # threads to load:upscale:save frames (default 1:2:2): reading and writing the PNGs is CPU
    # work that otherwise leaves the GPU waiting
    cmd = [*exe, "-i", src, "-o", dst, "-n", a.model, "-s", a.scale, "-f", "png",
           "-j", f"2:{threads}:4"]
    models = models_dir(a)
    if models.is_dir():
        cmd += ["-m", models]
    if gpu is not None:
        cmd += ["-g", gpu]
    if tile:
        cmd += ["-t", tile]
    elif size and compact:
        # the small (compact) models need little GPU memory: a whole frame in one piece instead
        # of the default 200-pixel tiles, whose overlaps cost ~20% extra work (and leave no
        # tile seams); the big x2plus/x4plus models keep the default
        cmd += ["-t", -(-max(size) // 32) * 32]
    return cmd


# what the upscaler prints, and carries on with exit code 0, when the GPU fails mid-run (driver
# reset, out of video memory) or a frame can't be read or written: its frames are then black,
# garbled or missing
GPU_ERRORS = re.compile(r"vk(QueueSubmit|WaitForFences|AllocateMemory|MapMemory)\w* failed|"
                        r"VK_ERROR_DEVICE_LOST|device lost|(en|de)code image .* failed|"
                        r"ncnn: GPU error", re.I)


def upscaler_log_tail(path, limit=8):
    """A short diagnostic excerpt for silent/incomplete upscaler runs (exit code 0)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    lines = [line.strip() for line in text.splitlines()
             if line.strip() and not line.strip().endswith("%")]
    if not lines:
        return ""
    return "\nUpscaler output (end):\n  " + "\n  ".join(
        line[-240:] for line in lines[-limit:])


def _read_log_updates(path, offset, pending=""):
    """Read only bytes appended since the last check, retaining an unfinished log line."""
    with open(path, "rb") as log:
        log.seek(offset)
        data = log.read()
        offset = log.tell()
    pending += data.decode("utf-8", "replace")
    lines = pending.splitlines(keepends=True)
    pending = ""
    if lines and not lines[-1].endswith(("\n", "\r")):
        pending = lines.pop()
    return offset, pending, lines


LANE_STATUS = {}     # what each helper GPU is doing, shown on the main progress line
# each helper GPU's upscale while it runs: [frames done, of, started, last new frame] (times)
LANE_PROGRESS = {}
# every upscaler's frames so far while it runs, the main one's too (""): lane -> (GPU, frames)
UPSCALER_FRAMES = {}
# the chunks being upscaled right now (on any GPU): chunk -> (bytes of frames it writes, frames)
UPSCALING = {}
DISK_LOCK = threading.Lock()


class GPUError(RuntimeError):
    """The upscaler reported a GPU failure (a reset, out of video memory). good: the frames it
    had finished before the first error (their names in the output folder)."""
    good = ()


GPU_REPORTS = []        # GPU errors written to gpu_errors.log this run
RESET_TIMES = []        # when the GPU was reset (time.time())


def gpu_report(a, work, label, err, tried, log_path):
    """After a GPU error: what happened, appended to <work folder>/gpu_errors.log (to send with a
    bug report), with what Windows recorded about its graphics drivers in the last 15 minutes and
    the NVIDIA GPU's state. Windows' records tell the causes apart: "Display" event 4101 or
    nvlddmkm 153 (the driver restarted the GPU after a timeout: a piece of work took over 2 s);
    nvlddmkm 13/14 or an "Xid" (the GPU hit a fault). Returns a one-line summary of those
    records ("" if none)."""
    GPU_REPORTS.append(time.time())
    out = [f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}  {label}  {a.model} x{a.scale}, {tried}",
           str(err)]
    try:
        text = log_path.read_bytes().decode("utf-8", "replace")
    except OSError:
        text = ""
    out += ["upscaler: " + x.strip() for x in text.splitlines()
            if re.match(r"\[\d+ ", x.strip()) or "fp16-" in x or "subgroup" in x][:12]

    def tool(cmd):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                               stdin=subprocess.DEVNULL, timeout=30)
            return p.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    summary = ""
    if len(GPU_REPORTS) <= 5:           # (the first few are what tell)
        smi = tool(["nvidia-smi", "--query-gpu=name,driver_version,pstate,temperature.gpu,"
                    "power.draw,power.limit,clocks.sm,clocks.max.sm,memory.used,memory.total,"
                    "utilization.gpu,clocks_throttle_reasons.active", "--format=csv"])
        if smi:
            out += ["nvidia-smi: " + x for x in smi.splitlines()]
        if sys.platform == "win32":
            ev = tool(["wevtutil", "qe", "System", "/c:12", "/rd:true", "/f:text",
                       "/q:*[System[(Provider[@Name='nvlddmkm'] or Provider[@Name='Display'] or "
                       "Provider[@Name='amdkmdag'] or Provider[@Name='igfx']) and "
                       "TimeCreated[timediff(@SystemTime) <= 900000]]]"])
            found = []
            for block in re.split(r"(?m)^Event\[\d+\]:", ev)[1:]:
                src = re.search(r"Source:\s*(.+)", block)
                eid = re.search(r"Event ID:\s*(\d+)", block)
                desc = block.split("Description:", 1)[-1].strip().splitlines()
                if src and eid:
                    found.append(f"{src.group(1).strip()} {eid.group(1)}"
                                 + (f" ({desc[0].strip()[:90]})" if desc and desc[0].strip()
                                    else ""))
            out += ["Windows: " + x for x in found] or ["Windows: no display driver events "
                                                         "in the last 15 minutes"]
            summary = "; ".join(found[:2])
    try:
        with open(work / "gpu_errors.log", "a", encoding="utf-8") as f:
            f.write("\n".join(out) + "\n\n")
    except OSError:
        pass
    return summary


class HandBack(Exception):
    """A helper GPU gives its chunk back to the main one (stopping, or the end of the video)."""


def lane_stopped(a, lane):
    """A helper GPU stops: everything is stopping (Ctrl+C, a failure), or its chunk was taken
    back by the main GPU."""
    return a.stop_lanes.is_set() or lane in a.cancel_lanes


def frames_to_come(job, frames, n_in):
    """Of a chunk being upscaled: the bytes of upscaled frames still to be written."""
    try:
        done = sum(1 for _ in os.scandir(job.tmp / "out"))
    except OSError:
        return 0
    return frames * max(0, n_in - done) / max(1, n_in)


def run_upscaler(a, src, dst, n_in, label, size=None, gpu=None, lane=None):
    """Real-ESRGAN prints its GPU details and a percentage for every tile of every frame: that
    goes to a log file, and one progress line is shown instead (the log's end if it fails).
    lane: a helper GPU (--gpu 0,1): its progress goes into LANE_STATUS instead, and it stops
    when a.stop_lanes is set."""
    log_path = Path(dst).parent / "upscaler_log.txt"
    hung = stopped = False
    # frames there at the last look at the log without a GPU error (at first: the ones a try
    # before this one finished, kept by Chunk.upscale)
    good = [e.name for e in os.scandir(dst)]
    with open(log_path, "wb") as log:
        p = subprocess.Popen([str(c) for c in esrgan_cmd(a, src, dst, size, gpu)],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            # watchdog, counted in 1 s waits rather than clock time so a laptop that slept in
            # between isn't taken for a stall
            last, same, all_there, tick = -1, 0, 0, 0
            log_offset, pending_log = 0, ""
            while True:
                names = [e.name for e in os.scandir(dst)]
                n = len(names)
                if n:           # (for the check of two upscalers on one GPU, see main)
                    UPSCALER_FRAMES[lane or ""] = (gpu if gpu is not None else gpu_list(a)[0], n)
                tick += 1
                if lane:
                    LANE_STATUS[lane] = f"{lane}: {label} {n}/{n_in}"
                    now = time.time()
                    prog = LANE_PROGRESS.setdefault(lane, [0, n_in, now, now])
                    if n > prog[0]:
                        prog[0], prog[3] = n, now
                    if lane_stopped(a, lane):
                        raise RuntimeError("stopped")
                else:
                    status_line(f"  {label}: upscaling frame {n} of {n_in}"
                                + "".join(f" | {x}" for x in list(LANE_STATUS.values())))
                # a GPU error: the upscaler would go on to the last frame (black or garbled
                # frames, a few seconds each after a GPU reset): stopped now instead. A helper
                # GPU's chunk goes back to the main GPU; the main GPU's chunk is tried again
                if tick % 5 == 0:
                    log_offset, pending_log, log_lines = _read_log_updates(
                        log_path, log_offset, pending_log)
                    bad = [line for line in (*log_lines, pending_log) if GPU_ERRORS.search(line)]
                    if not bad:
                        # (listed before this look at the log, so written before any error:
                        # the upscaler prints a GPU error before it saves the frame it spoils)
                        good = names
                    if bad and lane:
                        raise RuntimeError(f"the upscaler reported errors: "
                                           f"{bad[0].strip()[:100]}")
                    if bad:
                        stopped = True
                        break
                try:
                    p.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    pass
                same, last = (same + 1 if n == last else 0), n
                all_there = all_there + 1 if n >= n_in else 0
                if all_there > 60:          # every frame written a minute ago, still running
                    hung = True
                    break
                if same > 900:
                    raise RuntimeError("the upscaler stopped responding (no new frame for 15 "
                                       "minutes: a GPU driver hang?)")
        finally:
            if p.poll() is None:            # Ctrl+C or an error here: don't leave it running
                p.kill()
                p.wait()
            UPSCALER_FRAMES.pop(lane or "", None)
            if lane:
                LANE_PROGRESS.pop(lane, None)
    text = log_path.read_bytes().decode("utf-8", "replace")
    if not lane and not getattr(a, "gpu_shown", False):
        a.gpu_shown = True
        gpus = re.findall(r"^\[(\d+) (.+?)\]\s+queueC", text, re.M)
        if a.gpu is not None:
            devs = gpu_list(a)
            names = dict(gpus)
            if all(names.get(g) for g in devs):
                say(f"Upscaling on GPU {devs[0]}: {names[devs[0]]}" + "".join(
                    f", with GPU {g} ({names[g]}) upscaling whole chunks alongside it"
                    for g in devs[1:]))
        elif len(gpus) > 1:
            say("GPUs: " + ", ".join(f"{i} = {n}" for i, n in gpus)
                + " (the upscaler takes the dedicated one; --gpu N picks another)")
        elif gpus:
            say(f"Upscaling on: {gpus[0][1]}")
    errors = [x for x in text.splitlines() if GPU_ERRORS.search(x)]
    if (p.returncode and not hung) or errors:
        lines = errors or [x for x in text.splitlines()
                           if x.strip() and not x.strip().endswith("%")]
        # (two threads writing at once can run two error messages into one line)
        lines = [y for x in lines for y in re.split(r"(?<=\S)(?=vk[A-Z]\w* failed)", x)]
        # (a GPU reset gives one line per piece of work left: shown once, with a count)
        same = []
        for x in lines:
            if same and same[-1][0] == x:
                same[-1][1] += 1
            else:
                same.append([x, 1])
        lines = [x if k == 1 else f"{x}   (x{k})" for x, k in same]
        reset = re.search(r"(QueueSubmit|WaitForFences) failed -4\b|DEVICE_LOST|device lost|"
                          r"ncnn: GPU error \(extract returned -[14]\)", text, re.I)
        if lane:        # a helper GPU: one line, in the note that it stopped helping
            raise RuntimeError(("the upscaler reported errors" if errors else
                                f"the upscaler failed (exit code {p.returncode})")
                               + (f": {lines[0 if errors else -1].strip()[:100]}"
                                  if lines else ""))
        status_line()
        print("Upscaler output (end):\n" + "\n".join(lines[-25:]), flush=True)
        if re.search(r"invalid gpu device", text, re.I):
            print("(no GPU with that --gpu number: leave --gpu out, or see the GPU list the "
                  "upscaler prints at the top of its output)", flush=True)
        elif re.search(r"vkCreateInstance|vkEnumeratePhysicalDevices|no vulkan", text, re.I):
            print("(no usable Vulkan graphics driver: install or update the GPU's driver)",
                  flush=True)
        elif reset:
            print("(the GPU was reset: Windows resets a graphics card whose piece of work takes "
                  "over 2 seconds, or that faults on it; smaller tiles are smaller pieces of "
                  "work)", flush=True)
        elif re.search(r"memory|vkAllocate", text, re.I):
            print("(the GPU may have run out of memory: try adding --tile 128, or --gpu-jobs 1)",
                  flush=True)
        elif re.search(r"encode image", text):
            print("(it couldn't write the frames: is the work folder's drive full?)", flush=True)
        how = "stopped at the first one" if stopped else f"exit code {p.returncode}"
        err = None
        if reset:
            err = GPUError(f"the GPU was reset ({(errors or lines)[0].strip()[:60]}, "
                           f"{len(good)} of {n_in} frames done)")
        elif any(not re.search(r"(en|de)code image", x) for x in errors):
            err = GPUError(f"the upscaler reported GPU errors ({how})")
        if err:
            err.good = tuple(good)
            raise err
        if errors:
            raise RuntimeError(f"the upscaler reported errors ({how})")
        raise subprocess.CalledProcessError(p.returncode, f"{a.esrgan} (upscaler)")


def png_is_black(path):
    """A frame that is black through and through: every pixel 0. Exact: a PNG whose filtered
    rows are all zeros is all zeros whatever the filters (each pixel is built from zero
    neighbours), so decoding it is just one decompression. Only small files are looked at (a
    black 852x480 frame is a few KB); 8-bit RGB(A) only. Anything else: False (it is upscaled)."""
    import zlib
    try:
        if os.path.getsize(path) > 65536:
            return False
        data = Path(path).read_bytes()
        if data[:8] != b"\x89PNG\r\n\x1a\n":
            return False
        pos, idat, w, bpp = 8, [], 0, 0
        while pos + 8 <= len(data):
            n, kind = int.from_bytes(data[pos:pos + 4], "big"), data[pos + 4:pos + 8]
            body = data[pos + 8:pos + 8 + n]
            if kind == b"IHDR":
                w, depth, ctype, interlace = (int.from_bytes(body[0:4], "big"), body[8], body[9],
                                              body[12])
                if depth != 8 or ctype not in (2, 6) or interlace:
                    return False
                bpp = 3 if ctype == 2 else 4
            elif kind == b"IDAT":
                idat.append(body)
            pos += 12 + n
        if not idat or not w:
            return False
        raw, row = zlib.decompress(b"".join(idat)), 1 + w * bpp
        if len(raw) % row:
            return False
        # (the first byte of each row is its filter type, not a pixel)
        return all(not any(raw[i + 1:i + row]) for i in range(0, len(raw), row))
    except (OSError, ValueError, IndexError):
        return False


def write_black_png(path, w, h):
    """An all-black 8-bit RGB PNG of w x h pixels."""
    import zlib
    raw = bytes(1 + w * 3) * h

    def chunk(kind, body):
        return (len(body).to_bytes(4, "big") + kind + body
                + (zlib.crc32(kind + body) & 0xFFFFFFFF).to_bytes(4, "big"))
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n"
                          + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                          + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def check_frames(a, tmp, n_in, strict=False):
    """The upscaler can report success with black, garbled, empty or cut-off frames (GPU fault,
    full disk): every frame must be a whole PNG of the right size, and a sample of them about as
    bright as the frames that went in. strict: a brightness check that didn't run to the end
    (Ctrl+C) is a failure too (for a helper GPU, whose chunk then goes back)."""
    w, h = png_size(next(iter(sorted((tmp / "in").glob("*.png")))))
    size = (w * a.scale).to_bytes(4, "big") + (h * a.scale).to_bytes(4, "big")
    for f in sorted((tmp / "out").glob("*.png")):
        try:
            with open(f, "rb") as fh:
                head = fh.read(24)
                fh.seek(-12, os.SEEK_END)
                tail = fh.read(12)
        except OSError:
            head = tail = b""
        if head[:8] != b"\x89PNG\r\n\x1a\n" or head[16:24] != size or tail[4:8] != b"IEND":
            raise RuntimeError(f"the upscaler wrote a broken frame ({f.name}): is the work "
                               "folder's drive full?")
    picks = range(1, n_in + 1, max(1, n_in // 12))

    def brightness(folder):
        lst = tmp / f"check_{folder}.txt"
        lst.write_text("".join(f"file '{folder}/{i:06d}.png'\n" for i in picks), encoding="utf-8")
        rc, txt = capture(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                           "-vf", "format=yuv420p,signalstats,metadata=mode=print:key="
                           "lavfi.signalstats.YAVG"
                           ":file=-", "-f", "null", "-"])
        if strict and rc:
            raise RuntimeError("the check of the upscaled frames didn't finish")
        return [float(x) for x in re.findall(r"YAVG=([\d.]+)", txt)]
    before, after = brightness("in"), brightness("out")
    if len(after) < len(before):
        # (whole PNGs by their first and last bytes, but some can't be decoded: garbage inside)
        raise RuntimeError(f"{len(before) - len(after)} of the upscaled frames checked can't be "
                           "read (garbled - a GPU fault?)")
    if len(before) == len(after):
        for i, (x, y) in zip(picks, zip(before, after)):
            if abs(x - y) > 20:
                raise RuntimeError(f"upscaled frame {i} looks wrong (brightness {y:.0f} instead "
                                   f"of about {x:.0f}: black or garbled - a GPU fault?)")


TRAINED_MODEL = "upscale-training-x2"   # the single model older versions of upscale_training.py made


def trained_model(a):
    """--trained: the model upscale_training.py made for this kind of movie (it names them ai-anime-x2,
    ai-cgi-x2, ai-live-x2, ai-vhs-x2; from a 4K Blu-ray ai-<type>-x4), else the single model older versions
    made. --scale 2/4 picks the size; without it a 4K output (--height over 1080) prefers the x4 model and a
    1080p one the x2 model, each falling back on the other. Sets a.scale to the model's."""
    x2 = [(f"ai-{a.type}-x2", 2), (TRAINED_MODEL, 2)]
    x4 = [(f"ai-{a.type}-x4", 4)]
    S = a.trained_scale
    order = (x4 if S == 4 else x2) if S else (x4 + x2 if a.height > 1080 else x2 + x4)
    for m, sc in order:
        a.model, a.scale = m, sc
        if model_installed(a):
            if m == TRAINED_MODEL:
                print(f"NOTE: --trained: no ai-{a.type}-x2 model yet (upscale_training.py makes it from a "
                      f"{TYPE_NAMES[a.type]} movie): using {TRAINED_MODEL}, the earlier trained model")
            elif m != order[0][0]:
                print(f"NOTE: --trained: no {order[0][0]} model, using {m}"
                      + (" (made from a 4K Blu-ray: --height 2160 gives its full detail)" if sc == 4 else ""))
            return m
    a.model, a.scale = order[0]
    return a.model                      # (not there: reported as missing below)


def models_dir(a):
    return Path(a.esrgan_path).resolve().parent / "models"     # resolve() follows symlinks


def model_installed(a):
    d = models_dir(a)
    if not d.is_dir():
        return True     # unusual install layout: let the upscaler find its own models
    names = [a.model, f"{a.model}-x{a.scale}"]       # animevideov3 files carry the scale
    return any((d / f"{n}.param").exists() and (d / f"{n}.bin").exists() for n in names)


def check_upscaler(a):
    """Before the movie, a few seconds: the upscaler and its model on two small test pictures.
    Model files that are damaged, or a .param and a .bin that don't belong together (from two
    different downloads), give a GPU error or a smeared, garbled picture on every chunk: this
    says so at once, instead of after every chunk has failed (or a whole movie came out wrong).
    The test: each picture shrunk by the model's scale, then upscaled by the model, comes out
    nearly as close to the original as a plain resize does (measured on the anime, live and x4
    models: 0.3 better to 3.7 dB worse; a .bin from another conversion of realesrgan-x2plus:
    7.6 dB worse, another model's .bin: far worse). One frame at a time: the frames-at-once
    setting doesn't come into it."""
    with tempfile.TemporaryDirectory(prefix="upscaler_check_") as d:
        d = Path(d)
        for sub in ("big", "in", "out"):
            (d / sub).mkdir()
        k = a.scale
        for n, pattern in enumerate(("testsrc", "smptehdbars"), 1):
            run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"{pattern}=s=480x288:d=1",
                 "-frames:v", "1", str(d / "big" / f"{n:06d}.png")])
            run(["ffmpeg", "-y", "-v", "error", "-i", str(d / "big" / f"{n:06d}.png"), "-vf",
                 f"scale=iw/{k}:ih/{k}:flags=area", str(d / "in" / f"{n:06d}.png")])
        cmd = [str(c) for c in esrgan_cmd(a, d / "in", d / "out", (480 // k, 288 // k))]
        cmd[cmd.index("-j") + 1] = "1:1:1"
        try:
            p = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=600)
            text, rc = (p.stdout + p.stderr).decode("utf-8", "replace"), p.returncode
        except subprocess.TimeoutExpired:
            text, rc = "no result after 10 minutes", "a hang"
        errors = [x.strip() for x in text.splitlines() if GPU_ERRORS.search(x)]
        files = (f"{models_dir(a)}: {a.model}.param and {a.model}.bin must come from the same "
                 "download (for realesrgan-x2plus: unzip both from realesrgan-x2plus.zip again, "
                 "replacing the old ones)")
        outs = [d / "out" / f"{n:06d}.png" for n in (1, 2)]
        if rc or errors or not all(o.exists() for o in outs):
            status_line()
            lines = errors or [x.strip() for x in text.splitlines()
                               if x.strip() and not x.strip().endswith("%")]
            if getattr(a, "engine", "exe") == "ncnn":
                # (stdout is read before stderr, and ncnn's list of GPUs is on stderr: the last
                # line is then always the last GPU, never the reason. The reason is the
                # worker's own "ncnn: ..." line, or none at all: it crashed)
                own = [x for x in lines if x.startswith("ncnn")]
                why = (own[-1][:160] if own else "no message from the upscaler") + \
                    f"; it ended with exit code {rc}" + \
                    (" (a crash inside ncnn or the graphics driver)" if isinstance(rc, int)
                     and (rc < 0 or rc > 255) else "")
                print("  the current ncnn's output:", flush=True)
                for x in lines[-12:]:
                    print("    " + x[:160], flush=True)
                return ncnn_fallback(a, why)
            reset = re.search(r"QueueSubmit failed -4\b|DEVICE_LOST|device lost", text, re.I)
            sys.exit(f"The upscaler failed on a small test picture, before the movie started "
                     f"({lines[-1][:120] if lines else f'exit code {rc}'}).\n"
                     + ("The GPU was reset. " if reset else "")
                     + "If other movies work (anime, say) and only this kind fails, the model's "
                     f"files are the likely cause: {files}. Otherwise update the graphics "
                     "driver, and try the upscaler's own test in its folder: "
                     "realesrgan-ncnn-vulkan -i input.jpg -o test.png")

        def psnr(x, ref, scale_up=""):
            rc2, report = capture(["ffmpeg", "-v", "info", "-i", str(x), "-i", str(ref),
                                   "-lavfi", f"[0:v]{scale_up or 'null'}[x];[x][1:v]psnr",
                                   "-f", "null", "-"])
            m = re.search(r"PSNR .*?average:([0-9.]+)", report)
            return float(m.group(1)) if m else None
        diffs = []
        for n, o in enumerate(outs, 1):
            big = d / "big" / f"{n:06d}.png"
            model = psnr(o, big)
            plain = psnr(d / "in" / f"{n:06d}.png", big, f"scale=iw*{k}:ih*{k}:flags=lanczos")
            if model is not None and plain is not None:
                diffs.append(model - plain)
        if len(diffs) == 2 and sum(diffs) / 2 < -5.5:
            status_line()
            if getattr(a, "engine", "exe") == "ncnn":
                return ncnn_fallback(a, f"a smeared picture, {-sum(diffs) / 2:.1f} dB worse "
                                        "than a plain resize")
            sys.exit(f"The upscaler's model {a.model} gives a smeared or garbled picture: two "
                     f"small test pictures came out {-sum(diffs) / 2:.1f} dB worse than a plain "
                     "resize (a working model: within 4). Its files are damaged or don't belong "
                     f"together. {files}.")


def ncnn_fallback(a, why):
    """The current ncnn failed the start-up test: realesrgan-ncnn-vulkan instead (tested too)."""
    print(f"NOTE: the current ncnn didn't work here ({why}): using realesrgan-ncnn-vulkan's own "
          "engine instead (--engine exe). On NVIDIA drivers from 570 on that engine resets the "
          "GPU now and then and is much slower: send the lines above if you want this fixed",
          flush=True)
    a.engine, GPU_STEP[0], FRAMES_OK[0] = "exe", 0, 0
    load_gpu_step(a)
    print(f"GPU settings: {gpu_load_text(a)}; "
          + (f"up to {a.gpu_jobs} upscalers per GPU" if a.gpu_jobs > 1 else "one upscaler per GPU"))
    return check_upscaler(a)


def check_ffmpeg():
    """Exit with a clear message on an ffmpeg older than 5.1: its errors would only come later
    and say little (-fps_mode, filter options). Builds named by date or git commit pass."""
    rc, txt = capture(["ffmpeg", "-hide_banner", "-version"])
    m = re.match(r"ffmpeg version n?(\d+)\.(\d+)", txt)
    if m and (int(m[1]), int(m[2])) < (5, 1):
        sys.exit(f"This ffmpeg is version {m[1]}.{m[2]}: 5.1 or newer is needed. Download a "
                 "current build (ffmpeg.org lists them) and put it first on the PATH or next "
                 "to this script.")


def flush_to_disk(path):
    """Make a finished file's data durable before it is renamed into place: else a power cut
    soon after can leave a renamed but cut-short file that a resumed run trusts."""
    try:
        with open(path, "rb+") as f:
            os.fsync(f.fileno())
    except OSError:
        pass


def replace_file(src, dst):
    """os.replace, retried for a few seconds on Windows, where a virus scanner or the search
    indexer briefly holds a file it has just seen written."""
    for attempt in range(20):
        try:
            return os.replace(src, dst)
        except PermissionError:
            if os.name != "nt" or attempt == 19:
                raise
            time.sleep(0.5)


def write_durably(path, text):
    """Write a small file (settings.json) so that a crash or power cut never leaves it half
    written (which used to mean starting the movie over): a temporary file, flushed to disk,
    then renamed over it."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    replace_file(tmp, path)


def keep_awake():
    """CPU/GPU work doesn't count as activity: keep the computer from sleeping while this runs
    (released automatically when it exits; closing a laptop's lid still sleeps)."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        try:
            # (above-normal priority, inherited by the upscaler and ffmpeg processes: Windows
            # gives a busy browser or antivirus scan the processor first otherwise, and the GPU
            # then waits for the upscaler's next frame)
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass(ctypes.c_void_p(k32.GetCurrentProcess()), 0x00008000)
        except (OSError, AttributeError):
            pass
        return
    # macOS: caffeinate, Linux with systemd: an inhibitor; each lasts until this run exits
    pid = str(os.getpid())
    for cmd in (["caffeinate", "-i", "-w", pid],
                ["systemd-inhibit", "--what=sleep:idle", "--who=dvd_upscale.py",
                 "--why=upscaling a video", "tail", f"--pid={pid}", "-f", "/dev/null"]):
        if shutil.which(cmd[0]):
            try:
                subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            except OSError:
                pass
            return


# ---- keeping a long run safe: stray clicks and keys, sleep, shutdown (Windows) ----------------
# A night's run is lost to small things: a click in the window (QuickEdit mode freezes the
# program until a key is pressed, so the whole run just stops), keys typed while it runs (the
# console keeps them, and PowerShell runs them as a command once the script ends), a shutdown or
# a Windows Update restart. protect_run guards against them while the movie or queue runs; all of
# it is undone when the script ends. --no-guard turns it off (sleep is still prevented).
# Ctrl+C still stops the run, as it always has: it is the way to stop on purpose.

_GUARD = {"console": None, "window": None, "thread": None, "reason": "", "proc": None,
          "movie": ""}
WM_APP_REASON = 0x8000 + 1          # (WM_APP + 1: the reason text changed)


def guard_off():
    return os.name != "nt" or bool(os.environ.get("DVD_UPSCALE_NO_GUARD"))


def protect_run(movie=""):
    """Called once a movie (or a queue) really starts: sleep, clicks, stray keys, shutdown."""
    keep_awake()
    _GUARD["movie"] = movie
    if guard_off():
        return
    console_guard()
    # (a movie of --all/--queue: its queue's run blocks the shutdown and gave the warnings)
    if not os.environ.get("DVD_UPSCALE_QUEUE"):
        block_shutdown(f"dvd_upscale.py is upscaling {movie or 'movies'}: shutting down now "
                       "loses the chunk in progress (finished chunks are kept)")
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
            wc.lpszClassName = "dvd_upscale_shutdown_guard"
            u32.RegisterClassW(ctypes.byref(wc))       # (0 if already registered: fine)
            # a top-level window that is never shown: only those are asked before a shutdown
            hwnd = u32.CreateWindowExW(0, wc.lpszClassName, "dvd_upscale.py", 0, 0, 0, 0, 0,
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
    """Take a lock file, or exit with message if another run holds it. The OS releases the
    lock by itself when this run exits, crashes or the PC shuts down, so it never needs
    clearing. Keep the returned file open for as long as the lock is needed."""
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


def lock_work(work):
    """Lock the work folder so two runs can't use it at once (e.g. the queue and a command
    typed by hand)."""
    return hold_lock(work / ".lock", f"Another run is already using the work folder '{work}'. "
                                     "Wait for it to finish (or stop it) first.")


def pngs(folder):
    """ffmpeg image-sequence pattern for a folder (a literal % must be written %%)."""
    return str(folder).replace("%", "%%") + os.sep + "%06d.png"


def gap_message(idx, missing, expected):
    """A chunk came out short: the source's video has a gap there (a damaged stretch of the rip)
    or the film/video handling dropped frames. Holding the last picture keeps everything after it
    in sync with the sound (a short chunk would move the rest of the movie early)."""
    return (f"  WARNING: chunk {idx}: {missing} of {expected} frames are missing in the source "
            "here (a gap or damaged stretch?); a still picture fills them so the sound stays in "
            "sync. If this shows up for every chunk, the film/video mode is wrong for this disc: "
            "check it with --analyze.")


def stab_index(path):
    """--stabilize: where each frame's record is in the camera-motion file of vidstabdetect, which
    vid.stab writes as text ("VID.STAB 1", one "Frame N (...)" line a frame: ffmpeg 6.x builds)
    or binary ("TRF1", a 24-byte header, then a frame: int32 number, int32 count, 26 bytes a
    measured field: newer builds). Returns (header, [(offset, length)] in frame order, binary)."""
    recs = []
    with open(path, "rb") as f:
        if f.read(3) == b"TRF":
            f.seek(0)
            head, pos, size = f.read(24), 24, os.fstat(f.fileno()).st_size
            while pos + 8 <= size:
                n = struct.unpack("<ii", f.read(8))[1]
                end = pos + 8 + 26 * n
                if n < 0 or end > size:
                    break                       # (a cut-short last record)
                recs.append((pos, end - pos))
                pos = f.seek(end)
            return head, recs, True
        f.seek(0)
        head, pos = b"", 0
        for line in f:
            if line.startswith(b"Frame "):
                if line.endswith(b"\n"):
                    recs.append((pos, len(line)))
            elif not recs:
                head += line
            pos += len(line)
    return head, recs, False


def write_stab_slice(path, index, first, end, dst):
    """The records of frames first..end-1 (of the whole video) as a motion file of their own,
    numbered from 1 again: vidstabtransform takes its file's frame 1 for the first frame it is
    given, here the first of this chunk's margin frames."""
    head, recs, binary = index
    out = [head]
    with open(path, "rb") as f:
        for k, (pos, size) in enumerate(recs[first:end], 1):
            f.seek(pos)
            rec = f.read(size)
            out.append(struct.pack("<i", k) + rec[4:] if binary else
                       b"Frame %d " % k + rec.split(b" ", 2)[2])
    Path(dst).write_bytes(b"".join(out))


# ---------------------------------------------------------------------------------------------
# Face restoration (--faces): GFPGAN 1.4 or CodeFormer (ONNX models) redraws the faces of the
# upscaled frames, which OpenCV's YuNet detector finds. It runs in a process of its own for each
# chunk ("python dvd_upscale.py --faces-worker ...", see faces_worker_main), so a crash in
# onnxruntime or the GPU driver can't take the run down, and its GPU memory is given back after
# every chunk. numpy, OpenCV and onnxruntime are imported only in there: without --faces nothing
# extra needs installing.

FACE_STRENGTH = 0.6     # --faces without a number: the restored face's share of the final picture
#                         (0-1, at most --ai-blend: the rest is a plain upscale, see Chunk.upscale)
FACE_MODEL = "gfpgan"   # --face-model default: gfpgan (GFPGAN 1.4) or codeformer
FACE_FIDELITY = 0.7     # CodeFormer only: 0 = its own idea of the face, 1 = closest to the frame
FACE_MODEL_FILES = {"gfpgan": "gfpgan_1.4.onnx", "codeformer": "codeformer.onnx"}
FACE_MODEL_NAMES = {"gfpgan": "GFPGAN 1.4", "codeformer": "CodeFormer"}
FACE_DETECTOR = "face_detection_yunet_2023mar.onnx"
FACE_URLS = {
    FACE_DETECTOR: "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
                   "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "gfpgan_1.4.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/"
                       "models-3.0.0/gfpgan_1.4.onnx",
    "codeformer.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/"
                       "models-3.0.0/codeformer.onnx"}
# the 5-point face layout of the 512x512 crops GFPGAN and CodeFormer were trained on (FFHQ):
# eye (image left), eye (image right), nose tip, mouth corner (image left), mouth corner (right)
FACE_TEMPLATE = ((192.98138, 239.94708), (318.90277, 240.1936), (256.63416, 314.01935),
                 (201.26117, 371.41043), (313.08905, 371.15118))
FACE_EYES = 125.92      # eye distance in the template
# eye distance in the source (DVD) frame, in its pixels - what the restorer has to work from:
# smaller than MIN skipped (too little to go on), full strength from FULL; from TAPER on tapered
# off, to none at the template's own size (the source has the detail itself). Measured against
# the real HD face (dvd_upscale_dev/evalfaces.py): at 14 px, 0.6 drew more detail than the real
# face had and moved it furthest from the person, 0.45 came closer to the real face than no
# restoration; from 17 px on 0.6 matched the real face's detail
FACE_MIN_EYES, FACE_FULL_EYES, FACE_TAPER_EYES = 7, 16, 90
# one face worker at a time, whichever GPU's chunk it is: two would compete with the upscalers
# for the GPU's memory
FACE_LOCK = threading.Lock()
FACE_PROCS = []         # the face worker running now (killed if this run ends: no orphan)


def face_read(path):
    """A frame as a BGR picture. (cv2.imread can't open a path with letters outside the Windows
    code page - a movie named "Amélie" - so the bytes are read by numpy and decoded.)"""
    import numpy as np
    import cv2
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"can't read the frame {path}")
    return img


def face_write(path, img):
    """A frame saved as PNG under a temporary name, then renamed: never a half-written frame
    under the real name. (Encoded in memory and written by numpy: any letters in the path.)"""
    import cv2
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError(f"can't encode the frame {path}")
    tmp = str(path) + ".tmp"
    buf.tofile(tmp)
    replace_file(tmp, str(path))


def face_soft_mask(size=512, blur=0.3):
    """1 inside, falling smoothly to 0 at the crop's edges (the restorer's hair and background
    stay out)."""
    import numpy as np
    import cv2
    amount = int(size * 0.5 * blur)
    area = max(amount // 2, 1)
    m = np.ones((size, size), np.float32)
    m[:area, :] = m[-area:, :] = m[:, :area] = m[:, -area:] = 0
    return cv2.GaussianBlur(m, (0, 0), amount * 0.25)


class FaceRestorer:
    """The detector and the restoration model, loaded once for a folder of frames."""

    def __init__(self, models, model=FACE_MODEL, fidelity=FACE_FIDELITY, providers=None):
        import numpy as np
        import onnxruntime as ort
        avail = ort.get_available_providers()
        # the GPU if onnxruntime has a way to it (onnxruntime-gpu: CUDA, onnxruntime-directml:
        # any GPU on Windows), else the processor
        chosen = [p for p in (providers or ["CUDAExecutionProvider", "DmlExecutionProvider",
                                            "CPUExecutionProvider"]) if p in avail]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3         # (its warnings: the provider used is reported anyway)
        if "DmlExecutionProvider" in chosen:
            # DirectML can't run with these on (onnxruntime's documentation)
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(os.path.join(models, FACE_MODEL_FILES[model]), opts,
                                         providers=chosen or ["CPUExecutionProvider"])
        self.provider = self.sess.get_providers()[0]
        self.input = self.sess.get_inputs()[0].name
        # CodeFormer's second input "weight" is the fidelity: a float64 scalar in this export,
        # taken as the model declares it
        self.feed = {}
        for i in self.sess.get_inputs()[1:]:
            if i.name == "weight":
                self.feed["weight"] = np.array(
                    fidelity, np.float64 if "double" in i.type else np.float32).reshape(
                    [1] * len(i.shape or []))
        self.det_path = os.path.join(models, FACE_DETECTOR)
        self.det = None
        self.mask = face_soft_mask()
        self.template = np.array(FACE_TEMPLATE, np.float32)

    def detect(self, img):
        """[(5x2 landmarks, score)] for the faces in a BGR frame."""
        import numpy as np
        import cv2
        h, w = img.shape[:2]
        s = min(1.0, 1280 / max(w, h))        # (detection on a frame of at most 1280 px)
        small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) \
            if s < 1 else img
        size = (small.shape[1], small.shape[0])
        if self.det is None:
            # top_k: the candidates kept before merging them, several per face (OpenCV's
            # default; 50 lost faces in a crowd, a different few in each frame)
            try:    # from memory: OpenCV opens a path with a narrow fstream, so letters outside
                # the Windows code page fail (as in face_read)
                self.det = cv2.FaceDetectorYN.create(
                    "onnx", np.fromfile(self.det_path, np.uint8), np.empty(0, np.uint8), size,
                    0.7, 0.3, 5000)
            except (cv2.error, TypeError):  # OpenCV 4.8 has only the path form (its
                # Python binding then refuses the arguments with a TypeError)
                self.det = cv2.FaceDetectorYN.create(self.det_path, "", size, 0.7, 0.3, 5000)
        self.det.setInputSize(size)
        _, faces = self.det.detect(small)
        if faces is None:
            return []
        return [(f[4:14].reshape(5, 2) / s, float(f[14])) for f in faces]

    def restore_crop(self, crop):
        """A 512x512 aligned face (BGR) through the model."""
        import numpy as np
        x = crop[:, :, ::-1].astype(np.float32) / 255.0
        x = np.ascontiguousarray(((x - 0.5) / 0.5).transpose(2, 0, 1)[None])
        y = self.sess.run(None, {self.input: x, **self.feed})[0][0]
        y = (np.clip(y, -1, 1) + 1) / 2
        return (y.transpose(1, 2, 0) * 255).round().astype(np.uint8)[:, :, ::-1]

    def paste(self, img, pts, strength, ai_blend=1.0):
        """Restore the face at landmarks pts and blend it into img (in place), with a soft mask,
        over only the face's part of the frame. ai_blend: the share of these frames in the
        final picture (--ai-blend; the rest is a plain upscale without the restored face), made
        up for here so that strength is the face's share of the final picture (at most
        ai_blend)."""
        import numpy as np
        import cv2
        M = cv2.estimateAffinePartial2D(pts.astype(np.float32), self.template,
                                        method=cv2.LMEDS)[0]
        if M is None:
            return
        crop = cv2.warpAffine(img, M, (512, 512), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(135, 133, 132))
        out = self.restore_crop(crop)
        IM = cv2.invertAffineTransform(M)
        h, w = img.shape[:2]
        corners = np.array([[0, 0, 1], [512, 0, 1], [0, 512, 1], [512, 512, 1]],
                           np.float32) @ IM.T
        x0, y0 = np.floor(corners.min(0)).astype(int)
        x1, y1 = np.ceil(corners.max(0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return
        IMs = IM.copy()
        IMs[:, 2] -= (x0, y0)
        size = (x1 - x0, y1 - y0)
        face = cv2.warpAffine(out, IMs, size, flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
        m = np.minimum(1.0, cv2.warpAffine(self.mask, IMs, size, flags=cv2.INTER_LINEAR)
                       [:, :, None] * (strength / ai_blend))
        region = img[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] = np.clip(region * (1 - m) + face.astype(np.float32) * m + 0.5,
                                    0, 255).astype(np.uint8)


def face_eye_dist(pts):
    import numpy as np
    return float(np.linalg.norm(pts[1] - pts[0]))


def face_tracks(dets):
    """Link per-frame detections into tracks: [[(frame, landmarks)], ...]. A face continues a
    track when its eye centre is within one eye distance of where the track was 1-2 frames ago."""
    import numpy as np
    tracks, live = [], []                     # live: indexes into tracks
    for f, faces in enumerate(dets):
        used, still = set(), []
        for t in live:
            lf, lp = tracks[t][-1]
            if f - lf > 2:
                continue
            best, bd = None, None
            for k, (pts, _) in enumerate(faces):
                if k in used:
                    continue
                d = float(np.linalg.norm(pts[:2].mean(0) - lp[:2].mean(0)))
                if d < face_eye_dist(lp) and (bd is None or d < bd):
                    best, bd = k, d
            if best is not None:
                used.add(best)
                tracks[t].append((f, faces[best][0]))
            still.append(t)
        for k, (pts, _) in enumerate(faces):
            if k not in used:
                tracks.append([(f, pts)])
                still.append(len(tracks) - 1)
        live = [t for t in still if f - tracks[t][-1][0] <= 2]
    return tracks


def face_plan(dets, n, strength, scale=2, smooth=2, fade=4, min_len=5, edge_min=3, same=None):
    """Per frame: [(landmarks, strength)], from steadied tracks (steady alignment = less
    flicker). scale: the frames' size over the source's. A track of at least edge_min
    detections that reaches the first or last frame goes on in the next/previous chunk: no fade
    there (else a pulse at every chunk seam). One missed detection on that edge frame counts
    as reaching it (the face is held there) if same(edge frame, last detected frame, its
    landmarks) says the face box still shows the same (not a cut, or a face that has left).
    Its landmarks are steadied by a straight line through the ones around it, the window slid
    inward at an open edge: a moving face is aligned where it is, not lagging or leading at
    the seam."""
    import numpy as np
    out = [[] for _ in range(n)]
    for tr in face_tracks(dets):
        frames = [f for f, _ in tr]
        pts = {f: p for f, p in tr}
        first, last = frames[0], frames[-1]
        if len(tr) >= edge_min:
            if first == 1 and (same is None or same(0, 1, pts[1])):
                first = 0
            if last == n - 2 and (same is None or same(n - 1, n - 2, pts[n - 2])):
                last = n - 1
        open_start = len(tr) >= edge_min and first == 0
        open_end = len(tr) >= edge_min and last == n - 1
        if len(tr) < min_len and not (open_start or open_end):
            continue                           # a face seen for a moment: likely a false one
        # fill gaps of a frame or two (a missed detection) by interpolation
        for f in range(frames[0], frames[-1] + 1):
            if f not in pts:
                a = max(g for g in frames if g < f)
                b = min(g for g in frames if g > f)
                pts[f] = pts[a] + (pts[b] - pts[a]) * (f - a) / (b - a)
        for f in range(first, last + 1):
            lo, hi = f - smooth, f + smooth
            if open_end and hi > last:
                lo, hi = lo - (hi - last), last
            if open_start and lo < first:
                lo, hi = first, min(last, hi + (first - lo))
            gs = [g for g in range(lo, hi + 1) if g in pts]
            t = np.array(gs, float) - f
            Y = np.stack([pts[g] for g in gs])
            if len(gs) > 1 and t.std() > 0:     # (centred window: just the mean)
                slope = (((t - t.mean())[:, None, None] * (Y - Y.mean(0))).sum(0)
                         / ((t - t.mean()) ** 2).sum())
                p = Y.mean(0) - slope * t.mean()
            else:
                p = Y.mean(0)
            d = face_eye_dist(p) / scale
            if d < FACE_MIN_EYES:
                continue
            size = min(1.0, (d - FACE_MIN_EYES) / (FACE_FULL_EYES - FACE_MIN_EYES))
            size *= 1.0 if d <= FACE_TAPER_EYES else max(
                0.0, 1 - (d - FACE_TAPER_EYES) / (FACE_EYES - FACE_TAPER_EYES))
            edge = min(1.0, 1.0 if open_start else (f - first + 1) / fade,
                       1.0 if open_end else (last - f + 1) / fade)
            s = strength * size * edge
            if s > 0.01:
                out[f].append((p, s))
    return out


def restore_faces(folder, models, model=FACE_MODEL, strength=FACE_STRENGTH,
                  fidelity=FACE_FIDELITY, scale=2, dest=None, providers=None, progress=None,
                  ai_blend=1.0):
    """Every PNG in folder, in name order (= frame order), with its faces restored. Two passes:
    find every frame's faces, link them into tracks across frames and steady their landmarks,
    then restore and blend each face back, fading in and out where a track starts and ends.
    The frames with faces are written into dest under the same name (dest None: over the
    originals); the others aren't written at all. progress(step, done, of), step "detect" or
    "restore" (also after each face). ai_blend: see FaceRestorer.paste. Returns (faces
    restored, frames changed, the onnxruntime provider used)."""
    import numpy as np
    import cv2
    r = FaceRestorer(models, model, fidelity, providers)
    names = sorted(x for x in os.listdir(folder) if x.endswith(".png"))
    dets = []
    for i, name in enumerate(names):
        dets.append(r.detect(face_read(os.path.join(folder, name))))
        if progress:
            progress("detect", i + 1, len(names))

    def same(f, g, pts):
        """Frame f's face box (around landmarks pts, found in frame g) shows what frame g's
        does, compared as 8x8 averages: a moving face, up to about a tenth of its eye distance
        a frame, is the same; a cut or a face that has left the box is not (measured: under 2
        from frame to frame, 35 or more at a cut)."""
        c, rad = pts.mean(0), 1.2 * face_eye_dist(pts)
        small = []
        for k in (f, g):
            img = face_read(os.path.join(folder, names[k]))
            x0, y0 = max(0, int(c[0] - rad)), max(0, int(c[1] - rad))
            x1, y1 = min(img.shape[1], int(c[0] + rad) + 1), min(img.shape[0], int(c[1] + rad) + 1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                return False
            small.append(cv2.resize(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (8, 8),
                                    interpolation=cv2.INTER_AREA).astype(np.float32))
        return float(np.abs(small[0] - small[1]).mean()) < 12
    todo = face_plan(dets, len(names), strength, scale, same=same)
    faces = changed = 0
    for i, (name, found) in enumerate(zip(names, todo)):
        if found:
            img = face_read(os.path.join(folder, name))
            for pts, s in found:
                r.paste(img, pts, s, ai_blend)
                # after each face too (the frames done so far): a frame with a crowd can take
                # many minutes on the processor, which mustn't be taken for a hang
                if progress:
                    progress("restore", i, len(names))
            face_write(os.path.join(dest or folder, name), img)
            faces, changed = faces + len(found), changed + 1
        if progress:
            progress("restore", i + 1, len(names))
    return faces, changed, r.provider


def face_dists(*names):
    """Which of these pip packages are installed."""
    from importlib import metadata
    got = []
    for n in names:
        try:
            metadata.version(n)
            got.append(n)
        except metadata.PackageNotFoundError:
            pass
    return got


ORT_DISTS = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml")
CV_DISTS = ("opencv-python-headless", "opencv-python", "opencv-contrib-python",
            "opencv-contrib-python-headless")


NCNN_INSTALL = "python -m pip install --no-deps ncnn numpy"
NCNN_VERSION = "1.0.20260526"           # (the one tested)


def ncnn_install_hint():
    """How to install the current ncnn for the Python running this script: setup.bat for the
    one-folder install (its private Python isn't on the PATH), else pip for this Python. --no-deps:
    ncnn itself needs only numpy; its other listed packages include opencv-python, which can
    clash with the opencv-python-headless that --faces uses."""
    here = Path(__file__).resolve().parent
    if Path(sys.executable).resolve().parent == here / "python":
        return "run setup.bat again"
    return f"{pip_cmd()} install --no-deps ncnn=={NCNN_VERSION} numpy"


ESRGAN_DEFAULT = "realesrgan-ncnn-vulkan"
HELPER_MIN = 0.2        # a second GPU this fraction as fast as the first is kept as a helper
DEFAULT_GPU = "0"       # the GPU used without --gpu, until --ncnn-bench-gpu saves a faster one


def ncnn_available():
    """The current ncnn from pip is installed (it isn't loaded here: the GPU is used by the
    upscaler processes only)"""
    import importlib.util
    return all(importlib.util.find_spec(m) for m in ("ncnn", "numpy"))


def write_png(path, rgb):
    """An 8-bit RGB PNG (rows "Up"-filtered, zlib level 1), written to a temporary name next to
    the folder and then renamed: a frame in the folder is always whole."""
    import numpy as np
    h, w, _ = rgb.shape
    rows = rgb.reshape(h, w * 3)
    raw = np.empty((h, w * 3 + 1), np.uint8)
    raw[:, 0] = 2                                           # (filter type Up)
    raw[0, 1:] = rows[0]
    np.subtract(rows[1:], rows[:-1], out=raw[1:, 1:])       # (uint8: modulo 256, as PNG wants)

    import zlib

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 1)) + chunk(b"IEND", b""))
    tmp = Path(path).parent.parent / f".{Path(path).name}.{os.getpid()}.part"
    try:
        tmp.write_bytes(data)
        replace_file(tmp, path)         # (retried: a virus scanner may hold the new file)
    except OSError as e:
        raise OSError(f"encode image {Path(path).name} failed ({e})") from e


# ncnn options that can be switched off when the GPU faults with the current ncnn, mildest first
# (see ncnn_stress_main, which tries them and saves the first that survives in ncnn_opts.json)
NCNN_OPT_SETS = {
    "base": {},
    "nosubgroup": {"use_subgroup_ops": False},
    "notensor": {"use_tensor_storage": False},
    "nowinograd": {"use_winograd_convolution": False, "use_winograd23_convolution": False,
                   "use_winograd43_convolution": False, "use_winograd63_convolution": False},
    "fp32": {"use_fp16_packed": False, "use_fp16_storage": False, "use_int8_storage": False,
             "use_bf16_packed": False, "use_bf16_storage": False},
}
NCNN_OPT_SETS["safe"] = {k: v for d in NCNN_OPT_SETS.values() for k, v in d.items()}
# one winograd variant at a time (the three of ncnn's 3x3 convolution shaders; all off is
# "nowinograd", the set that survives): winograd is the fast way to do 3x3 convolutions, and
# the GPU fault may come from only one of them (see ncnn_winograd_main)
NCNN_OPT_SETS["w23"] = {"use_winograd43_convolution": False, "use_winograd63_convolution": False}
NCNN_OPT_SETS["w43"] = {"use_winograd23_convolution": False, "use_winograd63_convolution": False}
NCNN_OPT_SETS["w63"] = {"use_winograd23_convolution": False, "use_winograd43_convolution": False}
NCNN_OPT_SETS["nowinograd_fp16"] = {**NCNN_OPT_SETS["nowinograd"], "use_fp16_arithmetic": True}


def ncnn_opts_file():
    return Path(os.environ.get("DVD_UPSCALE_NCNN_OPTS_FILE")
                or Path(__file__).resolve().parent / "ncnn_opts.json")


def ncnn_saved():
    try:
        d = json.loads(ncnn_opts_file().read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def ncnn_opt_set_name():
    """The option set the upscaler worker uses: DVD_UPSCALE_NCNN_OPTS, else what the self-test
    saved (ncnn_opts.json next to the script), else "base"."""
    name = os.environ.get("DVD_UPSCALE_NCNN_OPTS")
    if not name:
        name = ncnn_saved().get("set")
    return name if name in NCNN_OPT_SETS else "base"


def ncnn_stress_main(argv):
    """python dvd_upscale.py --ncnn-stress [--gpu N] [--frames 300]: the current ncnn upscaling
    test frames with each option set of NCNN_OPT_SETS in turn (one process each, the real
    worker). The GPU of NVIDIA drivers that fault on some ncnn code paths is reset within a few
    hundred frames: the first set that gets through them all is saved in ncnn_opts.json, and
    every later run uses it."""
    p = argparse.ArgumentParser(prog="dvd_upscale.py --ncnn-stress")
    p.add_argument("--gpu")
    p.add_argument("--frames", type=int, default=300)
    w = p.parse_args(argv)
    here = Path(__file__).resolve().parent
    exe = shutil.which(ESRGAN_DEFAULT) or next((str(f) for f in here.glob(ESRGAN_DEFAULT + "*")), None)
    models = Path(exe).resolve().parent / "models" if exe else here / "models"
    with tempfile.TemporaryDirectory(prefix="ncnn_stress_") as d:
        d = Path(d)
        (d / "in").mkdir()
        print(f"Making {w.frames} test frames (720x480)...", flush=True)
        run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
             f"testsrc2=s=720x480:r=24:d={w.frames / 24:.3f}", "-frames:v", str(w.frames),
             str(d / "in" / "%06d.png")])
        results = []
        for name in NCNN_OPT_SETS:
            if name in ("nowinograd_fp16", "w23", "w43", "w63"):    # (speed trials, see ncnn_bench_main
                continue                                        # and ncnn_winograd_main)
            out = d / ("out_" + name)
            out.mkdir()
            cmd = [sys.executable, Path(__file__).resolve(), "--ncnn-upscaler", "-i", d / "in",
                   "-o", out, "-n", "realesrgan-x2plus", "-s", "2", "-f", "png", "-j", "1:1:1",
                   "-m", models]
            if w.gpu is not None:
                cmd += ["-g", w.gpu]
            t0 = time.time()
            r = subprocess.run([str(c) for c in cmd], capture_output=True,
                               stdin=subprocess.DEVNULL, env={**os.environ,
                                                              "DVD_UPSCALE_NCNN_OPTS": name})
            text = (r.stdout + r.stderr).decode("utf-8", "replace")
            n = len(list(out.glob("*.png")))
            ok = r.returncode == 0 and n == w.frames and not GPU_ERRORS.search(text)
            took = time.time() - t0
            print(f"  {name:11s} {'OK    ' if ok else 'FAILED'} {n}/{w.frames} frames, "
                  f"{n / took:.1f} frames/s" + ("" if ok else f" (exit code {r.returncode})"),
                  flush=True)
            if not ok:
                bad = [x.strip() for x in text.splitlines() if x.startswith("ncnn")
                       or GPU_ERRORS.search(x)]
                for x in bad[-3:]:
                    print("      " + x[:150], flush=True)
            results.append((name, ok, n / took))
            shutil.rmtree(out, ignore_errors=True)
            if ok:
                break
            time.sleep(15)      # (the driver recovers from the reset)
    good = [x for x in results if x[1]]
    if not good:
        print("No option set got through (see the lines above). If they say the GPU was reset: "
              "this driver faults on the current ncnn too, try an older one (before 570).")
        return 1
    ncnn_opts_file().write_text(json.dumps({**ncnn_saved(), "set": good[0][0]}),
                                  encoding="utf-8")
    print(f"Saved '{good[0][0]}' in {ncnn_opts_file().name}: the upscaler uses it from now on.")
    return 0


def ncnn_bench_main(argv):
    """python dvd_upscale.py --ncnn-bench [MOVIE] [--gpu N] [--frames 90]: the speed-ups that
    leave the picture as it is, tried on frames of the movie: whole frames instead of 200-pixel
    tiles (a fifth of the work is the tiles' overlaps; and no seams) and fp16 arithmetic (the
    GPU's fast path). Each is run through the real worker, its speed taken from the times its
    frames were written, and its frames compared with the reference (the current settings): a
    set must be 45 dB or closer (40+ is invisible; fp16 against fp32 rounding is about 55) and
    free of GPU errors. The fastest that passes is saved in ncnn_opts.json (set and tile)."""
    p = argparse.ArgumentParser(prog="dvd_upscale.py --ncnn-bench")
    p.add_argument("movie", nargs="?")
    p.add_argument("--gpu")
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--min-db", type=float, default=45.0)
    w = p.parse_args(argv)
    here = Path(__file__).resolve().parent
    exe = shutil.which(ESRGAN_DEFAULT) or next((str(f) for f in here.glob(ESRGAN_DEFAULT + "*")), None)
    models = Path(exe).resolve().parent / "models" if exe else here / "models"
    base_set = ncnn_saved().get("set") or "nowinograd"
    if base_set not in NCNN_OPT_SETS or base_set == "nowinograd_fp16":
        base_set = "nowinograd"
    trials = [(base_set, 200), (base_set, 1024), ("nowinograd_fp16", 200),
              ("nowinograd_fp16", 1024)]
    with tempfile.TemporaryDirectory(prefix="ncnn_bench_") as d:
        d = Path(d)
        (d / "in").mkdir()
        if w.movie:
            print(f"Taking {w.frames} frames from {w.movie}...", flush=True)
            rc, txt = capture(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                               "-of", "csv=p=0", w.movie])
            try:
                at = max(0.0, float(txt.strip()) * 0.4)         # (somewhere inside the film)
            except ValueError:
                at = 0.0
            run(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.1f}", "-i", w.movie, "-map",
                 "0:v:0", "-vf", "scale=iw*sar:ih,setsar=1,scale=720:480", "-frames:v",
                 str(w.frames), str(d / "in" / "%06d.png")])
        else:
            run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                 f"testsrc2=s=720x480:r=24:d={w.frames / 24:.3f}", "-frames:v", str(w.frames),
                 str(d / "in" / "%06d.png")])
        n_in = len(list((d / "in").glob("*.png")))
        ref, results = None, []
        for k, (name, tile) in enumerate(trials):
            out = d / f"out{k}"
            out.mkdir()
            cmd = [sys.executable, Path(__file__).resolve(), "--ncnn-upscaler", "-i", d / "in",
                   "-o", out, "-n", "realesrgan-x2plus", "-s", "2", "-f", "png", "-j", "1:1:1",
                   "-t", tile, "-m", models]
            if w.gpu is not None:
                cmd += ["-g", w.gpu]
            r = subprocess.run([str(c) for c in cmd], capture_output=True,
                               stdin=subprocess.DEVNULL,
                               env={**os.environ, "DVD_UPSCALE_NCNN_OPTS": name})
            text = (r.stdout + r.stderr).decode("utf-8", "replace")
            files = sorted(out.glob("*.png"))
            ok = r.returncode == 0 and len(files) == n_in and not GPU_ERRORS.search(text)
            label = f"{name} / {'whole frame' if tile > 720 else str(tile) + '-pixel tiles'}"
            fps = db = None
            if ok and len(files) > 3:
                times = [f.stat().st_mtime for f in files]
                fps = (len(times) - 1) / max(1e-6, max(times) - min(times))
                if ref is None:
                    ref, db = out, float("inf")
                else:
                    rc, rep = capture(["ffmpeg", "-v", "info", "-i", str(out / "%06d.png"),
                                       "-i", str(ref / "%06d.png"), "-lavfi", "psnr",
                                       "-f", "null", "-"])
                    m = re.search(r"PSNR .*?average:([0-9.]+|inf)", rep)
                    db = float(m.group(1)) if m else 0.0
            if ok and db is not None and db >= w.min_db:
                print(f"  {label:42s} {fps:4.2f} frames/s  "
                      + ("(reference)" if db == float("inf") else f"{db:.1f} dB from the reference"),
                      flush=True)
                results.append((fps, name, tile))
            else:
                why = ("GPU error" if GPU_ERRORS.search(text) or r.returncode else
                       f"picture differs ({db:.1f} dB)" if db is not None else "no result")
                print(f"  {label:42s} not used: {why}", flush=True)
            if ref is not out:
                shutil.rmtree(out, ignore_errors=True)
            time.sleep(10 if ok else 15)
    if not results:
        print("Nothing ran cleanly: the saved settings are unchanged.")
        return 1
    fps, name, tile = max(results)
    ncnn_opts_file().write_text(json.dumps({**ncnn_saved(), "set": name, "tile": tile}),
                                  encoding="utf-8")
    print(f"Saved the fastest ({name}, {'whole frames' if tile > 720 else str(tile) + '-pixel tiles'}, "
          f"{fps:.2f} frames/s) in {ncnn_opts_file().name}.")
    return 0


def ncnn_bench_gpu_main(argv):
    """python dvd_upscale.py --ncnn-bench-gpu [MOVIE] [--gpus 0,1,2] [--frames 40]: the same frames
    through the real worker on each GPU (the saved settings); the faster one is saved in
    ncnn_opts.json as the GPU used when --gpu isn't given (else DEFAULT_GPU). Also says what both together would
    be (the movie's helper-GPU mode: --gpu 0,1) when the slower GPU is worth having."""
    p = argparse.ArgumentParser(prog="dvd_upscale.py --ncnn-bench-gpu")
    p.add_argument("movie", nargs="?")
    p.add_argument("--gpus", default=None,
                   help="the GPU numbers to time (default: every Vulkan GPU except integrated "
                        "graphics, which can fail the test and bring up the driver's bug-report "
                        "window)")
    p.add_argument("--frames", type=int, default=40)
    w = p.parse_args(argv)
    here = Path(__file__).resolve().parent
    exe = shutil.which(ESRGAN_DEFAULT) or next((str(f) for f in here.glob(ESRGAN_DEFAULT + "*")), None)
    models = Path(exe).resolve().parent / "models" if exe else here / "models"
    saved = ncnn_saved()
    name = saved.get("set") if saved.get("set") in NCNN_OPT_SETS else "nowinograd"
    tile = int(saved.get("tile") or 0)
    if w.gpus is None:
        found = probe_vulkan_gpus()
        gpus = [str(i) for i, name in found if gpu_class(name) != "low"] \
            or [str(i) for i, _ in found] or ["0", "1"]
        print("Timing: " + ", ".join(f"GPU {i} ({dict(found).get(int(i), '?')})" for i in gpus))
    else:
        gpus = [g.strip() for g in w.gpus.split(",") if g.strip().isdigit()]
    results = []
    with tempfile.TemporaryDirectory(prefix="ncnn_gpu_") as d:
        d = Path(d)
        (d / "in").mkdir()
        if w.movie:
            rc, txt = capture(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                               "-of", "csv=p=0", w.movie])
            try:
                at = max(0.0, float(txt.strip()) * 0.4)
            except ValueError:
                at = 0.0
            run(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.1f}", "-i", w.movie, "-map",
                 "0:v:0", "-vf", "scale=iw*sar:ih,setsar=1,scale=720:480", "-frames:v",
                 str(w.frames), str(d / "in" / "%06d.png")])
        else:
            run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                 f"testsrc2=s=720x480:r=24:d={w.frames / 24:.3f}", "-frames:v", str(w.frames),
                 str(d / "in" / "%06d.png")])
        n_in = len(list((d / "in").glob("*.png")))
        for g in gpus:
            out = d / f"out{g}"
            out.mkdir()
            cmd = [sys.executable, Path(__file__).resolve(), "--ncnn-upscaler", "-i", d / "in",
                   "-o", out, "-n", "realesrgan-x2plus", "-s", "2", "-f", "png", "-j", "1:1:1",
                   "-g", g, "-m", models] + (["-t", tile] if tile else [])
            r = subprocess.run([str(c) for c in cmd], capture_output=True,
                               stdin=subprocess.DEVNULL,
                               env={**os.environ, "DVD_UPSCALE_NCNN_OPTS": name})
            text = (r.stdout + r.stderr).decode("utf-8", "replace")
            files = sorted(out.glob("*.png"))
            gname = re.search(r"GPU %s (.+?), realesrgan" % g, text)
            gname = gname.group(1) if gname else "?"
            if r.returncode == 0 and len(files) == n_in and n_in > 3 \
                    and not GPU_ERRORS.search(text):
                times = [f.stat().st_mtime for f in files]
                fps = (len(times) - 1) / max(1e-6, max(times) - min(times))
                print(f"  GPU {g} ({gname}): {fps:.2f} frames/s", flush=True)
                results.append((fps, g))
            else:
                print(f"  GPU {g} ({gname}): didn't run cleanly (exit code {r.returncode})",
                      flush=True)
            time.sleep(10)
    if not results:
        print("No GPU ran cleanly: nothing saved.")
        return 1
    results.sort(reverse=True)
    fps, g = results[0]
    saved["gpu"] = int(g)
    saved.pop("helper", None)
    msg = (f"GPU {g} is the faster one: it is now the default (saved in {ncnn_opts_file().name}; "
           "--gpu N picks another, --gpu 0,1 uses both).")
    if len(results) > 1:
        slow, h = results[1]        # (the next fastest: the one worth having as the helper)
        if slow / fps >= HELPER_MIN:
            saved["helper"] = int(h)
            msg += (f"\nGPU {h} is {slow / fps * 100:.0f}% as fast: worth having as a helper, so "
                    f"it works alongside it from now on (whole chunks; --gpu {g} alone turns "
                    "that off).")
        else:
            msg += (f"\nThe other GPU is {slow / fps * 100:.0f}% as fast: not worth adding as a "
                    "helper (it would only add heat).")
    ncnn_opts_file().write_text(json.dumps(saved), encoding="utf-8")
    print(msg)
    return 0


def ncnn_auto_main(argv):
    """python dvd_upscale.py --ncnn-auto [MOVIE] [--gpu N] [--skip-gpu-test] [--no-test-run]:
    everything to set the GPU up and see how fast it is, in one go: the old saved settings
    (gpu_steps.json, ncnn_opts.json) cleared; --ncnn-stress (the options the GPU survives);
    --ncnn-bench (whole frames / fp16, the fastest with the same picture); --ncnn-bench-gpu
    (which GPU is faster, saved as the default); --ncnn-models (every model's picture, against
    the original); then a --test 60 run of the movie while the
    GPU is watched (nvidia-smi), and a summary with the time the whole movie will take."""
    p = argparse.ArgumentParser(prog="dvd_upscale.py --ncnn-auto")
    p.add_argument("movie", nargs="?")
    p.add_argument("--gpu", default=DEFAULT_GPU)
    p.add_argument("--skip-gpu-test", action="store_true")
    p.add_argument("--no-test-run", action="store_true")
    p.add_argument("--skip-model-test", action="store_true")
    p.add_argument("--winograd", action="store_true",
                   help="also try the winograd variants (on an RTX 3060 laptop none was faster)")
    w = p.parse_args(argv)
    movie = [w.movie] if w.movie else []
    summary = []

    def step(n, text):
        print(f"\n=== {n}/6  {text} ===", flush=True)

    step(1, "clearing the old saved settings")
    for f in (gpu_step_file(), ncnn_opts_file()):
        try:
            f.unlink()
            print(f"  deleted {f.name}")
        except OSError:
            print(f"  no {f.name} (fine)")
    step(2, "which GPU is faster (--ncnn-bench-gpu)")
    if w.skip_gpu_test:
        print("  skipped")
    elif ncnn_bench_gpu_main(movie) == 0:
        summary.append(f"default GPU: {ncnn_saved().get('gpu')}")
    else:
        summary.append(f"default GPU: {w.gpu} (the GPU test didn't finish)")
    # (the options are then found on the GPU the movie will run on)
    gpu = str(ncnn_saved().get("gpu", w.gpu))
    step(3, f"finding the options GPU {gpu} survives (--ncnn-stress)")
    if ncnn_stress_main(["--gpu", gpu]) != 0:
        print("\nStopped: no setting survived (see above). Nothing more was tried.")
        return 1
    summary.append(f"settings the GPU survives: {ncnn_saved().get('set')}")
    step(4, "the fastest settings with the same picture (--ncnn-bench)")
    if ncnn_bench_main([*movie, "--gpu", gpu]) == 0:
        saved = ncnn_saved()
        tile = int(saved.get("tile") or 0)
        summary.append(f"fastest safe: {saved.get('set')}, "
                       + ("whole frames" if tile > 720 else f"{tile or 200}-pixel tiles"))
    else:
        summary.append("the speed-up test didn't finish: the settings of step 3 are kept")
    if w.winograd and ncnn_winograd_main([*movie, "--gpu", gpu]) == 0:
        summary.append(f"winograd: set '{ncnn_saved().get('set')}'")
    step(5, "every model on the same frames (--ncnn-models)")
    if w.skip_model_test or not w.movie:
        print("  skipped" + ("" if w.skip_model_test else " (give a movie to test the models)"))
    elif ncnn_models_main([w.movie, "--gpu", str(ncnn_saved().get("gpu", w.gpu)),
                           "--save"]) == 0:
        summary.append("models compared: see the table above and model_compare.png")
    else:
        summary.append("the model comparison didn't finish")
    step(6, "timing a 60-second test run, GPU watched")
    if w.no_test_run or not w.movie:
        print("  skipped" + ("" if w.no_test_run else " (give a movie to time it)"))
    else:
        # (a finished test run of this movie would be resumed, not timed)
        shutil.rmtree(Path(w.movie).with_name(Path(w.movie).stem + "_work_test"),
                      ignore_errors=True)
        util, temps, watts, clocks = [], [], [], []
        stop = threading.Event()

        def watch():
            while not stop.wait(2):
                try:
                    r = subprocess.run(
                        ["nvidia-smi",
                         "--query-gpu=utilization.gpu,temperature.gpu,power.draw,clocks.sm",
                         "--format=csv,noheader,nounits"], capture_output=True, text=True,
                        timeout=10)
                    v = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
                    util.append(v[0]), temps.append(v[1]), watts.append(v[2]), clocks.append(v[3])
                except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                    pass
        threading.Thread(target=watch, daemon=True).start()
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), w.movie,
                                 "--test", "60"], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        out = b""
        while True:
            chunk = os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                break
            out += chunk
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        proc.wait()
        stop.set()
        text = out.decode("utf-8", "replace")
        secs = re.findall(r"(\d+)s/chunk", text)
        bad = len(GPU_ERRORS.findall(text))
        if proc.returncode or not secs:
            summary.append(f"test run: didn't finish (exit code {proc.returncode})")
        else:
            spc = max(1, int(secs[-1]))
            rc, dur = capture(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                               "-of", "csv=p=0", w.movie])
            # (the run's own frames/s; a chunk is 480 frames for live action, 1440 for anime)
            rate = re.findall(r"Upscaled \d+ frames in .*?\(([\d.]+) frames/s\)", text)
            fps = float(rate[-1]) if rate else 480 / spc
            line = f"test run: {spc} s per chunk ({fps:.1f} frames/s)"
            try:
                line += (f"; this movie: about "
                         f"{float(dur.strip()) * 24000 / 1001 / fps / 3600:.1f} hours")
            except (ValueError, ZeroDivisionError):
                pass
            summary.append(line)
        summary.append(f"GPU errors in the test run: {bad}" if bad else "no GPU errors in the test run")
        if util:
            summary.append(f"GPU while upscaling: {sum(util) / len(util):.0f}% busy, "
                           f"up to {max(temps):.0f} C, {sum(watts) / len(watts):.0f} W, "
                           f"{sum(clocks) / len(clocks):.0f} MHz"
                           + ("  (well under 90% busy: something else is the limit)"
                              if sum(util) / len(util) < 80 else ""))
    print("\n=== Summary ===")
    for x in summary:
        print("  " + x)
    print("Run the movie with: python dvd_upscale.py <movie> <output>   (these settings are used "
          "automatically)")
    return 0


def ncnn_models_main(argv):
    """python dvd_upscale.py --ncnn-models MOVIE [--gpu N] [--frames 6]: every model in the
    upscaler's models folder, tried on the same frames of the movie, against the truth.
    The test: each frame is shrunk to half size, every model upscales it back, and the result is
    compared with the original frame (SSIM and PSNR; a plain Lanczos resize is the baseline: a
    model has to beat it). The score is fidelity to the original: it rewards a faithful picture
    and punishes invented detail, so it ranks models by how true they are, not how sharp, and
    smooth models can score well. So a picture is written too, model_compare.png (next to the
    script): the original and each model's result, the same part of the frame side by side, to
    judge with your own eyes. Speed is relative to realesrgan-x2plus."""
    p = argparse.ArgumentParser(prog="dvd_upscale.py --ncnn-models")
    p.add_argument("movie")
    p.add_argument("--gpu", default=None)
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--save", action="store_true",
                   help="save the best model for live action / 3D animation in ncnn_opts.json "
                        "(used only by an upscale run with --best-quality; 10 frames at least)")
    w = p.parse_args(argv)
    if w.save:
        w.frames = max(w.frames, 10)
    here = Path(__file__).resolve().parent
    exe = shutil.which(ESRGAN_DEFAULT) or next((str(f) for f in here.glob(ESRGAN_DEFAULT + "*")), None)
    models = Path(exe).resolve().parent / "models" if exe else here / "models"
    found = {}
    for f in sorted(models.glob("*.param")):
        m = re.fullmatch(r"(realesr-animevideov3)-x([234])", f.stem)
        if m:
            if m[2] == "2":                     # (its x3 and x4 files: other scales of the same net)
                found[m[1]] = 2
        elif (models / (f.stem + ".bin")).exists():
            m = re.search(r"x([24])", f.stem)
            if m:
                found[f.stem] = int(m[1])
    if not found:
        print(f"No models found in {models}")
        return 1
    saved = ncnn_saved()
    name = saved.get("set") if saved.get("set") in NCNN_OPT_SETS else "nowinograd"
    tile = int(saved.get("tile") or 0)
    rc, dur = capture(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                       "csv=p=0", w.movie])
    try:
        total = float(dur.strip())
    except ValueError:
        print(f"Can't read {w.movie}")
        return 1
    with tempfile.TemporaryDirectory(prefix="ncnn_models_") as d:
        d = Path(d)
        for sub in ("hr", "lr"):
            (d / sub).mkdir()
        print(f"Taking {w.frames} frames from {w.movie}...", flush=True)
        for i in range(w.frames):
            at = total * (0.15 + 0.7 * i / max(1, w.frames - 1))
            run(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.1f}", "-i", w.movie, "-map",
                 "0:v:0", "-vf", "scale=iw*sar:ih,setsar=1,scale=720:480", "-frames:v", "1",
                 str(d / "hr" / f"{i + 1:06d}.png")])
        run(["ffmpeg", "-y", "-v", "error", "-i", str(d / "hr" / "%06d.png"), "-vf",
             "scale=360:240:flags=area", str(d / "lr" / "%06d.png")])
        n_in = len(list((d / "lr").glob("*.png")))

        def score(folder):
            rc, rep = capture(["ffmpeg", "-v", "info", "-i", str(folder / "%06d.png"), "-i",
                               str(d / "hr" / "%06d.png"), "-lavfi", "ssim", "-f", "null", "-"])
            ssim = re.search(r"All:([0-9.]+)", rep)
            rc, rep = capture(["ffmpeg", "-v", "info", "-i", str(folder / "%06d.png"), "-i",
                               str(d / "hr" / "%06d.png"), "-lavfi", "psnr", "-f", "null", "-"])
            psnr = re.search(r"average:([0-9.]+)", rep)
            return (float(ssim[1]) if ssim else 0.0), (float(psnr[1]) if psnr else 0.0)
        plain = d / "plain"
        plain.mkdir()
        run(["ffmpeg", "-y", "-v", "error", "-i", str(d / "lr" / "%06d.png"), "-vf",
             "scale=720:480:flags=lanczos", str(plain / "%06d.png")])
        rows = [("plain resize (baseline)", *score(plain), None, plain)]
        by_model = {}
        for mname, k in found.items():
            out, fit = d / f"out_{mname}", d / f"fit_{mname}"
            out.mkdir()
            fit.mkdir()
            cmd = [sys.executable, Path(__file__).resolve(), "--ncnn-upscaler", "-i", d / "lr",
                   "-o", out, "-n", mname, "-s", k, "-f", "png", "-j", "1:1:1", "-m", models]
            if tile:
                cmd += ["-t", tile]
            if w.gpu is not None:
                cmd += ["-g", w.gpu]
            r = subprocess.run([str(c) for c in cmd], capture_output=True,
                               stdin=subprocess.DEVNULL, env={**os.environ,
                                                              "DVD_UPSCALE_NCNN_OPTS": name})
            text = (r.stdout + r.stderr).decode("utf-8", "replace")
            files = sorted(out.glob("*.png"))
            if r.returncode or len(files) != n_in or GPU_ERRORS.search(text):
                print(f"  {mname:28s} didn't run cleanly (exit code {r.returncode})", flush=True)
                time.sleep(15)
                continue
            times = [f.stat().st_mtime for f in files]
            fps = (len(times) - 1) / max(1e-6, max(times) - min(times)) if len(times) > 2 else 0
            run(["ffmpeg", "-y", "-v", "error", "-i", str(out / "%06d.png"), "-vf",
                 "scale=720:480:flags=lanczos", str(fit / "%06d.png")])
            rows.append((f"{mname} (x{k})", *score(fit), fps, fit))
            by_model[mname] = rows[-1]
            time.sleep(5)
        ref_speed = next((r[3] for r in rows if r[0].startswith("realesrgan-x2plus")), None)
        print(f"\n  {'model':32s} {'SSIM':>7s} {'PSNR':>7s}   speed (x2plus = 1.0)")
        for label, ssim, psnr, fps, _ in sorted(rows, key=lambda r: -r[1]):
            sp = "" if fps is None or not ref_speed else f"{fps / ref_speed:.1f}"
            print(f"  {label:32s} {ssim:7.4f} {psnr:6.1f}   {sp}")
        # a picture to judge by eye: the same part of frame 3 (original, then each model)
        pick = d / "hr" / f"{min(n_in, 3):06d}.png"
        ins = [("original", pick)] + [(r[0], r[4] / pick.name) for r in rows]
        cmd = ["ffmpeg", "-y", "-v", "error"]
        for _, f in ins:
            cmd += ["-i", str(f)]
        filt = "".join(f"[{i}:v]crop=240:160:240:160,scale=480:320:flags=neighbor[c{i}];"
                       for i in range(len(ins)))
        filt += "".join(f"[c{i}]" for i in range(len(ins))) + f"hstack=inputs={len(ins)}"
        cmd += ["-filter_complex", filt, "-frames:v", "1", str(here / "model_compare.png")]
        rc, _ = capture(cmd)
        if rc == 0:
            print("\n  model_compare.png (next to the script), left to right: "
                  + ", ".join(x[0] for x in ins))
    best = max(rows[1:], key=lambda r: r[1], default=None)
    if best:
        print(f"\n  Closest to the original: {best[0]} (SSIM {best[1]:.4f}). Look at "
              "model_compare.png before choosing: a higher score is a truer picture, not always "
              "a nicer one.")
    if w.save:
        return save_best_model(by_model)
    return 0


QUALITY_MARGIN = 0.002      # SSIM a model must beat realesrgan-x2plus by to replace it
QUALITY_MIN_SPEED = 0.3     # ...and it must run at least this fast against it


def save_best_model(by_model):
    """From --ncnn-models --save: the model for live action and 3D animation. Only the ones
    made for photographs (x2plus, x4plus, the general video model): the anime ones are trained
    on drawings. realesrgan-x2plus is what those types use already, so another model takes
    over only when it is clearly truer to the original (SSIM better by QUALITY_MARGIN) and not
    much slower (at least QUALITY_MIN_SPEED of its speed: x4plus does 4x the work). The choice
    is saved as best_model in ncnn_opts.json and used only by a run with --best-quality."""
    photo = {m: r for m, r in by_model.items()
             if "anime" not in m and re.search(r"x2plus|x4plus|general", m)}
    if not photo:
        print("  No model for live action in the models folder: nothing saved.")
        return 1
    base = photo.get("realesrgan-x2plus")
    pick, why = base and "realesrgan-x2plus", "the default for live action and 3D animation"
    if base is None:
        pick = max(photo, key=lambda m: photo[m][1])
        why = "the only choice it could test is the closest to the original"
    else:
        for m, r in sorted(photo.items(), key=lambda kv: -kv[1][1]):
            if m == "realesrgan-x2plus":
                break
            slow = (r[3] or 0) / max(1e-6, base[3] or 1e-6)
            if r[1] - base[1] >= QUALITY_MARGIN and slow >= QUALITY_MIN_SPEED:
                pick, why = m, (f"truer than realesrgan-x2plus by {r[1] - base[1]:.4f} SSIM, "
                                f"at {slow:.1f}x its speed")
                break
    saved = ncnn_saved()
    saved["best_model"] = pick
    ncnn_opts_file().write_text(json.dumps(saved), encoding="utf-8")
    print(f"  Saved best_model = {pick} ({why}). Nothing changes by itself: add --best-quality "
          "to an upscale of live action or 3D animation to use it (a movie already started "
          "keeps its model).")
    return 0


def ncnn_winograd_main(argv):
    """python dvd_upscale.py --ncnn-winograd [MOVIE] [--gpu N] [--frames 300]: each of ncnn's
    three winograd convolution variants on its own, on 300 frames of the movie (whole frames,
    the saved tile), against "nowinograd" (the safe set the stress test found). Winograd is the
    fast way to do 3x3 convolutions (nearly all of realesrgan-x2plus) and was switched off
    whole because the GPU got reset with it on; if only one variant faults, the others can come
    back. A variant must get through all the frames without a GPU error and give the same
    picture (48 dB or closer: only fp16 rounding differs). The fastest is saved as the set."""
    p = argparse.ArgumentParser(prog="dvd_upscale.py --ncnn-winograd")
    p.add_argument("movie", nargs="?")
    p.add_argument("--gpu")
    p.add_argument("--frames", type=int, default=300)
    w = p.parse_args(argv)
    here = Path(__file__).resolve().parent
    exe = shutil.which(ESRGAN_DEFAULT) or next((str(f) for f in here.glob(ESRGAN_DEFAULT + "*")), None)
    models = Path(exe).resolve().parent / "models" if exe else here / "models"
    saved = ncnn_saved()
    tile = int(saved.get("tile") or 1024)
    trials = ["nowinograd", "w23", "w43", "w63"]
    names = {"nowinograd": "no winograd (the safe set)", "w23": "winograd 2x2", "w43": "winograd 4x4",
             "w63": "winograd 6x6"}
    with tempfile.TemporaryDirectory(prefix="ncnn_wino_") as d:
        d = Path(d)
        (d / "in").mkdir()
        if w.movie:
            rc, txt = capture(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                               "-of", "csv=p=0", w.movie])
            try:
                at = max(0.0, float(txt.strip()) * 0.4)
            except ValueError:
                at = 0.0
            print(f"Taking {w.frames} frames from {w.movie}...", flush=True)
            run(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.1f}", "-i", w.movie, "-map",
                 "0:v:0", "-vf", "scale=iw*sar:ih,setsar=1,scale=720:480", "-frames:v",
                 str(w.frames), str(d / "in" / "%06d.png")])
        else:
            run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                 f"testsrc2=s=720x480:r=24:d={w.frames / 24:.3f}", "-frames:v", str(w.frames),
                 str(d / "in" / "%06d.png")])
        n_in = len(list((d / "in").glob("*.png")))
        ref, results, trial_secs = None, [], []
        est = len(trials) * (n_in / 3.0 + 25)
        print(f"{len(trials)} runs of {n_in} frames: about {est / 60:.0f} minutes ("
              f"{eta_text(est)})", flush=True)
        for k, name in enumerate(trials):
            out = d / f"out{k}"
            out.mkdir()
            cmd = [sys.executable, Path(__file__).resolve(), "--ncnn-upscaler", "-i", d / "in",
                   "-o", out, "-n", "realesrgan-x2plus", "-s", "2", "-f", "png", "-j", "1:1:1",
                   "-t", tile, "-m", models]
            if w.gpu is not None:
                cmd += ["-g", w.gpu]
            proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    env={**os.environ, "DVD_UPSCALE_NCNN_OPTS": name})
            captured = []
            reader = threading.Thread(target=lambda pr=proc, box=captured: box.append(pr.stdout.read()),
                                      daemon=True)
            reader.start()
            t_trial = time.time()
            while proc.poll() is None:
                time.sleep(2)
                try:
                    n_now = len(os.listdir(out))
                except OSError:
                    n_now = 0
                spent = time.time() - t_trial
                rate = n_now / spent if n_now >= 10 and spent > 0 else 3.0
                this_left = max(0.0, (n_in - n_now) / rate)
                each = (sum(trial_secs) / len(trial_secs)) if trial_secs else n_in / rate + 10
                left = this_left + (len(trials) - k - 1) * (each + 15)
                status_line(f"  {names[name]}: frame {n_now} of {n_in} | run {k + 1} of "
                            f"{len(trials)} | {eta_text(left)}")
            status_line()
            proc.wait()
            reader.join(10)         # (all of its output, so no error line is missed)
            trial_secs.append(time.time() - t_trial)
            text = b"".join(captured).decode("utf-8", "replace")
            r = proc
            files = sorted(out.glob("*.png"))
            ok = r.returncode == 0 and len(files) == n_in and not GPU_ERRORS.search(text)
            if not ok:
                print(f"  {names[name]:28s} GPU error after {len(files)} of {n_in} frames", flush=True)
                shutil.rmtree(out, ignore_errors=True)
                time.sleep(15)          # (the driver recovers)
                continue
            times = [f.stat().st_mtime for f in files]
            fps = (len(times) - 1) / max(1e-6, max(times) - min(times))
            db = float("inf")
            if ref is None:
                ref = out
            else:
                rc, rep = capture(["ffmpeg", "-v", "info", "-i", str(out / "%06d.png"), "-i",
                                   str(ref / "%06d.png"), "-lavfi", "psnr", "-f", "null", "-"])
                m = re.search(r"PSNR .*?average:([0-9.]+|inf)", rep)
                db = float(m.group(1)) if m else 0.0
                shutil.rmtree(out, ignore_errors=True)
            if db >= 48:
                print(f"  {names[name]:28s} {fps:.2f} frames/s, got through all {n_in} frames"
                      + ("" if db == float("inf") else f", {db:.0f} dB from the reference"),
                      flush=True)
                results.append((fps, name))
            else:
                print(f"  {names[name]:28s} {fps:.2f} frames/s but the picture differs "
                      f"({db:.0f} dB): not used", flush=True)
            time.sleep(5)
    if not results:
        print("Nothing ran cleanly: the saved settings are unchanged.")
        return 1
    fps, name = max(results)
    base = next((f for f, n in results if n == "nowinograd"), None)
    ncnn_opts_file().write_text(json.dumps({**ncnn_saved(), "set": name, "tile": tile}),
                                encoding="utf-8")
    print(f"Saved '{name}' ({fps:.2f} frames/s"
          + (f", {fps / base * 100 - 100:+.0f}% against no winograd" if base and name != "nowinograd" else "")
          + f") in {ncnn_opts_file().name}.")
    return 0


def ncnn_upscaler_main(argv):
    """python dvd_upscale.py --ncnn-upscaler -i IN -o OUT -n MODEL -s SCALE [-m MODELS] [-t TILE]
    [-g GPU] [-j L:P:S] [-f png]: realesrgan-ncnn-vulkan's job done with the current ncnn from
    pip (NCNN_INSTALL) instead of the April 2022 ncnn built into it.
    Why: NVIDIA drivers from 570 on give a program robust buffer access only when it asks, and
    the 2022 ncnn doesn't ask (fixed in ncnn 20250916, "fix hangs with NVIDIA >565 drivers").
    Big networks (x2plus, x4plus) then hang the GPU now and then and Windows resets it
    ("vkWaitForFences/vkQueueSubmit failed -4"; seen on an RTX 3060 laptop, driver 610, after
    10-200 frames, while the small anime model ran for hours).
    The same command line (so run_upscaler and all around it work unchanged) and the same
    picture: realesrgan.cpp's tiling ported as is. Tiles of -t pixels (0: by the GPU's memory,
    200 above 1.9 GB) overlap by 10 pixels, mirrored at the frame edges; in RGB / 255, out x 255
    rounded; fp16 storage, fp32 arithmetic. Only fp16 rounding differs (55 dB). One frame at a
    time (ncnn's Python module holds the interpreter while the GPU works); frames are read in
    one ffmpeg stream and written by 3 threads. A GPU error ends it at once, with a line
    run_upscaler knows (GPU_ERRORS)."""
    p = argparse.ArgumentParser(prog="dvd_upscale.py --ncnn-upscaler")
    for opt in ("-i", "-o", "-n", "-s", "-m", "-t", "-g", "-j", "-f"):
        p.add_argument(opt)
    w = p.parse_args(argv)
    import faulthandler
    faulthandler.enable()       # (a crash inside ncnn or the driver then prints where, on stderr)
    try:
        import numpy as np
        import ncnn
    except ImportError as e:
        print(f"ncnn: can't load the current ncnn ({e}): {ncnn_install_hint()}", flush=True)
        return 1
    from concurrent.futures import ThreadPoolExecutor
    scale, tile, pad = int(w.s), int(w.t or ncnn_saved().get("tile") or 0), 10
    models = Path(w.m or "models")
    for name in (w.n, f"{w.n}-x{scale}"):           # (the anime models carry the scale)
        if (models / f"{name}.param").exists():
            break
    else:
        print(f"ncnn: {w.n}.param not found in {models}", flush=True)
        return 1
    if ncnn.get_gpu_count() == 0:       # (ncnn has printed the Vulkan problem)
        print("ncnn: no Vulkan GPU found (no vulkan)", flush=True)
        return 1
    gpu = int(w.g) if w.g not in (None, "") else ncnn.get_default_gpu_index()
    net = ncnn.Net()
    o = net.opt
    o.use_vulkan_compute = True
    o.use_fp16_packed = o.use_fp16_storage = True
    o.use_fp16_arithmetic = o.use_bf16_storage = o.use_bf16_packed = False
    o.use_int8_storage = True
    for opt_name, opt_value in NCNN_OPT_SETS[ncnn_opt_set_name()].items():
        if hasattr(o, opt_name):
            setattr(o, opt_name, opt_value)
    net.set_vulkan_device(gpu)
    # (loaded from inside the models folder, by bare file names: ncnn's fopen takes a path in
    # Windows' ANSI code page, so a folder like "Vidéos" in it would fail; the folder itself the
    # system finds by its real name)
    here = os.getcwd()
    try:
        os.chdir(models)
        bad = net.load_param(f"{name}.param") or net.load_model(f"{name}.bin")
    except OSError:
        bad = True
    finally:
        os.chdir(here)
    if bad:
        print(f"ncnn: can't load {models / name}.param/.bin", flush=True)
        return 1
    if not tile:        # (realesrgan-ncnn-vulkan's own rule, main.cpp)
        budget = ncnn.get_gpu_device(gpu).get_heap_budget()
        tile = 200 if budget > 1900 else 100 if budget > 550 else 64 if budget > 190 else 32
    src_dir, dst_dir = Path(w.i), Path(w.o)
    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".png"))
    print(f"ncnn {ncnn.__version__}: GPU {gpu} {ncnn.get_gpu_info(gpu).device_name()}, "
          f"{name}, x{scale}, {tile}-pixel tiles, {len(files)} frames", flush=True)
    if not files:
        return 0
    iw, ih = png_size(src_dir / files[0])
    # (the PNGs fed in one stream, not named in a list: image2 would read a "%20d" in a movie's
    # name as a frame-number pattern)
    reader = subprocess.Popen(["ffmpeg", "-v", "error", "-f", "image2pipe", "-c:v", "png",
                               "-i", "-", "-fps_mode", "passthrough", "-f", "rawvideo",
                               "-pix_fmt", "rgb24", "-"],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def feed():
        try:
            for f in files:
                reader.stdin.write((src_dir / f).read_bytes())
            reader.stdin.close()
        except (OSError, ValueError):
            pass
    threading.Thread(target=feed, daemon=True).start()
    saver = ThreadPoolExecutor(3)
    pending = []
    # (the tests: a GPU error after this many frames, in a process started with a tile size
    # above 100; see dvd_upscale_dev)
    fail_at = int(os.environ.get("DVD_UPSCALE_TEST_NCNN_FAIL", "-1")) if tile > 100 else -1
    try:
        for k, f in enumerate(files):
            buf = reader.stdout.read(iw * ih * 3)
            if len(buf) < iw * ih * 3:
                print(f"ncnn: decode image {f} failed", flush=True)
                return 1
            img = np.frombuffer(buf, np.uint8).reshape(ih, iw, 3)
            src = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
            src = np.ascontiguousarray(src.transpose(2, 0, 1), dtype=np.float32) * (1 / 255)
            out = np.empty((ih * scale, iw * scale, 3), np.uint8)
            for y0 in range(0, ih, tile):
                th = min(tile, ih - y0)
                for x0 in range(0, iw, tile):
                    tw = min(tile, iw - x0)
                    part = np.ascontiguousarray(src[:, y0:y0 + th + 2 * pad, x0:x0 + tw + 2 * pad])
                    ex = net.create_extractor()
                    # (a copy: ncnn.Mat(array) keeps numpy's channel stride, h*w, but the GPU
                    # upload wants each channel to start at a multiple of 4 values: a tile of
                    # odd size, padding included, would get its green and blue shifted)
                    mat = ncnn.Mat(part).clone()
                    ex.input("data", mat)
                    ret, res = ex.extract("output")
                    if k == fail_at:
                        ret = -4
                    if ret:
                        print(f"ncnn: GPU error (extract returned {ret}) on {f}", flush=True)
                        return 1
                    res = np.array(res)[:, pad * scale:(pad + th) * scale,
                                        pad * scale:(pad + tw) * scale]
                    out[y0 * scale:(y0 + th) * scale, x0 * scale:(x0 + tw) * scale] = \
                        np.clip(np.floor(res * 255 + 0.5), 0, 255).astype(np.uint8).transpose(1, 2, 0)
                    del ex, mat, part
            pending.append(saver.submit(write_png, dst_dir / f"{Path(f).stem}.png", out))
            while len(pending) > 6:     # (frames waiting to be written: a few)
                pending.pop(0).result()
        for fut in pending:
            fut.result()
        return 0
    except OSError as e:
        print(f"ncnn: {e}", flush=True)
        return 1
    finally:
        saver.shutdown()        # (the frames finished before an error are good: written)
        reader.kill()
        reader.wait()


def faces_worker_main(argv):
    """python dvd_upscale.py --faces-worker ...: the face restoration of one chunk's frames, in
    a process of its own (started by Chunk.restore_faces, never by hand). It only reads the
    frames folder and writes into --dest. --check: the packages and model files are there and
    the models run; prints the provider used. --cpu: on the processor (the retry: the GPU may
    be short of memory next to the upscalers). Exit codes: 0 done, 3 a package is missing,
    fails to load or is too old, 4 a model file is missing, 5 a model file is damaged (or not
    fully downloaded), 130 stopped (Ctrl+C), anything else a failure."""
    p = argparse.ArgumentParser(prog="dvd_upscale.py --faces-worker")
    p.add_argument("--check", action="store_true")
    p.add_argument("--frames")
    p.add_argument("--dest")
    p.add_argument("--models", required=True)
    p.add_argument("--model", choices=sorted(FACE_MODEL_FILES), default=FACE_MODEL)
    p.add_argument("--strength", type=float, default=FACE_STRENGTH)
    p.add_argument("--fidelity", type=float, default=FACE_FIDELITY)
    p.add_argument("--scale", type=float, default=2)
    p.add_argument("--ai-blend", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true")
    w = p.parse_args(argv)
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")     # (its warnings mean nothing to users)
    if not w.check:
        # the main run holds this process's input open: when it ends, however it ends (even
        # killed), the input closes and this process stops too instead of running on alone
        def watch_parent():
            # (os.read, not sys.stdin: a thread still waiting in sys.stdin when this process
            # exits makes Python abort the exit)
            try:
                while os.read(0, 4096):
                    pass
            except OSError:
                pass
            os._exit(1)
        threading.Thread(target=watch_parent, daemon=True).start()
    try:
        missing, broken = [], []
        for module, package, dists in (("numpy", "numpy", ("numpy",)),
                                       ("cv2", "opencv-python-headless", CV_DISTS),
                                       ("onnxruntime", "onnxruntime", ORT_DISTS)):
            try:
                __import__(module)
            except ImportError as e:
                # missing only if the module itself isn't there and pip doesn't list it either:
                # installed but failing to load (built for another numpy, a DLL missing, its
                # files deleted, ...) is another matter
                if isinstance(e, ModuleNotFoundError) and e.name == module \
                        and not face_dists(*dists):
                    missing.append(package)
                else:
                    broken += face_dists(*dists) or [package]
                print(f"({module}: {e})", flush=True)
                continue
            if getattr(sys.modules[module], "__file__", None) is None:
                # only its folder is left (another package that shared it was uninstalled):
                # Python takes a bare folder for a module, which then has nothing in it
                print(f"({module}: its files are missing)", flush=True)
                broken += face_dists(*dists) or [package]
        if missing:
            print("MISSING-PACKAGES " + " ".join(missing), flush=True)
        if broken:
            print("BROKEN-PACKAGES " + " ".join(broken), flush=True)
        if missing or broken:
            return 3
        import cv2
        # (the 2023mar YuNet model needs OpenCV 4.8 or newer)
        if tuple(int(x) for x in re.findall(r"\d+", cv2.__version__)[:2]) < (4, 8) or \
                not hasattr(cv2, "FaceDetectorYN"):
            print(f"OLD-OPENCV {cv2.__version__}", flush=True)
            return 3
        gone = [f for f in (FACE_DETECTOR, FACE_MODEL_FILES[w.model])
                if not os.path.isfile(os.path.join(w.models, f))]
        if gone:
            print("MISSING-MODELS " + " ".join(gone), flush=True)
            return 4
        # an empty or nearly empty file is a download that failed (every model is far bigger;
        # onnxruntime and OpenCV report one in words that don't say so)
        bad = [f for f in (FACE_DETECTOR, FACE_MODEL_FILES[w.model])
               if os.path.getsize(os.path.join(w.models, f)) < 1024]
        if bad:
            print("(empty or nearly empty)\nBAD-MODEL " + " ".join(bad), flush=True)
            return 5
        if w.check:
            import numpy as np
            import onnxruntime as ort
            # both models load and run; a file that doesn't parse is damaged or half downloaded
            # (any other error, a GPU provider's say, is reported as it is)
            try:
                r = FaceRestorer(w.models, w.model, w.fidelity)
            except Exception as e:
                if type(e).__name__ not in ("InvalidProtobuf", "NoSuchFile", "InvalidGraph") \
                        and "Protobuf parsing failed" not in str(e) \
                        and "does not have a graph" not in str(e):
                    raise
                print(f"({e})\nBAD-MODEL {FACE_MODEL_FILES[w.model]}", flush=True)
                return 5
            try:
                r.detect(np.full((64, 64, 3), 128, np.uint8))
            except cv2.error as e:
                if "parse" not in str(e).lower() and "has_graph" not in str(e):
                    raise
                print(f"({e})\nBAD-MODEL {FACE_DETECTOR}", flush=True)
                return 5
            r.restore_crop(np.full((512, 512, 3), 128, np.uint8))
            # (for the advice if it ends up on the processor)
            print("AVAILABLE " + " ".join(ort.get_available_providers()), flush=True)
            print("PACKAGES " + " ".join(face_dists(*ORT_DISTS)), flush=True)
            print(f"PROVIDER {r.provider}", flush=True)
            return 0

        def progress(step, done, of):
            print(f"{step} {done} {of}", flush=True)
        faces, changed, provider = restore_faces(
            w.frames, w.models, w.model, w.strength, w.fidelity, w.scale, w.dest,
            ["CPUExecutionProvider"] if w.cpu else None, progress, w.ai_blend)
        print(f"DONE {faces} {changed} {provider}", flush=True)
        return 0
    except KeyboardInterrupt:
        return 130


def face_models_dir(a):
    """Where the face model files are: --face-models, else face_models next to this script."""
    return Path(a.face_models or Path(__file__).resolve().parent / "face_models").resolve()


def pip_cmd():
    """pip for the Python running this script, where the face worker looks for its packages:
    "python -m pip" when that is the python on the PATH, else this Python's own path (a plain
    "pip" can belong to another Python, or not be there at all)."""
    exe = sys.executable
    on_path = shutil.which("python")
    # (the paths compared as they are: a venv's python is a link to the system one, which
    # mustn't count as the same Python)
    if on_path and os.path.normcase(os.path.abspath(on_path)) == \
            os.path.normcase(os.path.abspath(exe)):
        return "python -m pip"
    if " " not in exe:
        return f"{exe} -m pip"
    if os.name == "nt" and sys.prefix == sys.base_prefix:
        # (the py launcher: no quotes, which PowerShell would need an & in front of)
        return f"py -{sys.version_info[0]}.{sys.version_info[1]} -m pip"
    return f'"{exe}" -m pip'


def face_install_hint(pkgs=("onnxruntime", "opencv-python-headless", "numpy"), gpu_only=False):
    """The pip command that installs these packages for --faces on this computer, onnxruntime
    as its GPU version where there is one (gpu_only: "" where there is none)."""
    rest = "".join(" " + p for p in pkgs if p != "onnxruntime")
    if "onnxruntime" not in pkgs:
        return f"{pip_cmd()} install -U{rest}"
    if os.name == "nt":
        return (f"{pip_cmd()} install -U onnxruntime-directml{rest}   (any GPU; with an NVIDIA "
                "card and CUDA and cuDNN installed, onnxruntime-gpu instead of "
                "onnxruntime-directml)")
    if shutil.which("nvidia-smi"):
        return f"{pip_cmd()} install -U onnxruntime-gpu{rest}   (needs CUDA and cuDNN)"
    return f"{pip_cmd()} install -U onnxruntime{rest}" if not gpu_only else ""


def face_reinstall_hint():
    """The pip command that reinstalls the --faces packages installed here: for files damaged
    or deleted under a package pip still lists (pip install -U alone does nothing then)."""
    ort = face_dists(*ORT_DISTS)
    if len(ort) > 1:        # (they share one folder and break each other)
        return f"{pip_cmd()} uninstall {' '.join(ort)}, then {face_install_hint()}"
    got = ort + face_dists(*CV_DISTS) + face_dists("numpy")
    return (f"{pip_cmd()} install -U --force-reinstall {' '.join(got)}" if got
            else face_install_hint())


class FacesUnavailable(Exception):
    """The automatic face restoration can't run here (packages or model files missing)."""


def faces_check(a):
    """--faces, before anything long: the packages and model files are there and the models
    load (in a worker, as for the chunks). Exits with what to install or download if not (when
    it was turned on by itself, raises FacesUnavailable instead: the movie goes on without);
    prints what it will use."""
    def stop(msg):
        if getattr(a, "faces_auto", False):
            raise FacesUnavailable(msg)
        sys.exit(msg)
    models = face_models_dir(a)
    a.face_models_path = str(models)
    status_line("  checking the face restoration (loading its models)...")
    try:    # (not capture: a model load that hangs mustn't stop the run for good)
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--faces-worker",
                            "--check", "--models", str(models), "--model", a.face_model,
                            "--fidelity", str(FACE_FIDELITY)],
                           capture_output=True, stdin=subprocess.DEVNULL, timeout=600)
    except subprocess.TimeoutExpired:
        status_line()
        stop("--faces: the face restoration's check didn't finish (its models didn't load "
                 "in 10 minutes). Reinstalling its packages may help:\n  "
                 + face_reinstall_hint())
    rc, text = p.returncode, (p.stdout + p.stderr).decode("utf-8", "replace")
    status_line()
    lines = [x for x in text.splitlines() if x.strip()]
    found = dict(x.split(" ", 1) for x in lines if " " in x and x.split(" ", 1)[0].isupper())
    if rc == 3:
        old_cv, broken = found.get("OLD-OPENCV"), found.get("BROKEN-PACKAGES")
        miss = found.get("MISSING-PACKAGES", "").split()
        why = [x for x in (
            f"this OpenCV, {old_cv}, is too old for its face detector: it needs 4.8 or newer"
            if old_cv else "",
            f"{' '.join(miss)} missing" if miss else "",
            f"{broken} installed but failing to load" if broken else "") if x]
        # only the ones missing, and the OpenCV package installed if it is too old: a second
        # onnxruntime (or OpenCV) package next to a working one shares its folder and breaks it
        need = miss + ((face_dists(*CV_DISTS) or ["opencv-python-headless"]) if old_cv else [])
        stop("--faces needs a few Python packages (" + "; ".join(why) + ")"
                 + "".join(f"\n  {x.strip()}" for x in lines if x.startswith("("))
                 + (f"\n{'Install them' if miss else 'Update it'} with:\n  "
                    + face_install_hint(need) if need else "")
                 + (f"\n{'and r' if need else 'R'}einstall the ones that fail to load:\n  "
                    f"{pip_cmd()} install -U --force-reinstall {broken}"
                    + (" (and install the Microsoft Visual C++ Redistributable)"
                       if os.name == "nt" and "onnxruntime" in broken else "") if broken else "")
                 + "\nThen run the same command again.")
    if rc == 4:
        files = found.get("MISSING-MODELS", "").split()
        stop(f"--faces needs its model files in '{models}'"
                 + ("" if a.face_models else " (a face_models folder next to dvd_upscale.py; "
                    "--face-models DIR for another folder)")
                 + ". Missing:\n" + "".join(f"  {f}  from  {FACE_URLS.get(f, '?')}\n"
                                            for f in files)
                 + "Download them into that folder, then run the same command again.")
    if rc == 5:
        files = found.get("BAD-MODEL", "").split()
        stop(f"--faces: a model file in '{models}' is damaged or not fully downloaded:\n"
                 + "".join(f"  {f}  from  {FACE_URLS.get(f, '?')}\n" for f in files)
                 + "Delete it and download it again into that folder, then run the same command "
                   "again.")
    provider = found.get("PROVIDER")
    if rc or not provider:
        stop("--faces: the face restoration couldn't start. Its output (end):\n  "
                 + "\n  ".join(lines[-15:])
                 + "\nReinstalling its packages may help:\n  " + face_reinstall_hint())
    a.face_provider = provider.strip()
    gpu = a.face_provider != "CPUExecutionProvider"
    name = FACE_MODEL_NAMES[a.face_model] + (f" (fidelity {FACE_FIDELITY:g})"
                                             if a.face_model == "codeformer" else "")
    print(f"Faces: {name} at strength {a.faces:g}"
          + (f" (at most {a.ai_blend:g}: the AI frames' share of the picture, --ai-blend)"
             if a.faces > a.ai_blend else "")
          + f" on {a.face_provider} ({'GPU' if gpu else 'processor'})")
    if a.face_model == "codeformer":
        print("  (CodeFormer's licence, S-Lab License 1.0, allows non-commercial use only)")
    if not gpu:
        hint = face_install_hint(["onnxruntime"], gpu_only=True)
        pkgs = found.get("PACKAGES", "").split() or ["onnxruntime"]
        gpu_ways = [x for x in found.get("AVAILABLE", "").split()
                    if x in ("CUDAExecutionProvider", "DmlExecutionProvider")]
        # (onnxruntime-gpu says which CUDA and cuDNN it needs when it can't start on them)
        failed = [x.strip() for x in lines if "Failed to create" in x or "EP Error" in x]
        if len(pkgs) > 1:
            how = (f" More than one onnxruntime package is installed ({', '.join(pkgs)}), which "
                   f"share one folder and break each other: {pip_cmd()} uninstall "
                   f"{' '.join(pkgs)}, then {hint or face_install_hint(['onnxruntime'])}")
        elif gpu_ways:
            how = (f" {pkgs[0]} can use the GPU ({gpu_ways[0]}), but that didn't start"
                   + (f": {failed[-1][:300]}" if failed else "") + "."
                   + (" It needs the CUDA and cuDNN versions it names installed."
                      if gpu_ways[0] == "CUDAExecutionProvider" else "")
                   + (f" Or, for any GPU: {pip_cmd()} uninstall {pkgs[0]}, then {pip_cmd()} "
                      "install -U onnxruntime-directml" if os.name == "nt"
                      and pkgs[0] != "onnxruntime-directml" else ""))
        elif hint:
            how = f" For the GPU: {pip_cmd()} uninstall {pkgs[0]}, then {hint}"
        else:
            how = " There is no GPU version of onnxruntime set up for this computer."
        print("NOTE: the face restoration runs on the processor: very slow (seconds per face, "
              "so a chunk with many faces can take many minutes)." + how)


THROTTLE_BITS = ((0x4, "power cap"), (0x8, "hardware slowdown"), (0x20, "thermal"),
                 (0x40, "hardware thermal"), (0x80, "power brake"))


class GpuWatch:
    """While a chunk is upscaled: the NVIDIA GPU's clock, temperature, power and throttle
    reasons (nvidia-smi, every 4 s), to tell a GPU that slows itself down (heat, power) from
    one that waits for the processor. summary() is one short line, "" if there is no nvidia-smi."""

    def __init__(self):
        self.rows, self.stop = [], threading.Event()
        if shutil.which("nvidia-smi"):
            threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while not self.stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,"
                     "power.draw,utilization.gpu,clocks_throttle_reasons.active",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    timeout=10, stdin=subprocess.DEVNULL)
                v = [x.strip() for x in r.stdout.splitlines()[0].split(",")]
                self.rows.append((float(v[0]), float(v[1]), float(v[2]), float(v[3]),
                                  float(v[4]), int(v[5], 16)))
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            self.stop.wait(4)

    def summary(self):
        self.stop.set()
        rows = self.rows[2:] or self.rows         # (the first samples are the upscaler starting)
        if not rows:
            return ""
        med = lambda i: statistics.median(r[i] for r in rows)
        why = [name for bit, name in THROTTLE_BITS
               if sum(1 for r in rows if r[5] & bit) > len(rows) / 4]
        return (f"GPU {med(0):.0f} of {med(1):.0f} MHz, {med(2):.0f} C, {med(3):.0f} W, "
                f"{med(4):.0f}% busy" + (f", slowed by: {', '.join(why)}" if why else ""))


class Chunk:
    """One chunk of the movie, in three steps: read its frames out of the source (extract), run
    them through the upscaler (upscale, on the GPU) and encode them into the chunk file
    (finish). It is set up in the main thread, where the settings that differ from chunk to
    chunk are fixed, so extract and finish can then run in other threads: the next chunk's
    frames are read, and the previous chunk encoded, while this one is upscaled."""

    def __init__(self, a, idx, start, length, expected, work, label, last=False, warm=False,
                 stab=None):
        self.a, self.idx, self.expected, self.last, self.label = a, idx, expected, last, label
        self.start, self.length = float(start), float(length)
        # the GPU it is upscaled on, and for a helper GPU its name and where its messages go
        self.gpu, self.lane, self.note = None, None, None
        # the main GPU's chunk, with helper GPUs running: stops them when the drive is too full
        # for both (returns True if it did)
        self.make_room = None
        self.out = work / f"chunk_{idx:05d}.mkv"
        self.part = work / f"chunk_{idx:05d}.part.mkv"     # only renamed when fully encoded
        self.tmp = work / f"tmp_{idx:05d}"
        # read ~0.25s extra: deinterlace/IVTC filters lose a frame or two at the end of a cut,
        # and -frames:v below trims the result to exactly `expected` frames (no per-chunk drift)
        # (no -ss at all for the first chunk: an AVI with B-frames can't seek to 0 and lost the
        # frames read while opening it)
        self.src = [*(["-ss", start] if float(start) > 0 else []),
                    "-t", f"{float(length) + 0.25:.6f}", "-i", a.chunk_input, "-an", "-sn"]
        a.vhs_trim = a.dvd_trim = 0
        if a.type == "vhs":
            pre = 0
            if warm:
                # VHS: start a few tape frames early so the IVTC (fieldmatch/decimate), bwdif and
                # the strong temporal denoise don't restart cold on every chunk (measured: without
                # it a chunk's first frames came out 26-39 dB off an uncut run - wrong IVTC
                # frames, a noise pulse; with it 58 dB or better); the frames made from them are
                # cut off again at the end of the prefilter
                pre, a.vhs_trim = a.vhs_warm
            seek = float(start) - pre / a.vhs_fin
            # (no -ss for the first chunk here either: on an AVI with B-frames, -ss 0 gave
            # garbled frames and lost the first ~1.6 s, or no frames at all for H.264)
            self.src = [*(["-ss", f"{seek:.6f}"] if seek > 0 else []),
                        "-t", f"{float(length) + pre / a.vhs_fin + 0.25:.6f}",
                        "-i", a.chunk_input, "-an", "-sn"]
        elif warm:
            # inverse telecine: start two 5-frame cycles early so fieldmatch/decimate (and the
            # denoise) don't restart cold at the cut (else, at one of the 5 cadence positions,
            # every chunk's first frame came out combed); the 8 frames made from them are cut
            # off again
            pre = 10 * 1001 / 30000
            self.src = ["-ss", f"{float(start) - pre:.6f}",
                        "-t", f"{float(length) + pre + 0.25:.6f}",
                        "-i", a.chunk_input, "-an", "-sn"]
            a.dvd_trim = 8
        self.pre = prefilter(a)             # (fixed now: it reads this chunk's warm-up trim)
        # --stabilize: (first, end, margin): this chunk's piece of the camera-motion file,
        # frames first..end-1 of the whole video: `margin` frames before this chunk, which are
        # read and cut off again after the stabilizer, and some after it, so the smoothing has
        # the same frames around each of this chunk's frames as in one run through the video
        self.stab = stab
        if stab:
            self.pre += (f",vidstabtransform=input=stab.trf:{a.stab_tf}"
                         + (f",trim=start_frame={stab[2]},setpts=PTS-STARTPTS" if stab[2] else ""))
        self.encode = None
        self.up_info = self.gpu_info = ""
        # --faces: what finish is doing while the main GPU waits for it (None: not restoring
        # faces), and the face restoration's time on the first chunk, shown once
        self.stage = self.face_msg = None

    def clear_frames(self):
        # GBs of frames that a retry makes again anyway: don't leave them filling the drive
        # (upscaler_log.txt stays, for a look at what went wrong)
        for d in ("in", "in_rest", "in_black", "out", "faces"):
            shutil.rmtree(self.tmp / d, ignore_errors=True)

    def no_video(self):
        """The video ended before this chunk: fine for the last one (the length estimate ran a
        little past the end), else the file is cut short."""
        if not self.last:
            sys.exit(f"No video found from {self.start / 60:.1f} min on, although the file "
                     "says it is longer: it is probably incomplete or damaged (an interrupted "
                     "copy or rip?). Running it again won't help: copy or rip it again.")
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.part.unlink(missing_ok=True)

    def extract(self):
        """The frames as PNGs. Returns (frame count, warning or None); 0 frames: no video."""
        tmp = self.tmp
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            (tmp / "in").mkdir(parents=True)
            (tmp / "out").mkdir()
            # fast PNG compression: the same pixels, written several times faster. Into RGB with
            # the colour interpolated (full_chroma_int): ffmpeg's default repeats each colour
            # sample of the half-width DVD colour on two pixels, a stair-step on every colour
            # edge that the upscaler then sharpens (not for a movie started without it)
            rgb = (",scale=flags=bicubic+accurate_rnd+full_chroma_int,format=rgb24"
                   if getattr(self.a, "rgb_interp", False) else "")
            run(["ffmpeg", "-y", "-v", "error", *self.src, "-vf", self.pre + rgb,
                 "-frames:v", self.expected, "-compression_level", "1", pngs(tmp / "in")],
                cwd=self.write_stab())
            n_in = len(list((tmp / "in").glob("*.png")))
            warning = None
            if 0 < n_in < self.expected and not self.last:
                warning = gap_message(self.idx, self.expected - n_in, self.expected)
                for k in range(n_in + 1, self.expected + 1):
                    shutil.copyfile(tmp / "in" / f"{n_in:06d}.png", tmp / "in" / f"{k:06d}.png")
                n_in = self.expected
            return n_in, warning
        except BaseException:
            self.clear_frames()
            raise

    def upscale(self, n_in):
        """The upscaler, with its output checked (and one retry); then the encode is set up."""
        a, tmp, label = self.a, self.tmp, self.label
        try:
            w, h = png_size(tmp / "in" / "000001.png")
            self._reserve_disk(n_in, w, h)
            watch, t_up0, blacks = None, time.time(), set()
            try:
                watch = GpuWatch() if not self.lane else None
                blacks = self._set_aside_blacks(n_in)
                n_ai = n_in - len(blacks)       # (frames the upscaler is to make)
                attempt, src, kept = 0, tmp / "in", 0
                while True:
                    attempt += 1
                    try:
                        run_upscaler(a, src, tmp / "out", n_ai, label, (w, h), self.gpu,
                                     self.lane)
                        n_out = len(list((tmp / "out").glob("*.png")))
                        if n_out != n_ai:
                            # (the upscaler's last lines: a helper GPU's note stays one line)
                            raise RuntimeError(f"upscaler produced {n_out} of {n_ai} frames "
                                               "(GPU out of memory? try --tile 128)"
                                               + ("" if self.lane else upscaler_log_tail(
                                                   tmp / "upscaler_log.txt")))
                        self._put_back_blacks(blacks, w, h)
                        check_frames(a, tmp, n_in, strict=bool(self.lane))
                        if not self.lane:
                            FRAMES_OK[0] += n_ai - kept
                            save_gpu_step(a, frames_only=True)
                        break
                    except (RuntimeError, subprocess.CalledProcessError) as e:
                        # (no second try on a helper GPU: the main GPU redoes its chunk)
                        if self.lane:
                            raise
                        src, kept = self._before_retry(e, attempt, kept, n_ai, blacks)
            finally:
                if watch:
                    watch.stop.set()        # (also when the chunk failed or was stopped)
                with DISK_LOCK:
                    UPSCALING.pop(self, None)
            self._note_timing(t_up0, blacks, watch)
            if watch:
                watch.stop.set()
            self.encode = ["ffmpeg", "-y", "-v", "error", *self._encode_inputs(),
                           "-filter_complex", self._encode_graph(n_in), *encode_args(a),
                           self.part]
        except BaseException:
            self.clear_frames()
            raise

    def _reserve_disk(self, n_in, w, h):
        """Waits until the drive has room for this chunk's upscaled frames (counting what the
        other upscalers are still to write), and books it."""
        a, tmp = self.a, self.tmp
        frames = n_in * w * h * a.scale ** 2 * 1.8      # measured ~1.7 bytes a pixel
        # (--faces: the frames with faces are written once more, next to them)
        need = frames * (2 if a.faces else 1) + 1e9
        # a helper GPU (--gpu 0,1) also leaves room for the main GPU's next chunk
        want = need + frames if self.lane else need
        while True:
            with DISK_LOCK:
                # the frames the other GPUs are still to write count as used already
                free = shutil.disk_usage(tmp).free - sum(
                    frames_to_come(c, *v) for c, v in list(UPSCALING.items())
                    if c is not self)
                if free >= want:
                    UPSCALING[self] = (frames, n_in)
                    return
            if self.make_room and self.make_room("disk"):
                continue        # the other GPUs stopped and their frames are gone: again
            if self.lane:
                raise RuntimeError("the drive with the work folder is too full for two "
                                   "chunks at once")
            raise RuntimeError(
                f"the drive with the work folder needs about {need / 1e9:.0f} GB free for "
                f"this chunk's upscaled frames and has {free / 1e9:.1f} GB: make room (or "
                "use --work on another drive)")

    def _set_aside_blacks(self, n_in):
        """Black frames upscale to black frames: not sent to the upscaler. Their inputs are moved
        out of in/ (back after the upscale, for the checks and the blend) and their outputs
        written only after it, so out/ holds exactly what the GPU made (the retry, the frame
        counts and the progress all read it that way). Returns their names."""
        tmp = self.tmp
        if getattr(self.a, "no_skip_black", False):
            return set()
        found = [f.name for f in sorted((tmp / "in").glob("*.png")) if png_is_black(f)]
        if len(found) == n_in:
            found = found[:-1]      # (the upscaler is given at least one frame)
        if found:
            (tmp / "in_black").mkdir(exist_ok=True)
            for name in found:
                os.replace(tmp / "in" / name, tmp / "in_black" / name)
        return set(found)

    def _put_back_blacks(self, blacks, w, h):
        """The black frames' outputs, and their inputs back where the checks and the blend look
        for them."""
        if not blacks:
            return
        tmp, scale = self.tmp, self.a.scale
        write_black_png(tmp / "black.png", w * scale, h * scale)
        for name in sorted(blacks):
            shutil.copyfile(tmp / "black.png", tmp / "out" / name)
            os.replace(tmp / "in_black" / name, tmp / "in" / name)
        (tmp / "black.png").unlink()

    def _before_retry(self, e, attempt, kept, n_ai, blacks):
        """The main upscaler failed on this chunk (e): either gives up (raises) or gets the next
        try ready. Returns (folder of the frames still to do, frames kept from before)."""
        a, tmp, label = self.a, self.tmp, self.label
        # less on the GPU at once needs less of its memory and gives shorter pieces of GPU work:
        # one step down per failed try, for the rest of the run (see GPU_STEPS; --gpu-threads
        # and --tile are kept). Tried again once in any case, and as long as there is a step left
        tried = gpu_load_text(a)
        # the frames finished before a GPU error are kept, and the next try does only the rest:
        # so a GPU that is reset now and then still gets through, and at the last step a try that
        # got further goes on
        good = set(getattr(e, "good", ()))
        new, further = len(good) - kept, len(good) > kept
        kept = len(good)
        # only a GPU error lowers the load (not a full disk, say: smaller tiles change the
        # picture a little), and only when it came soon: after 120 new frames or more, a reset
        # costs less (~25 s) than a step down would for the rest of the run
        gpu_err = isinstance(e, GPUError)
        # (counted across chunks and runs: one reset early in a chunk after hours without one
        # doesn't step down for good)
        since = FRAMES_OK[0] + max(0, new)
        if gpu_err:
            FRAMES_OK[0] = 0
            save_gpu_step(a, frames_only=True)
        lower = gpu_err and since < 120 and lower_gpu_load(a)
        if attempt >= 2 and not lower and not (further and attempt < 12):
            if GPU_STEP[0] and gpu_err:
                raise RuntimeError(
                    f"{str(e).rstrip('.')}, also with {gpu_load_text(a)} (the least this "
                    "script tries). Update the graphics driver, plug the laptop in, and close "
                    "other programs that use the GPU"
                    + ("; or try --model realesrgan-x4plus" if "x2plus" in a.model else "")
                ) from e
            raise e
        # a new upscaler process gets a fresh GPU device (after a driver reset, say). Another
        # upscaler on the same GPU (--gpu-jobs) may have taken the memory this one needed: it
        # stops, and this one goes alone
        if self.make_room:
            self.make_room("retry")
        retry_note = f"going down to {gpu_load_text(a)} and " if lower else ""
        windows = (gpu_report(a, tmp.parent, label, e, tried, tmp / "upscaler_log.txt")
                   if gpu_err else "")
        status_line()
        # (the retry on the first line, any upscaler output below it)
        first, _, rest = str(e).partition("\n")
        print(f"  {label}: {first.rstrip('.')} - "
              + (f"keeping those {kept}; " if kept else "") + retry_note
              + (f"trying the other {n_ai - kept} once more." if kept else
                 "trying this chunk once more.")
              + (f"\n{rest}" if rest else ""), flush=True)
        if windows:
            print(f"  (Windows recorded: {windows})", flush=True)
        if gpu_err and len(GPU_REPORTS) == 1:
            print(f"  (details for a bug report: "
                  f"{tmp.parent.resolve() / 'gpu_errors.log'})", flush=True)
        for f in list((tmp / "out").iterdir()):
            if f.name not in good:
                f.unlink(missing_ok=True)
        # (a failed check after the black frames were put in: their inputs go aside again, else
        # the upscaler gets them and makes too many)
        for name in blacks:
            if (tmp / "in" / name).exists():
                (tmp / "in_black").mkdir(exist_ok=True)
                os.replace(tmp / "in" / name, tmp / "in_black" / name)
        shutil.rmtree(tmp / "in_rest", ignore_errors=True)
        src = tmp / "in"
        if good:
            src = tmp / "in_rest"
            src.mkdir()
            for f in (tmp / "in").glob("*.png"):
                if f.name not in good:
                    try:
                        os.link(f, src / f.name)
                    except OSError:
                        shutil.copyfile(f, src / f.name)
        if "reset" in first:
            # Windows takes a few seconds to restart the graphics driver, and crashes for good
            # when it has to 6 times within a minute: 15 s, and never more than 3 resets in 60 s
            now = time.time()
            RESET_TIMES.append(now)
            recent = [t for t in RESET_TIMES if now - t < 60]
            wait = max(15, 60 - (now - recent[-3]) if len(recent) >= 3 else 0)
            status_line(f"  {label}: waiting {wait:.0f} s for the graphics driver to recover")
            time.sleep(wait)
        return src, kept

    def _note_timing(self, t_up0, blacks, watch):
        """Where the upscale's seconds went: the upscaler starting (to its first frame), the
        frames at full speed, and what came after the last one (checks)."""
        try:
            times = sorted(f.stat().st_mtime for f in (self.tmp / "out").glob("*.png")
                           if f.name not in blacks)
            if len(times) > 10 and not self.lane:
                self.up_info = (f"start {times[0] - t_up0:.0f}s, "
                                f"{(len(times) - 1) / max(1e-6, times[-1] - times[0]):.2f} "
                                f"frames/s, after {time.time() - times[-1]:.0f}s")
                self.gpu_info = watch.summary() if watch else ""
            if blacks and not self.lane:
                self.up_info += f"{', ' if self.up_info else ''}{len(blacks)} black frames not upscaled"
        except OSError:
            pass

    def _encode_inputs(self):
        a, tmp = self.a, self.tmp
        ins = ["-framerate", a.fps, "-i", pngs(tmp / "out")]
        if a.ai_blend < 1:
            ins += ["-framerate", a.fps, "-i", pngs(tmp / "in")]
        return ins

    def _encode_graph(self, n_in):
        """The filters from the upscaled frames to the encoder: the blend with a plain upscale
        (--ai-blend), then the post filters (smoothing warmed up, colour, resize, sharpen)."""
        a = self.a
        if a.ai_blend < 1:
            # mix the AI frames with a plain upscale of the same input frames, so frames where
            # the model adds detail and frames where it doesn't look less different
            if getattr(a, "detail_blend", False):
                # crowds, grass, gravel (dense small detail): less of the AI there, which paints
                # it flat and makes it shimmer; the rest keeps --ai-blend (see detail_blend_mask)
                graph = (f"[1:v]split=2[p1][p2];[p1]scale=iw*{a.scale}:ih*{a.scale}:"
                         f"flags=lanczos,format=gbrp[plain];[p2]{detail_blend_mask(a)},"
                         f"scale=iw*{a.scale}:ih*{a.scale}:flags=bilinear,format=gbrp[mask];"
                         f"[0:v]format=gbrp[ai];[ai][plain][mask]maskedmerge")
            else:
                graph = (f"[1:v]scale=iw*{a.scale}:ih*{a.scale}:flags=lanczos,"
                         f"format=gbrp[plain];[0:v]format=gbrp[ai];"
                         f"[ai][plain]blend=all_mode=normal:all_opacity={a.ai_blend}")
        else:
            graph = "[0:v]null"
        k = min(12, n_in - 1)
        if a.smooth > 0 and k > 0:
            # the temporal smoothing would start "cold" on every chunk's first frame (a visible
            # sharp-to-soft breath every chunk); warm it up on the next k frames played
            # backwards, then cut those warm-up frames off again
            graph += (f",split[wa][wb];[wa]trim=start_frame=1:end_frame={k + 1},reverse[wr];"
                      f"[wr][wb]concat=n=2:v=1:a=0,{postfilter(a)},"
                      f"trim=start_frame={k},setpts=PTS-STARTPTS")
        else:
            graph += "," + postfilter(a)
        return graph

    def write_stab(self):
        """--stabilize: this chunk's piece of the camera-motion file, in its tmp folder, where
        its ffmpeg then runs (so the file name in the filter needs no escaping: drive letters,
        quotes). Returns that folder, or None (no --stabilize)."""
        if not self.stab:
            return None
        self.tmp.mkdir(parents=True, exist_ok=True)
        write_stab_slice(self.a.stab_file, self.a.stab_index, self.stab[0], self.stab[1],
                         self.tmp / "stab.trf")
        return self.tmp

    def fast(self):
        """--fast: filters only, straight into the chunk file. Returns the frame count."""
        a = self.a
        cwd = self.write_stab()
        run(["ffmpeg", "-y", "-v", "error", *self.src,
             "-vf", self.pre + "," + postfilter(a), "-frames:v", self.expected,
             *encode_args(a), self.part], cwd=cwd)
        try:
            got = count_frames(self.part)
        except subprocess.CalledProcessError:   # encoder got no frames: file is unreadable
            got = 0
        if 0 < got < self.expected and not self.last:
            status_line()
            print(gap_message(self.idx, self.expected - got, self.expected), flush=True)
            run(["ffmpeg", "-y", "-v", "error", *self.src, "-vf", self.pre
                 + ",tpad=stop_mode=clone:stop=-1," + postfilter(a), "-frames:v", self.expected,
                 *encode_args(a), self.part], cwd=cwd)
        return got

    def finish(self):
        """Restore the faces (--faces), encode (unless --fast did already) and keep the chunk.
        Returns (path, warning)."""
        t_enc = time.time()
        if self.encode:
            try:
                if self.a.faces:
                    self.restore_faces()
                run(self.encode)
            except BaseException:
                self.clear_frames()
                raise
        self.enc_secs = time.time() - t_enc
        n = count_frames(self.part)
        flush_to_disk(self.part)
        replace_file(self.part, self.out)
        shutil.rmtree(self.tmp, ignore_errors=True)
        return self.out, (f"  WARNING: chunk {self.idx} has {n} frames, expected {self.expected} "
                          "(can cause small sync drift)"
                          if n != self.expected and not self.last else None)


    def stopping(self):
        """The run is stopping (Ctrl+C, a failure), or for a helper GPU, it is to stop."""
        return self.a.stopping.is_set() or bool(self.lane) and lane_stopped(self.a, self.lane)

    def restore_faces(self):
        """--faces: the faces of the upscaled frames restored by a worker process (see
        faces_worker_main), one at a time over all GPUs (FACE_LOCK). The worker only writes the
        frames it changed, into tmp/faces; once it has finished they are moved over the ones in
        tmp/out. So a worker that crashes leaves tmp/out as the upscaler wrote it, and a retry
        never restores a frame twice. One retry with a fresh process, on the processor (the GPU
        may be short of memory next to the upscalers; not on a helper GPU, whose chunk the main
        GPU redoes), then the run stops: it resumes with this chunk."""
        a, tmp, label = self.a, self.tmp, self.label
        dest, key = tmp / "faces", self.lane or "faces"
        self.stage = "waiting to restore the faces"
        try:
            while not FACE_LOCK.acquire(timeout=1):
                if self.stopping():
                    raise RuntimeError("stopped")
            try:
                self.stage = "starting the face restoration"
                t0 = time.time()
                for attempt in (1, 2):
                    shutil.rmtree(dest, ignore_errors=True)
                    dest.mkdir()
                    cpu = attempt == 2 and a.face_provider != "CPUExecutionProvider"
                    rc, lines = self.face_worker(dest, key, cpu)
                    done = [x.split() for x in lines if x.startswith("DONE ")]
                    if rc == 0 and done:
                        break
                    if self.stopping() or rc in STOPPED or rc == -signal.SIGINT:
                        raise RuntimeError("stopped")
                    why = ("it stopped responding (no progress for 10 minutes)" if rc == "hung"
                           else f"exit code {rc}" if rc else "it didn't report its result")
                    tail = [x.strip() for x in lines
                            if x.strip() and not re.match(r"(detect|restore) \d+ \d+$", x)]
                    if self.lane:
                        raise RuntimeError(f"the face restoration failed ({why}"
                                           + (f": {tail[-1][:100]}" if tail else "") + ")")
                    if attempt == 2:
                        # (--all can't take --work: a movie there starts over in its own folder)
                        fresh = (f"delete the folder '{self.tmp.parent.resolve()}' to start "
                                 "this movie fresh" if os.environ.get("DVD_UPSCALE_QUEUE") else
                                 "start that in a new --work folder")
                        # (the GPU's memory only where it runs on the GPU: a run of the same
                        # command tries the GPU first again)
                        gpu_hint = "" if a.face_provider == "CPUExecutionProvider" else (
                            "if the GPU is short of memory: run the same command again with "
                            "--gpu-jobs 1 and/or --gpu-threads 2; it keeps the finished "
                            "chunks. ")
                        raise RuntimeError(
                            f"the face restoration failed twice on {label} ({why}). Its output "
                            "(end):\n  " + "\n  ".join(tail[-20:]) + "\n(" + gpu_hint
                            + f"--no-faces upscales without it (for a movie already started: delete its "
                            f"work folder first): {fresh})")
                    status_line()
                    print(f"  {label}: the face restoration failed ({why}) - trying it once "
                          + ("more on the processor (slower; the GPU may be short of memory "
                             "next to the upscalers)." if a.face_provider != "CPUExecutionProvider"
                             else "more."), flush=True)
                took = time.time() - t0
                if not getattr(a, "face_time_shown", False):
                    a.face_time_shown = True
                    _, faces, changed, provider = done[-1][:4]
                    self.face_msg = (
                        f"  Faces: {short_time(took)} for {label} ({faces} faces in {changed} "
                        f"of its frames, on {provider}). This runs while the next chunk is "
                        "upscaled, and slows the movie down only where it takes longer.")
                    if self.note:           # (a helper GPU's messages go to the main one)
                        self.note(self.face_msg)
                        self.face_msg = None
            finally:
                FACE_LOCK.release()
            for f in sorted(dest.glob("*.png")):
                replace_file(f, tmp / "out" / f.name)
            shutil.rmtree(dest, ignore_errors=True)
        finally:
            if not self.lane:
                LANE_STATUS.pop(key, None)
        self.stage = "encoding"

    def face_worker(self, dest, key, cpu=False):
        """One run of the face worker on this chunk's frames (cpu: on the processor), its
        progress shown (main GPU: on the progress line and while the main GPU waits for it; a
        helper GPU: as its status). Killed if the run stops, or if it makes no progress for 10
        minutes. Returns (exit code, or "hung", its output lines)."""
        a, tmp = self.a, self.tmp
        log_path = tmp / "faces_log.txt"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--faces-worker",
               "--frames", str(tmp / "out"), "--dest", str(dest), "--models", a.face_models_path,
               "--model", a.face_model, "--strength", repr(float(a.faces)),
               "--fidelity", repr(float(FACE_FIDELITY)), "--scale", str(a.scale),
               "--ai-blend", repr(float(a.ai_blend)), *(["--cpu"] if cpu else [])]
        steps = {"detect": "finding the faces", "restore": "restoring the faces"}
        with open(log_path, "wb") as log:
            # (its input is a pipe this run holds open: see watch_parent)
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, PYTHONUNBUFFERED="1"))
            FACE_PROCS.append(p)
            try:
                # watchdog (as for the upscaler), counted in 1 s waits rather than clock time
                # so a laptop that slept in between isn't taken for a stall: seconds since the
                # last frame found or face restored (no fixed limit on a chunk or a frame: a slow
                # processor is fine)
                offset, pending, quiet, done, rc = 0, "", 0, False, None
                while True:
                    try:
                        p.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    if self.stopping():
                        raise RuntimeError("stopped")
                    offset, pending, lines = _read_log_updates(log_path, offset, pending)
                    quiet += 1
                    for line in lines:
                        done = done or line.startswith("DONE ")
                        m = re.match(r"(detect|restore) (\d+) (\d+)$", line.strip())
                        if m or line.startswith("DONE "):
                            quiet = 0
                        if m:
                            self.stage = f"{steps[m[1]]}, frame {m[2]} of {m[3]}"
                            LANE_STATUS[key] = (f"{self.lane}: {self.label} faces {m[2]}/{m[3]}"
                                                if self.lane else
                                                f"faces of {self.label}: {m[2]}/{m[3]}")
                    if done and quiet > 60:
                        # every frame written (each one whole), but it doesn't exit
                        # (onnxruntime's teardown?): its result counts
                        rc = 0
                        break
                    if quiet > 600:         # a GPU driver hang? A failed attempt
                        rc = "hung"
                        break
            finally:
                if p.poll() is None:        # stopping, or an error here: don't leave it running
                    p.kill()
                    p.wait()
                FACE_PROCS.remove(p)
                try:
                    p.stdin.close()
                except OSError:
                    pass
        return (p.returncode if rc is None else rc,
                log_path.read_bytes().decode("utf-8", "replace").splitlines())


def stop_face_workers():
    """At exit: a face worker still running (its chunk's thread didn't get to stop it) is
    stopped, never left running on its own."""
    for p in list(FACE_PROCS):
        try:
            p.kill()
        except OSError:
            pass


atexit.register(stop_face_workers)


class Background:
    """One step running in another thread; wait() hands back its result or raises its error."""

    def __init__(self, fn):
        self.box, self.done = {}, threading.Event()

        def work_():
            try:
                self.box["result"] = fn()
            except BaseException as e:
                self.box["error"] = e
            finally:
                self.done.set()
        self.thread = threading.Thread(target=work_, daemon=True)
        self.thread.start()

    def wait(self):
        # (a timeout keeps Ctrl+C working while it waits. An Event, not thread.join(): a join
        # that Ctrl+C interrupts leaves the thread looking ended while it still runs - Python
        # 3.11 - and the wait after Ctrl+C then let the run exit before the step had stopped)
        while not self.done.wait(0.5):
            pass
        if "error" in self.box:
            raise self.box["error"]
        return self.box.get("result")


def probe_or_exit(src):
    try:
        return probe(src)
    except subprocess.CalledProcessError:
        sys.exit(f"Could not read '{src}' (see the ffprobe error above)")
    except subprocess.TimeoutExpired:
        sys.exit(f"Could not read '{src}': ffprobe got no answer from it in "
                 f"{LONG_PROBE_TIMEOUT} s (a damaged file, or a drive that stopped answering?)")
    except (ValueError, KeyError) as e:
        sys.exit(f"Could not read '{src}': {e}")


def vhs_setup(a, find):
    """--type vhs: check its options, then tell movie tape from camcorder tape (that decision
    picks the model and settings, so it comes before everything else). Returns the probe info."""
    for t in ("ffmpeg", "ffprobe"):
        if not find(t):
            sys.exit(f"Missing tool: {t}")
    try:
        a.mask = tuple(int(x) for x in a.mask.split(":"))
    except ValueError:
        a.mask = ()
    if len(a.mask) != 4 or min(a.mask) < 0:
        sys.exit("--mask needs four numbers left:right:top:bottom, e.g. 8:8:2:12")
    a.chroma_shift = None
    if a.chroma_delay not in ("auto", "off"):
        try:
            dx, dy = (int(v) for v in a.chroma_delay.split(":"))
        except ValueError:
            sys.exit("--chroma-delay must be auto, off or right:down in pixels, e.g. 3:1")
        a.chroma_shift = f"cbh={-dx}:crh={-dx}:cbv={-dy}:crv={-dy}"
    info = probe_or_exit(a.input)
    # the tape's frame rate: NTSC 29.97 or PAL 25 (a file's slightly-off average rate snapped)
    fin = info["fps"]
    a.vhs_fin = (Fraction(30000, 1001) if abs(fin - Fraction(30000, 1001)) < 1 else
                 Fraction(25) if abs(fin - 25) < 1 else
                 fin.limit_denominator(1001) if fin > 0 else Fraction(30000, 1001))
    if a.telecine:
        a.mode = "telecine"
    a.vhs_auto, a.vhs_dedup = a.mode == "auto", None
    if a.mode == "progressive":
        a.parity, a.vhs_detail = "tff", "forced"
    else:
        mode, a.parity, detail = vhs_detect(a, info)
        old = previous_settings(a)
        if a.mode == "auto" and old and old.get("mode") in ("telecine", "interlaced") \
                and old["mode"] != mode:
            # a resumed tape keeps the decision it started with (the detection was improved
            # since: cartoon tapes; the finished chunks must match the rest)
            detail += (f"; {old['mode']} kept from the earlier run in "
                       f"'{a.work + ('_test' if a.test else '')}', the check now says {mode}")
            mode = old["mode"]
        if mode != "progressive" and a.vhs_fin > 47:
            # interlaced frames at 50/60 frames/s: a capture app (e.g. OBS set to 60 fps) that
            # stored every tape frame twice. Use each once, then it is the 25/29.97 tape it was
            # (else camcorder tapes would come out at 100/120 fps)
            half = a.vhs_fin / 2
            a.vhs_fin = (Fraction(30000, 1001) if abs(half - Fraction(30000, 1001)) < 1 else
                         Fraction(25) if abs(half - 25) < 1 else half)
            a.vhs_dedup = f"fps={a.vhs_fin.numerator}/{a.vhs_fin.denominator}"
            detail += (f"; the file repeats every tape frame ({float(fin):.2f} fps), so every "
                       f"second one is used ({float(a.vhs_fin):.3f} fps). Capture at "
                       f"{float(a.vhs_fin):.2f} fps next time")
        a.vhs_detail = detail if a.vhs_auto else f"forced; the detection says {mode}, {detail}"
        a.mode = mode if a.vhs_auto else a.mode
    return info


# ---- what's in the file (--type auto) ------------------------------------------------------------
TYPE_NAMES = {"anime": "anime / drawn animation", "live": "live action",
              "cgi": "3D animation (CGI)", "vhs": "VHS tape"}

# ---- auto-detection, content, look at single pictures: flat colour areas and ink outlines (drawn) vs soft shading
# ---- model (fitted on the training split, see REPORT.md) ---------------------------------------
_cs_W_ANIME = (3.2616312366286953, 2.795601085467864, 0.07764657454479618)   # logit(eshare), logit(eshare2x), noise
_cs_B_ANIME = 9.154252224821379
_cs_W_CGI = (0.5447004345669803, -0.4198416396817829)                        # sat_std, logit(eshare)
_cs_B_CGI = -5.499787115874285

_cs_N_SAMPLES = 32
_cs_BW_INK, _cs_BW_INK_HIGH = 1.2, 1.5               # dark-line ratio; non-anime training clips: max 1.19 (Caminandes 2)
_cs_NPL = 7                                  # planes per sample: Y, median(Y), Laplacian, black/white top-hat, U, V


# ---- shared by the three detectors below (pictures, motion, VHS) ----------------------------

_NO_WINDOW = ({"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
              if os.name == "nt" else {})      # no console window flashing for every ffmpeg call


def _quiet_run(cmd, timeout=60):
    """(exit code, stdout bytes) of a command, stderr discarded; (-1, b"") if it couldn't run or
    took longer than timeout. Never raises."""
    try:
        p = subprocess.run([str(c) for c in cmd], stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           timeout=timeout, **_NO_WINDOW)
        return p.returncode, p.stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return -1, b""


def _median(v, empty=None):
    """The median of v (the mean of the middle two for an even count); `empty` for none."""
    v = list(v)
    return statistics.median(v) if v else empty


def _cs_logit(x):
    x = min(max(x, 0.005), 0.95)
    return math.log(x / (1.0 - x))


def _cs_sig(z):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


def _cs_num(x):
    try:
        if x in (None, "", "N/A"):
            return None
        if isinstance(x, str) and ":" in x:          # Matroska DURATION tag 01:23:45.678
            s = 0.0
            for part in x.split(":"):
                s = s * 60 + float(part)
            return s
        if isinstance(x, str) and "/" in x:
            a, b = x.split("/")
            return float(a) / float(b) if float(b) else None
        return float(x)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def _cs_probe(path, ffprobe):
    rc, out = _quiet_run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                    "stream=width,height,duration:stream_tags:format=duration,size,bit_rate",
                    "-of", "json", path])
    try:
        j = json.loads(out.decode("utf-8", "replace") or "{}")
    except ValueError:
        j = {}
    st = (j.get("streams") or [{}])[0]
    fm = j.get("format") or {}
    w, h = int(st.get("width") or 0), int(st.get("height") or 0)
    dur = _cs_num(fm.get("duration")) or _cs_num(st.get("duration"))
    if not dur:
        tag = next((v for k, v in (st.get("tags") or {}).items() if k.upper().startswith("DURATION")), None)
        dur = _cs_num(tag)
    if not dur:                                     # no duration in the container: estimate from size/rate
        size = _cs_num(fm.get("size")) or (os.path.getsize(path) if os.path.isfile(path) else None)
        rate = _cs_num(fm.get("bit_rate"))
        if not rate:
            rc, pk = _quiet_run([ffprobe, "-v", "error", "-read_intervals", "%+20", "-show_entries",
                           "packet=pts_time,size", "-of", "csv=p=0", path])
            tot, ts = 0, []
            for line in pk.decode("ascii", "replace").splitlines():
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    t, s = _cs_num(parts[0]), _cs_num(parts[1])
                    if s:
                        tot += s
                    if t is not None:
                        ts.append(t)
            if ts and max(ts) - min(ts) > 1:
                rate = 8.0 * tot / (max(ts) - min(ts))
        if size and rate:
            dur = 0.97 * 8.0 * size / rate
    return w, h, dur


def _cs_geometry(w, h):
    """Analysis grid: one field, half horizontal resolution of a 720-wide SD frame."""
    pre = ""
    if h > 600 or w > 1024:                         # HD / odd sources: bring to SD-like 720x480 first
        pre = "scale=720:480:flags=area,"
        w, h = 720, 480
    W = max(64, (w // 2) // 2 * 2)
    H = max(48, ((h + 1) // 2) // 2 * 2)
    return pre, W, H


def _cs_vf(pre, W, H):
    return (f"[0:v:0]{pre}field=top,scale={W}:{H}:flags=area,format=yuv444p,"
            f"extractplanes=y+u+v[y][u][v];[y]split=7[y1][y2][y3][y4][y5][y6][y7];[y2]median=radius=1[m];"
            f"[y3]convolution=0m='1 -2 1 -2 4 -2 1 -2 1':0rdiv=1:0bias=128[l];"
            f"[y4]dilation,erosion[c];[c][y6]blend=all_mode=difference[bth];"
            f"[y5]erosion,dilation[o];[y7][o]blend=all_mode=difference[wth];"
            f"[y1][m][l][bth][wth][u][v]vstack=inputs=7[out]")


def _cs_grab(ffmpeg, path, t, pre, W, H, keyonly=True, count=1, select=None, limit=None):
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1"]
    if keyonly:
        cmd += ["-skip_frame", "nokey"]
    if t is not None:
        cmd += ["-ss", f"{t:.3f}"]
    if limit:
        cmd += ["-t", f"{limit:.0f}"]
    cmd += ["-i", path, "-filter_threads", "1", "-filter_complex_threads", "1"]
    fc = _cs_vf(pre, W, H)
    if select:
        fc = fc.replace("[0:v:0]", f"[0:v:0]{select},")
    cmd += ["-filter_complex", fc, "-map", "[out]", "-an", "-sn", "-dn", "-frames:v", str(count),
            "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    rc, out = _quiet_run(cmd, timeout=120)
    n = W * H * _cs_NPL
    return [out[i:i + n] for i in range(0, len(out) - n + 1, n)]


def _cs_find_bars(planes, W, H):
    """Rows/columns (from each edge) that are dark (mean < 30) in >= 90% of the non-dark samples."""
    frames = [p[:W * H] for p in planes]
    ok = [f for f in frames if sum(f) >= 30 * W * H] or frames
    rdark = [0] * H
    cdark = [0] * W
    for f in ok:
        for r in range(H):
            if sum(f[r * W:(r + 1) * W]) < 30 * W:
                rdark[r] += 1
        for c in range(W):
            if sum(f[c::W]) < 30 * H:
                cdark[c] += 1

    def run(a, n):
        k = 0
        while k < n // 3 and a[k] >= 0.9 * len(ok):
            k += 1
        return k
    t, b = run(rdark, H), run(rdark[::-1], H)
    l, r = run(cdark, W), run(cdark[::-1], W)
    mh, mw = max(2, round(H * 0.03)), max(2, round(W * 0.02))
    return t + mh, b + mh, l + mw, r + mw


def _cs_absdiff(a, b):
    return list(map(abs, map(sub, a, b)))


def _cs_frame_features(buf, W, H, crop):
    t, b, l, r = crop
    rows = range(t, H - b)
    w = W - l - r
    h = len(rows)
    if w < 16 or h < 16:
        return None
    P = W * H

    def plane(k):
        base = k * P
        return [buf[base + y * W + l: base + y * W + l + w] for y in rows]
    Y, M, L, BT, WT, U, V = (plane(k) for k in range(_cs_NPL))
    n = w * h
    ysum = sum(sum(row) for row in Y)
    dark_tab = bytes(1 if v < 35 else 0 for v in range(256))
    dark = sum(row.translate(dark_tab).count(1) for row in Y) / n
    f = {"meanY": ysum / n, "dark": dark}
    # dark thin lines (black top-hat > 25) vs light thin lines (white top-hat > 25): ink outlines are dark
    th_tab = bytes(1 if v > 25 else 0 for v in range(256))
    bt = sum(row.translate(th_tab).count(1) for row in BT) / n
    wt = sum(row.translate(th_tab).count(1) for row in WT) / n
    f["bw_ratio"] = bt / (wt + 0.002)

    # gradients g = max(|dx|, |dy|) over (h-1) x (w-1) for the median image M and the original Y
    def grad_hist(img):
        gs = []
        for y in range(h - 1):
            a, c = img[y], img[y + 1]
            gx = _cs_absdiff(a[1:], a[:-1])
            gy = _cs_absdiff(c[:-1], a[:-1])
            g = bytes(map(max, gx, gy))
            gs.append(g)
        return gs
    gm = grad_hist(M)
    # histogram via bytes.count on a class table (fast in C)
    cls = bytes((2 if v >= 40 else 1 if 6 < v < 24 else 3 if v <= 2 else 0) for v in range(256))
    cs = cm = cf = 0
    for g in gm:
        tg = g.translate(cls)
        cs += tg.count(2)
        cm += tg.count(1)
        cf += tg.count(3)
    ng = (h - 1) * (w - 1)
    s_, m_ = cs / ng, cm / ng
    f["m_eshare"] = s_ / (s_ + m_ + 1e-4)
    f["m_flat2"] = cf / ng

    # 2x2 block sums of M (exact integer version of the 2x2 mean), then the same edge share
    h2, w2 = h // 2, w // 2
    S = []
    for y in range(h2):
        rr = list(map(int.__add__, M[2 * y][:2 * w2], M[2 * y + 1][:2 * w2]))
        S.append(list(map(int.__add__, rr[0::2], rr[1::2])))
    cs2 = cm2 = 0
    for y in range(h2 - 1):
        a, c = S[y], S[y + 1]
        gx = _cs_absdiff(a[1:], a[:-1])
        gy = _cs_absdiff(c[:-1], a[:-1])
        for d in map(max, gx, gy):
            if d >= 120:
                cs2 += 1
            elif 16 < d < 80:
                cm2 += 1
    ng2 = max(1, (h2 - 1) * (w2 - 1))
    s2, m2 = cs2 / ng2, cm2 / ng2
    f["m_deshare"] = s2 / (s2 + m2 + 1e-4)

    # noise: mean |Laplacian| over the low-gradient half of the original image
    go = grad_hist(Y)                                # rows 0..h-2, cols 0..w-2
    cnt = Counter(b"".join(g[:w - 2] for g in go[:h - 2]))
    hist = [cnt.get(v, 0) for v in range(256)]
    tot = (h - 2) * (w - 2)
    thr = _cs_percentile50(hist, tot)
    lsum = lcnt = 0
    lab = [abs(v - 128) for v in range(256)]
    for y in range(h - 2):
        g = go[y][:w - 2]
        lrow = L[y + 1][1:w - 1]
        for gv, lv in zip(g, lrow):
            if gv <= thr:
                lsum += lab[lv]
                lcnt += 1
    f["noise"] = lsum / lcnt if lcnt else 0.0

    # saturation spread: std of chroma magnitude
    s1 = s2_ = 0.0
    sq = _cs_SQ
    for u, v in zip(U, V):
        for cu, cv in zip(u, v):
            c = sq[cu][cv]
            s1 += c
            s2_ += c * c
    mean = s1 / n
    f["sat_std"] = math.sqrt(max(0.0, s2_ / n - mean * mean))
    return f


_cs_SQ = [[math.sqrt((u - 128) ** 2 + (v - 128) ** 2) for v in range(256)] for u in range(256)]


def _cs_percentile50(hist, tot):
    """numpy.percentile(x, 50) (linear interpolation) from an integer histogram."""
    pos = (tot - 1) * 0.5
    lo_i, hi_i = int(math.floor(pos)), int(math.ceil(pos))
    acc = 0
    lo_v = hi_v = None
    for v, c in enumerate(hist):
        if lo_v is None and acc + c > lo_i:
            lo_v = v
        if hi_v is None and acc + c > hi_i:
            hi_v = v
            break
        acc += c
    if lo_v is None:
        return 0.0
    if hi_v is None:
        hi_v = lo_v
    return lo_v + (hi_v - lo_v) * (pos - lo_i)


def _cs_p_anime(f):
    return _cs_sig(_cs_W_ANIME[0] * _cs_logit(f["m_eshare"]) + _cs_W_ANIME[1] * _cs_logit(f["m_deshare"])
                + _cs_W_ANIME[2] * min(max(f["noise"], 0.0), 20.0) + _cs_B_ANIME)


def _cs_p_cgi(f):
    return _cs_sig(_cs_W_CGI[0] * f["sat_std"] + _cs_W_CGI[1] * _cs_logit(f["m_eshare"]) + _cs_B_CGI)


def _cs_detect_content(path, ffmpeg="ffmpeg", ffprobe="ffprobe", info=None):
    t0 = time.time()
    path = str(path)
    w = h = dur = None
    if info:
        w, h, dur = info.get("w"), info.get("h"), info.get("duration")
    if not (w and h and dur):
        pw, ph, pd = _cs_probe(path, ffprobe)
        w, h, dur = w or pw, h or ph, dur or pd

    def result(content, conf, reasons, scores):
        return dict(content=content, confidence=conf, reasons=reasons, scores=scores,
                    seconds=round(time.time() - t0, 2))
    if not (w and h):
        return result("live", "low", "could not read the video stream (assuming live action)", {})
    pre, W, H = _cs_geometry(int(w), int(h))

    planes, seen = [], set()
    if dur and dur > 0:
        n = _cs_N_SAMPLES if dur >= 20 else max(8, int(dur * 1.5))
        keyonly, fails = True, 0
        for i in range(n):
            t = dur * (0.05 + 0.90 * (i + 0.5) / n)
            got = _cs_grab(ffmpeg, path, t, pre, W, H, keyonly)
            if not got and keyonly:
                fails += 1
                if fails >= 3 and not planes:       # no flagged key frames? decode normally from now on
                    keyonly = False
                    got = _cs_grab(ffmpeg, path, t, pre, W, H, keyonly)
            for g in got:
                k = hash(g)
                if k not in seen:
                    seen.add(k)
                    planes.append(g)
        if len(planes) < 8 and dur < 600:           # short file or long GOP: decode exact positions
            for i in range(16):
                t = dur * (0.05 + 0.90 * (i + 0.5) / 16)
                for g in _cs_grab(ffmpeg, path, t, pre, W, H, keyonly=False):
                    k = hash(g)
                    if k not in seen:
                        seen.add(k)
                        planes.append(g)
    if len(planes) < 8:                             # no duration / seeking impossible: read sequentially
        # key frames only, one every 36 s of the first 20 minutes after the first minute (logos);
        # a short file gets one every 2 s from the start
        for start, step, lim in ((60.0, 36, 1200), (None, 2, 600)):
            sel = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{step})'"
            for g in _cs_grab(ffmpeg, path, start, pre, W, H, keyonly=True, count=_cs_N_SAMPLES, select=sel, limit=lim):
                k = hash(g)
                if k not in seen:
                    seen.add(k)
                    planes.append(g)
            if len(planes) >= 8:
                break
    if not planes:
        return result("live", "low", "no frames could be decoded (assuming live action)", {})

    crop = _cs_find_bars(planes, W, H)
    feats = [f for f in (_cs_frame_features(p, W, H, crop) for p in planes) if f]
    valid = [f for f in feats if f["meanY"] >= 30 and f["dark"] < 0.6]
    use = valid or feats
    if not use:
        return result("live", "low", "picture area too small after cropping (assuming live action)", {})
    med = {k: statistics.median([f[k] for f in use]) for k in ("m_eshare", "m_deshare", "noise", "sat_std", "m_flat2", "bw_ratio")}
    pa, pc = _cs_p_anime(med), _cs_p_cgi(med)
    agree = sum(1 for f in use if _cs_p_anime(f) >= 0.5) / len(use)
    inked = sum(1 for f in use if f["bw_ratio"] >= _cs_BW_INK) / len(use)
    scores = dict(p_anime=round(pa, 3), p_cgi_vs_live=round(pc, 3), edge_share=round(med["m_eshare"], 3),
                  edge_share_2x=round(med["m_deshare"], 3), noise=round(med["noise"], 2),
                  sat_spread=round(med["sat_std"], 2), flat_share=round(med["m_flat2"], 3),
                  dark_line_ratio=round(med["bw_ratio"], 2), inked_frames=round(inked, 2),
                  drawn_frames=round(agree, 2), frames_used=len(use), frames_sampled=len(planes),
                  crop_tblr=list(crop), grid=f"{W}x{H}")
    ev = (f"hard edges {med['m_eshare']:.0%} of detail pixels (2x: {med['m_deshare']:.0%}), "
          f"dark-line ratio {med['bw_ratio']:.2f}, "
          f"flat areas {med['m_flat2']:.0%}, noise {med['noise']:.1f}, colour spread {med['sat_std']:.1f}, "
          f"{agree:.0%} of {len(use)} frames look drawn")
    if len(valid) < 4:
        ev += "; few usable frames (mostly dark)"

    few = len(valid) < 6
    if pa >= 0.55:
        conf = "high" if (pa >= 0.8 and agree >= 0.5 and len(valid) >= 8) else "medium"
        return result("anime", conf, f"drawing-like: {ev}", scores)
    if med["bw_ratio"] >= _cs_BW_INK and len(use) >= 6:
        # second cue: thin DARK lines clearly outnumber thin light lines = ink outlines (cel anime on a
        # soft DVD transfer whose edges are too blurred for the edge-share cue)
        conf = "high" if (med["bw_ratio"] >= _cs_BW_INK_HIGH and len(valid) >= 8) else "medium"
        return result("anime", conf, f"dark ink outlines: {ev}", scores)
    label = "cgi" if pc >= 0.5 else "live"
    if pa >= 0.4 or few:
        return result("live", "low", f"undecided (P(anime) {pa:.2f}): {ev}", scores)
    if pa < 0.15 and abs(pc - 0.5) >= 0.3:
        conf = "high"
    elif pa < 0.3:
        conf = "medium"
    else:
        conf = "low"
    kind = "colourful clean render" if label == "cgi" else "photographic/muted"
    return result(label, conf, f"not drawn, {kind}: {ev}", scores)

# ---- auto-detection, content, look at motion: drawings held for 2-3 frames (animation) vs a new picture every frame, film grain
_ct_W, _ct_H = 240, 120          # field picture size used for analysis
_ct_BX, _ct_BY = 6, 6            # block size of the difference map
_ct_BW, _ct_BH = _ct_W // _ct_BX, _ct_H // _ct_BY
_ct_N_BURSTS = 36
_ct_BURST_FRAMES = 23        # frames decoded per burst (first 2 are dropped: open-GOP B frames)
_ct_SKIP = 2

def _ct_num(x):
    try:
        if isinstance(x, str) and "/" in x:
            a, b = x.split("/", 1)
            return float(a) / float(b) if float(b) else 0.0
        v = float(x)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _ct_hms(x):
    try:
        secs = 0.0
        for part in str(x).split(":"):
            secs = secs * 60 + float(part)
        return secs if secs > 0 and math.isfinite(secs) else 0.0
    except ValueError:
        return 0.0


def _ct_probe(path, ffprobe):
    """Candidate video durations (best first), nominal fps, container start time."""
    out = _quiet_run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=avg_frame_rate,r_frame_rate,duration:stream_tags:format=duration,start_time",
                "-of", "json", path])[1]
    try:
        j = json.loads(out.decode("utf-8", "replace") or "{}")
    except ValueError:
        j = {}
    s = (j.get("streams") or [{}])[0]
    fmt = j.get("format", {})
    start = _ct_num(fmt.get("start_time"))
    fd = _ct_hms(fmt.get("duration"))
    tag = 0.0
    for k, v in (s.get("tags") or {}).items():
        if k.upper().startswith("DURATION"):
            tag = _ct_hms(v)
    # the container value can include a start offset; a copied DURATION tag can be stale
    cands = [fd, fd - start if start > 1 and fd > start else 0.0, _ct_hms(s.get("duration")), tag]
    fps = _ct_num(s.get("avg_frame_rate")) or _ct_num(s.get("r_frame_rate"))
    return [d for d in cands if d > 1.0], fps, start


def _ct_coded_rate(path, t, ffprobe):
    """Frames actually stored per second (packets), around time t. Soft-telecined DVD video is
    flagged 29.97 but stores 23.976 frames with repeat-field flags: decoders output 23.976."""
    out = _quiet_run([ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals",
                "%.3f%%+4" % max(0.0, t), "-show_entries", "packet=pts_time", "-of", "csv=p=0", path])[1]
    ts = sorted(_ct_num(x) for x in out.decode("ascii", "replace").split() if x and x != "N/A")
    if len(ts) < 20 or ts[-1] - ts[0] <= 0.5:
        return 0.0
    return (len(ts) - 1) / (ts[-1] - ts[0])


def _ct_has_frame(path, t, ffmpeg):
    out = _quiet_run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", "%.3f" % t,
                "-i", path, "-an", "-sn", "-dn", "-frames:v", "1", "-vf", "scale=16:16",
                "-f", "rawvideo", "-pix_fmt", "gray", "-"], timeout=60)[1]
    return len(out) >= 256


def _ct_find_end(path, ffmpeg):
    """Length of a file whose container has no duration: probe with accurate seeks."""
    lo, hi = 0.0, 60.0
    while hi < 6 * 3600 and _ct_has_frame(path, hi, ffmpeg):
        lo, hi = hi, hi * 2
    for _ in range(7):
        mid = (lo + hi) / 2
        if _ct_has_frame(path, mid, ffmpeg):
            lo = mid
        else:
            hi = mid
    return lo


def _ct_bursts_sequential(path, ffmpeg, n, period=150):
    """Fallback for files that cannot be seeked (no duration/index): one decode from the start,
    keeping BURST_FRAMES consecutive frames out of every `period`."""
    graph = ("[0:v]select='lt(mod(n\\,%d)\\,%d)',field=top,format=gray,scale=%d:%d:flags=area,"
             "split[o][x];[x]tblend=all_mode=difference,scale=%d:%d:flags=area,pad=%d:%d[d];"
             "[o]trim=start_frame=1[o2];[o2][d]vstack" % (period, _ct_BURST_FRAMES, _ct_W, _ct_H, _ct_BW, _ct_BH, _ct_W, _ct_BH))
    out = _quiet_run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
                "-i", path, "-an", "-sn", "-dn", "-filter_complex", graph,
                "-frames:v", str(n * _ct_BURST_FRAMES - 1), "-fps_mode", "passthrough",
                "-f", "rawvideo", "-pix_fmt", "gray", "-"], timeout=300)[1]
    fsz = _ct_W * (_ct_H + _ct_BH)
    frames, diffs = [], []
    for i in range(len(out) // fsz):
        f = out[i * fsz:(i + 1) * fsz]
        frames.append(f[:_ct_W * _ct_H])
        d = f[_ct_W * _ct_H:]
        diffs.append(b"".join(d[r * _ct_W:r * _ct_W + _ct_BW] for r in range(_ct_BH)))
    # output frame i is source frame i+1 of the selection: re-cut into the original bursts
    frames, diffs = [b""] + frames, [b""] + diffs
    return [(frames[i:i + _ct_BURST_FRAMES][_ct_SKIP:], diffs[i:i + _ct_BURST_FRAMES][_ct_SKIP:])
            for i in range(0, len(frames) - _ct_BURST_FRAMES + 1, _ct_BURST_FRAMES)]


def _ct_burst(path, t, ffmpeg, threads=1):
    """Decoded top fields (W*H bytes each) and diff maps (BW*BH bytes each, vs previous field)."""
    graph = ("[0:v]field=top,format=gray,scale=%d:%d:flags=area,split[o][x];"
             "[x]tblend=all_mode=difference,scale=%d:%d:flags=area,pad=%d:%d[d];"
             "[o]trim=start_frame=1[o2];[o2][d]vstack" % (_ct_W, _ct_H, _ct_BW, _ct_BH, _ct_W, _ct_BH))
    out = _quiet_run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", str(threads),
                "-noaccurate_seek", "-ss", "%.3f" % max(0.0, t), "-i", path,
                "-an", "-sn", "-dn", "-filter_complex", graph, "-frames:v", str(_ct_BURST_FRAMES),
                "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"])[1]
    fsz = _ct_W * (_ct_H + _ct_BH)
    frames, diffs = [], []
    for i in range(len(out) // fsz):
        f = out[i * fsz:(i + 1) * fsz]
        frames.append(f[:_ct_W * _ct_H])
        d = f[_ct_W * _ct_H:]
        diffs.append(b"".join(d[r * _ct_W:r * _ct_W + _ct_BW] for r in range(_ct_BH)))
    return frames, diffs


# ---------------------------------------------------------------------------------------------
def _ct_crop(frames):
    """Black bars from the sampled pictures: (x0, x1, y0, y1) in W x H field pixels."""
    rows = [0] * _ct_H
    cols = [0] * _ct_W
    n = 0
    for f in frames:
        if sum(f) < 30 * _ct_W * _ct_H:              # black/dark picture tells nothing about bars
            continue
        n += 1
        for y in range(_ct_H):
            if max(f[y * _ct_W:(y + 1) * _ct_W]) > 48:
                rows[y] += 1
        for x in range(_ct_W):
            if max(f[x::_ct_W]) > 48:
                cols[x] += 1
    if n == 0:
        return 0, _ct_W, 0, _ct_H
    lim = 0.10 * n                            # a line is picture if bright in >10% of samples
    y0 = next((y for y in range(_ct_H) if rows[y] > lim), 0)
    y1 = next((y for y in range(_ct_H - 1, -1, -1) if rows[y] > lim), _ct_H - 1) + 1
    x0 = next((x for x in range(_ct_W) if cols[x] > lim), 0)
    x1 = next((x for x in range(_ct_W - 1, -1, -1) if cols[x] > lim), _ct_W - 1) + 1
    # keep away from the soft bar edge
    y0, y1 = (y0 + 2, y1 - 2) if y0 > 0 or y1 < _ct_H else (y0, y1)
    x0, x1 = (x0 + 3, x1 - 3) if x0 > 0 or x1 < _ct_W else (x0, x1)
    if y1 - y0 < _ct_H // 4 or x1 - x0 < _ct_W // 4:
        return 0, _ct_W, 0, _ct_H
    return x0, x1, y0, y1


def _ct_cut(f, c):
    x0, x1, y0, y1 = c
    return b"".join(f[y * _ct_W + x0:y * _ct_W + x1] for y in range(y0, y1)), x1 - x0


def _ct_stats(img):
    """Mean and contrast (5th..95th percentile spread) of a grey picture."""
    m = sum(img) / float(len(img))
    v = sorted(img[::37])
    return m, v[int(0.95 * (len(v) - 1))] - v[int(0.05 * (len(v) - 1))]


def _ct_spatial(img, w):
    """Edge/texture statistics of one cropped grey field picture (240x120 scale)."""
    gx = list(map(abs, map(sub, img[1:], img[:-1])))
    gy = list(map(abs, map(sub, img[w:], img[:-w])))
    hist = Counter(map(max, gx, gy))
    tot = float(sum(hist.values()))
    flat = (hist[0] + hist[1]) / tot
    low = (hist[0] + hist[1] + hist[2] + hist[3]) / tot
    mid = sum(hist[i] for i in range(4, 15)) / tot
    strong = sum(c for k, c in hist.items() if k >= 24) / tot
    # thin dark lines: darker than both horizontal or both vertical neighbours by >= 16
    a = img
    hl = map(sub, map(min, a[w - 1:-w - 1], a[w + 1:-w + 1]), a[w:-w])
    vl = map(sub, map(min, a[:-2 * w], a[2 * w:]), a[w:-w])
    lh = Counter(map(max, hl, vl))
    line = sum(c for k, c in lh.items() if k >= 16) / float(sum(lh.values()))
    # luminance histogram peakiness: share of pixels in the 6 fullest 4-level bins
    ch = Counter(a[::3])
    bins = [0] * 64
    for k, c in ch.items():
        bins[k >> 2] += c
    peak = sum(sorted(bins)[-6:]) / float(len(a[::3]))
    return dict(flat=flat, low=low, mid=mid, strong=strong, line=line, peak=peak)


def _ct_q(v, p):
    s = sorted(v)
    if not s:
        return 0.0
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def _ct_extract(path, ffmpeg="ffmpeg", ffprobe="ffprobe", info=None, n_bursts=_ct_N_BURSTS):
    """Sample the file and return raw per-transition / per-picture measurements."""
    path = os.fspath(path)
    dur = fps = 0.0
    cands, fps, start = [], 0.0, None
    if info:      # dvd_upscale.probe() dict
        cands = [_ct_num(info.get("duration") or 0), _ct_num(info.get("vduration") or 0)]
        fps = _ct_num(info.get("fps") or 0)
        if info.get("cstart") is not None:
            start = _ct_num(info.get("cstart"))
    if not fps or start is None or not any(c > 1.0 for c in cands):
        c2, f2, s2 = _ct_probe(path, ffprobe)
        cands += c2
        fps = fps or f2
        start = s2 if start is None else start
    dur = 0.0
    for c in cands:                        # first candidate that really has video near its end
        if c > 1.0 and _ct_has_frame(path, 0.9 * c, ffmpeg):
            dur = c
            break
    if not dur:
        dur = _ct_find_end(path, ffmpeg)
    if not dur or dur >= 6 * 3600:         # seeking does not work: sample the start in one pass
        return dict(duration=0.0, fps=fps, rate=fps, bursts=_ct_bursts_sequential(path, ffmpeg, 24, 75))
    rate = _ct_coded_rate(path, start + 0.5 * dur, ffprobe) or fps
    span0, span1 = 0.05 * dur, 0.95 * dur
    if dur < 60:                                   # short clips: use (almost) everything
        span0, span1 = min(1.0, 0.02 * dur), max(0.5, dur - 1.0)
    n = max(4, min(n_bursts, int((span1 - span0) / 1.0)))
    bursts = []
    for i in range(n):
        t = span0 + (span1 - span0) * (i + 0.5) / n
        fr, df = _ct_burst(path, t, ffmpeg)
        bursts.append((fr[_ct_SKIP:], df[_ct_SKIP:]))
    return dict(duration=dur, fps=fps, rate=rate, bursts=bursts)


def _ct_measure(raw):
    """Per-transition change strengths and per-picture spatial stats from extract() output."""
    fps = raw["fps"]
    allf = [f for fr, _ in raw["bursts"] for f in fr[::4]]
    c = _ct_crop(allf)
    x0, x1, y0, y1 = c
    bx0, bx1 = -(-x0 // _ct_BX), x1 // _ct_BX
    by0, by1 = -(-y0 // _ct_BY), y1 // _ct_BY
    # skip the outer ring of blocks: VHS head-switching noise, edge jitter, soft bar edges
    bidx = [by * _ct_BW + bx for by in range(by0 + 1, by1 - 1) for bx in range(bx0 + 1, bx1 - 1)]
    nb = len(bidx)
    trans = []      # per burst: list of (s2, med, frac_changed) or None for blank
    spat = []
    for fr, df in raw["bursts"]:
        if len(fr) < 6:
            continue
        info = []
        for f in fr:
            img, w = _ct_cut(f, c)
            info.append(_ct_stats(img))
        tl = []
        for k in range(1, len(fr)):
            (m0, s0), (m1, s1) = info[k - 1], info[k]
            blank = min(m0, m1) < 24 or min(s0, s1) < 12
            d = df[k]
            v = sorted(d[i] for i in bidx)
            med = v[nb // 2]
            top = v[-3] if nb >= 3 else v[-1]
            p90 = v[int(0.9 * (nb - 1))]
            fc = (nb - _bisect(v, med + 6)) / float(nb)
            quiet = v[:_bisect(v, med + 2)]
            nz = sum(quiet) / float(len(quiet)) if quiet else float(med)
            tl.append(dict(blank=blank, med=med, s=top - med, p90=p90 - med, fc=fc, nz=nz))
        trans.append(tl)
        for k in (4, 12):
            if k + 2 < len(fr) and info[k][0] >= 24 and info[k][1] >= 24:
                img, w = _ct_cut(fr[k], c)
                d = _ct_spatial(img, w)
                # same statistics on the mean of 4 consecutive fields: tape/video noise halves,
                # held drawings stay sharp
                acc = list(map(add, map(add, fr[k - 1], fr[k]), map(add, fr[k + 1], fr[k + 2])))
                img4, w = _ct_cut(bytes([x >> 2 for x in acc]), c)
                for key, val in _ct_spatial(img4, w).items():
                    d[key + "4"] = val
                spat.append(d)
    return dict(crop=c, fps=fps, rate=raw.get("rate", fps), trans=trans, spat=spat)


def _ct_decimate(vals, keyfn):
    """Drop the smallest transition in every group of 5 (repeat of 3:2 pulldown)."""
    out = []
    for i in range(0, len(vals) - 4, 5):
        g = vals[i:i + 5]
        j = min(range(5), key=lambda q: keyfn(g[q]))
        out.extend(g[:j] + g[j + 1:])
    return out


_ct_FL_ACTIVE, _ct_FL_MOVER, _ct_FL_HOLD, _ct_FL_ADD = 2.0, 1.8, 1.4, 2.0   # noise-floor multiples (tapes)


def _ct_alt_burst(tl, floor=0.0):
    """Share of transitions that are held drawings between changes (animation on 2s/3s/4s).
    floor: change level of a known exact repeat (3:2 pulldown) = noise floor, 0 if unknown."""
    s = [None if t["blank"] else t["s"] for t in tl]
    v = [x for x in s if x is not None]
    if len(v) < 8:
        return None
    big = _ct_q(v, 0.8)
    if big < max(10.0, _ct_FL_ACTIVE * floor):   # no real motion in this burst: no evidence
        return None
    mover = [x is not None and x >= max(0.5 * big, 8.0, _ct_FL_MOVER * floor) for x in s]
    hold = [x is not None and x <= max(0.25 * big, _ct_FL_HOLD * floor + _ct_FL_ADD) for x in s]
    c = 0
    for k in range(len(s)):
        # only holds that come in a rhythm (another hold within 2 transitions): an isolated
        # repeat is a frame-rate conversion (silent films at 16-20 fps shown at 24/25/30), not
        # drawings held on 2s/3s (found on unseen Chaplin/Melies DVDs)
        if (hold[k] and any(mover[max(0, k - 3):k]) and any(mover[k + 1:k + 4])
                and any(hold[j] for j in range(max(0, k - 2), min(len(s), k + 3)) if j != k)):
            c += 1
    return c / float(len(v))


def _ct_pulldown(tl):
    """~30 fps burst: find the 3:2 pulldown repeat (one transition in five, fixed phase) and drop
    it. Returns (remaining transitions, noise floor = change level of the repeats or 0)."""
    s = [t["s"] for t in tl]
    n = len(s) - len(s) % 5
    if n < 10:
        return tl, 0.0
    med = [_median(s[p:n:5], 0.0) for p in range(5)]
    p = min(range(5), key=lambda q: med[q])
    rest = _median([x for q in range(5) if q != p for x in s[q:n:5]], 0.0)
    if med[p] <= 0.6 * rest:           # a clear repeat phase: pulldown
        return [t for i, t in enumerate(tl[:n]) if i % 5 != p], med[p]
    return _ct_decimate(tl, lambda t: t["s"]), 0.0


def _ct_aggregate(m):
    """File-level features from measure() output."""
    alts = []
    noise = []
    for tl in m["trans"]:
        floor = 0.0
        if m["rate"] > 27:                 # ~30 fps: drop the 3:2 pulldown repeat first
            tl, floor = _ct_pulldown(tl)
        a = _ct_alt_burst(tl, floor)
        if a is not None:
            alts.append(a)
        noise += [t["nz"] for t in tl if not t["blank"] and t["fc"] < 0.02]
    sp = m["spat"]
    f = dict(alt=sum(alts) / len(alts) if alts else 0.0,
             twos=sum(1 for a in alts if a >= 0.2) / float(len(alts)) if alts else 0.0,
             active=len(alts), bursts=len(m["trans"]), pictures=len(sp),
             noise=_median(noise, 0.0) if noise else -1.0, static=len(noise))
    for k in ("mid", "line", "low", "flat", "strong", "peak"):
        f[k] = _median([x[k] for x in sp], 0.0) if sp else -1.0
        f[k + "4"] = _median([x[k + "4"] for x in sp], 0.0) if sp else -1.0
    return f


# ---- decision ---------------------------------------------------------------------------------
# Tiny logistic models fitted on TRAIN splits only (see REPORT.md):
#   anime vs not:  z = 12.466*alt - 19.788*mid4 + 4.35     (38 DVD + 26 VHS-set train clips)
#   cgi vs live:   y = 2.548 - 2.508*noise - 4.932*mid     (24 DVD train clips; only a label:
#                  cgi and live get the same upscale settings; on tapes the noise is tape noise)
# plus a temporal override: a strong held-drawing cadence (alt >= 0.25) is called anime even when
# the picture is textured (grain, painted backgrounds) - added after checking the user's DBZ clip.
_ct_A_W, _ct_A_B = (12.466, -19.788), 4.35
_ct_C_W, _ct_C_B = (-2.508, -4.932), 2.548
_ct_ALT_STRONG = 0.25      # held-drawing cadence that no train live/CGI DVD clip reached (max 0.21)
_ct_ALT_GOOD = 0.15
_ct_TAPE_NOISE = 1.8       # frame-to-frame noise above this is tape/video noise, not content


def _ct_classify(f):
    """(content, confidence, reasons, scores) from aggregate() features."""
    alt, mid4, mid = f["alt"], f["mid4"], f["mid"]
    few_motion = f["active"] < 4
    few_pics = f["pictures"] < 4
    if f["bursts"] == 0:
        return "live", "low", "could not decode the video (assuming live action)", {}
    if few_motion:
        alt = 0.0                          # no usable motion: rely on the pictures only
    if few_pics:
        mid4 = mid = 0.30                  # neutral texture value when every sample is dark
    z = _ct_A_W[0] * alt + _ct_A_W[1] * mid4 + _ct_A_B
    noise = f["noise"]
    y = _ct_C_W[0] * max(0.0, noise) + _ct_C_W[1] * mid + _ct_C_B
    ev = []
    if not few_motion:
        ev.append("held drawings %d%% of moving frames" % round(100 * alt) if alt >= 0.06
                  else "changes every frame (held drawings %d%%)" % round(100 * alt))
    else:
        ev.append("too little motion to judge the frame cadence")
    if not few_pics:
        ev.append("%s (mid-level texture %d%%)" % (
            "flat drawn-looking pictures" if mid4 < 0.22 else
            "textured pictures" if mid4 > 0.30 else "some texture", round(100 * mid4)))
        if f["line4"] >= 0.025:
            ev.append("dark outlines")
    else:
        ev.append("pictures too dark to judge")
    if alt >= _ct_ALT_STRONG:
        content = "anime"
        conf = "high" if z >= -1.0 else "medium"
    elif z >= 1.0 or (z > 0 and alt >= _ct_ALT_GOOD):
        content = "anime"
        conf = "high" if z >= 2.0 else "medium"
    else:
        content = "cgi" if y > 0 and noise >= 0 else "live"   # noise -1: not measured
        if z <= -2.0:
            conf = "high"
        elif z <= -1.0:
            conf = "medium"
        else:
            conf = "low"
            content = "live" if z > -0.5 else content   # undecided -> the safe choice
        if noise > _ct_TAPE_NOISE:
            content = "live"
            ev.append("heavy tape/video noise %.1f (cgi/live not judged)" % noise)
        else:
            if noise >= 0:
                ev.append("%s (frame-to-frame noise %.2f)" % (
                    "clean, noise-free" if noise < 0.35 else "grain/noise changes every frame",
                    noise))
            if abs(y) < 0.5:
                ev.append("cgi/live uncertain")
                conf = "medium" if conf == "high" else conf
    if few_motion or few_pics:
        conf = "low" if conf != "high" or (few_motion and few_pics) else "medium"
    scores = dict(anime_score=round(z, 3), cgi_vs_live=round(y, 3))
    scores.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()})
    return content, conf, ", ".join(ev), scores


def _ct_detect_content(path, ffmpeg="ffmpeg", ffprobe="ffprobe", info=None):
    """Guess the content type of a video file. Deterministic; never raises for a bad file."""
    t0 = time.time()
    try:
        raw = _ct_extract(path, ffmpeg, ffprobe, info)
        f = _ct_aggregate(_ct_measure(raw))
        content, conf, reasons, scores = _ct_classify(f)
    except Exception as e:                 # analysis must never stop an upscale
        content, conf, reasons, scores = "live", "low", "analysis failed (%s)" % e, {}
    return dict(content=content, confidence=conf, reasons=reasons, scores=scores,
                seconds=round(time.time() - t0, 2))

# ---- auto-detection, source: VHS tape (head-switching band at the bottom, ragged noisy side edges) vs DVD
_src_LOSSLESS = {"huffyuv", "ffvhuff", "ffv1", "utvideo", "lagarith", "rawvideo", "v210", "v410",
            "yuv4", "r210", "magicyuv", "y41p", "ayuv", "zlib", "mszh", "cllc", "vble"}
_src_CAPTURE_EXT = (".avi", ".mpg", ".mpeg", ".dv")
_src_NPOS = 10                 # sample positions
_src_NFRAMES = 2               # consecutive frames per position (each gives 2 fields)
_src_TIME_BUDGET = 11.0        # stop sampling after this many seconds (hard limit for callers: 15 s)


# ---------------------------------------------------------------------------------- helpers ---
def _src_frac(x, default=0.0):
    try:
        f = Fraction(str(x).replace(":", "/"))
        return float(f)
    except (ValueError, ZeroDivisionError, TypeError):
        return default


def _src_hms(x):
    try:
        secs = 0.0
        for part in str(x).split(":"):
            secs = secs * 60 + float(part)
        return secs if secs > 0 else None
    except (ValueError, TypeError):
        return None


def _src_probe(path, ffprobe, info):
    """Stream facts. Uses dvd_upscale.py's probe() dict (w, h, sar, fps, duration) when given."""
    meta = {}
    out = _quiet_run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=codec_name,width,height,sample_aspect_ratio,avg_frame_rate,r_frame_rate,"
                "field_order,duration:stream_tags:format=duration,format_name",
                "-of", "json", path], 30)[1]
    try:
        j = json.loads(out.decode("utf-8", "replace") or "{}")
    except ValueError:
        j = {}
    s = (j.get("streams") or [{}])[0]
    f = j.get("format") or {}
    meta["codec"] = s.get("codec_name", "")
    meta["format"] = f.get("format_name", "")
    meta["field_order"] = s.get("field_order", "")
    meta["w"] = int(s.get("width") or 0)
    meta["h"] = int(s.get("height") or 0)
    meta["sar"] = _src_frac(s.get("sample_aspect_ratio"), 0.0) or 1.0
    meta["fps"] = _src_frac(s.get("avg_frame_rate"), 0.0) or _src_frac(s.get("r_frame_rate"), 0.0)
    dur = _src_hms(f.get("duration")) or _src_hms(s.get("duration"))
    if dur is None:
        tag = next((v for k, v in (s.get("tags") or {}).items()
                    if k.upper().startswith("DURATION")), None)
        dur = _src_hms(tag)
    meta["duration"] = dur or 0.0
    if info:                                     # the caller's probe wins where it has a value
        for k in ("w", "h", "duration"):
            if info.get(k):
                meta[k] = float(info[k]) if k == "duration" else int(info[k])
        if info.get("sar"):
            meta["sar"] = float(info["sar"]) or meta["sar"]
        if info.get("fps"):
            meta["fps"] = float(info["fps"]) or meta["fps"]
    return meta


def _src_repeat_flags(path, ffprobe):
    """Share of the first ~36 frames carrying MPEG-2 repeat-field flags (soft telecine)."""
    out = _quiet_run([ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#36",
                "-show_entries", "frame=repeat_pict", "-of", "csv=p=0", path], 20)[1]
    vals = [ln.strip().strip(",") for ln in out.decode("ascii", "replace").splitlines()]
    vals = [v for v in vals if v.lstrip("-").isdigit()]
    if len(vals) < 8:
        return None
    return sum(1 for v in vals if int(v) > 0) / len(vals)


def _src_grab(path, ffmpeg, t, W, H, n, timeout=20.0):
    """n consecutive frames at t s, luma only, fields deinterleaved (top field rows on top)."""
    raw = _quiet_run([ffmpeg, "-v", "error", "-nostdin", "-threads", "2", "-ss", "%.3f" % t,
                "-i", path, "-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", str(n),
                "-vf", "scale=%d:%d:flags=neighbor,il=l=d:c=d,format=gray" % (W, H),
                "-f", "rawvideo", "-"], timeout)[1]
    fs = W * H
    return [raw[k * fs:(k + 1) * fs] for k in range(len(raw) // fs)]


# ----------------------------------------------------------------------------- measurements ---
def _src_dec2(row):
    """Pair-sum (2:1 horizontal decimation), zero-mean, plus its std."""
    d = [row[i] + row[i + 1] for i in range(0, len(row) - 1, 2)]
    m = sum(d) / len(d)
    d = [x - m for x in d]
    return d, math.sqrt(sum(x * x for x in d) / len(d))


def _src_best_shift(a, b, rng):
    """Integer shift s (in decimated samples) maximizing correlation of a[x] with b[x+s]."""
    n = len(a)
    den = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    if den <= 0:
        return 0, 0.0, 0.0
    best, bs, c0 = -2.0, 0, 0.0
    for s in range(-rng, rng + 1):
        if s >= 0:
            v = sum(map(operator.mul, a[:n - s], b[s:]))
        else:
            v = sum(map(operator.mul, a[-s:], b[:n + s]))
        c = v / den
        if s == 0:
            c0 = c
        if c > best:
            best, bs = c, s
    return bs, best, c0


class _src_Acc:
    def __init__(self):
        self.hs = []              # sideways skew (px) of bottom lines vs the picture above
        self.ref = []             # same, 20 lines higher (what the picture itself does)
        self.rd_bot = 0.0         # mean |line - line above|, bottom lines
        self.rd_body = 0.0        # same, picture lines above
        self.n_rd = 0
        self.e1 = 0.0             # horizontal detail energies
        self.e4 = 0.0
        self.colvals = ([[] for _ in range(40)], [[] for _ in range(40)])
        self.fields = 0


def _src_hs_field(acc, row, Hf):
    """Head switching: the field's bottom lines are skewed sideways (by 5-30 px, growing towards
    the last line, same direction every field) against the picture a few lines higher. The same
    measurement 20 lines higher is the reference for what the picture itself does."""
    def skew(ref, targets, out):
        a, sa = _src_dec2(row(ref))
        if sa < 8:                               # (pair sums: ~4 levels per pixel)
            return
        for r in targets:
            b, sb = _src_dec2(row(r))
            if sb < 8:
                continue
            s, c, c0 = _src_best_shift(a, b, 16)
            if c >= 0.25:
                out.append(2 * s if c - c0 >= 0.03 else 0)
    skew(Hf - 9, (Hf - 4, Hf - 3, Hf - 2), acc.hs)
    skew(Hf - 29, (Hf - 24, Hf - 23, Hf - 22), acc.ref)
    # bottom-line anomaly: how different each of the last 4 lines is from the line above
    def rdiff(r):
        a, b = row(r - 1), row(r)
        return sum(abs(x - y) for x, y in zip(a[::2], b[::2])) / max(1, len(a[::2]))
    acc.rd_bot += sum(rdiff(r) for r in range(Hf - 4, Hf)) / 4
    acc.rd_body += sum(rdiff(r) for r in range(Hf - 24, Hf - 8, 2)) / 8
    acc.n_rd += 1


def _src_border_cols(acc, row, Hf, W):
    for r in range(6, Hf - 8, 4):
        x = row(r)
        L, R = acc.colvals
        for i in range(40):
            L[i].append(x[i])
            R[i].append(x[W - 1 - i])


def _src_border_stats(fields, W, bw, side):
    """Noise of the dark border and line-to-line wobble of its inner edge, per field -> medians."""
    noise, rag = [], []
    if bw < 4:
        return None, None
    for row, Hf in fields:
        vals, pos = [], []
        for r in range(6, Hf - 8):
            x = row(r)
            seg = x[:bw + 10] if side == 0 else x[W - 1:W - bw - 11:-1]
            inner = seg[1:bw - 2]
            if r % 2 == 0:
                vals.extend(inner)
            lo = sum(inner) / len(inner)
            hi_seg = seg[bw + 2:bw + 8]
            hi = sum(hi_seg) / max(1, len(hi_seg))
            if hi - lo < 16:
                pos.append(None)
                continue
            thr = (lo + hi) / 2
            p = None
            for i in range(max(1, bw - 4), min(len(seg), bw + 7)):
                if seg[i] > thr:
                    a, b = seg[i - 1], seg[i]
                    if a <= thr:
                        p = i - 1 + (thr - a) / max(b - a, 1e-3)
                    break
            pos.append(p)
        if vals:
            m = sum(vals) / len(vals)
            noise.append(math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)))
        d = [abs(pos[i] - pos[i - 1]) for i in range(1, len(pos))
             if pos[i] is not None and pos[i - 1] is not None]
        if len(d) >= 20:
            rag.append(_median(d))
    return _median(noise), (_median(rag) if len(rag) >= 2 else None)


def _src_detail_field(acc, row, Hf, x0, x1):
    for r in range(10, Hf - 12, 8):
        x = row(r)[x0:x1]
        m = sum(x) / len(x)
        if m < 24:
            continue
        a, b, c = x[1:], x[:-1], x[4:]
        e1 = sum((p - q) ** 2 for p, q in zip(a, b))
        e4 = sum((p - q) ** 2 for p, q in zip(c, x[:-4]))
        if e4 < 9 * len(x):                          # flat line: no information
            continue
        acc.e1 += e1
        acc.e4 += e4


# --------------------------------------------------------------------------------- the API ---
def _src_detect_source(path, ffmpeg="ffmpeg", ffprobe="ffprobe", info=None, _all=False):
    t0 = time.time()
    path = str(path)
    meta = _src_probe(path, ffprobe, info)
    W, H, D = meta["w"], meta["h"], meta["duration"]
    sc = {"w": W, "h": H, "codec": meta["codec"], "fps": round(meta["fps"], 3),
          "duration": round(D, 2)}
    reasons, prior = [], 0.0

    def done(source, conf):
        sc["seconds"] = round(time.time() - t0, 2)
        return {"source": source, "confidence": conf, "reasons": "; ".join(reasons),
                "scores": sc, "seconds": round(time.time() - t0, 2)}

    if not W or not H:
        reasons.append("could not read the video stream, assuming DVD")
        return done("dvd", "low")
    if W < 240 or H < 200:
        reasons.append("%dx%d is too small for a tape capture" % (W, H))
        return done("dvd", "medium")
    dar = W * meta["sar"] / H
    sc["dar"] = round(dar, 3)
    # ---- hard metadata vetoes (never a tape capture) ----
    veto = None
    if H > 590 or W > 1024:
        veto = "%dx%d is larger than any standard-definition tape capture" % (W, H)
    elif dar > 1.6:
        veto = ("display aspect %.2f:1 (16:9 / anamorphic widescreen): VHS captures are 4:3" % dar)
    sc["veto"] = veto
    if veto and not _all:
        reasons.append(veto)
        return done("dvd", "high")
    # ---- soft priors ----
    fps = meta["fps"]
    ext = os.path.splitext(path)[1].lower()
    rep = _src_repeat_flags(path, ffprobe) if meta["codec"] in ("mpeg2video", "mpeg1video") else None
    sc["repeat_flags"] = None if rep is None else round(rep, 2)
    if rep is not None and rep > 0.2:
        prior -= 2.5
        reasons.append("soft telecine (repeat-field flags on %.0f%% of frames): a DVD film "
                       "master" % (100 * rep))
    elif abs(fps - 24000 / 1001) < 0.03 or abs(fps - 24) < 0.03:
        prior -= 2.0
        reasons.append("%.3f fps film-rate stream: tapes are captured at 25/29.97" % fps)
    elif not (abs(fps - 30000 / 1001) < 0.05 or abs(fps - 25) < 0.05 or
              abs(fps - 60000 / 1001) < 0.1 or abs(fps - 50) < 0.1):
        prior -= 1.0
        reasons.append("unusual frame rate %.3f for a tape capture" % fps)
    if meta["codec"] in _src_LOSSLESS:
        prior += 0.7
        reasons.append("lossless capture codec (%s)" % meta["codec"])
    elif meta["codec"] == "dvvideo":
        prior += 0.2
    if ext in _src_CAPTURE_EXT:
        prior += 0.3
    if not (W in (640, 704, 720) and H in (480, 486, 576) or (W, H) in ((768, 576),)):
        prior -= 1.0
        reasons.append("frame size %dx%d is not a usual capture size" % (W, H))
    sc["prior"] = round(prior, 2)

    # ---- sample frames ----
    if D <= 0:
        D = 10.0
    if D >= 60:
        pos = [D * (0.06 + 0.88 * k / (_src_NPOS - 1)) for k in range(_src_NPOS)]
    else:
        n = _src_NPOS if D >= 10 else max(3, int(D))
        pos = [max(0.0, D * (0.04 + 0.90 * k / max(1, n - 1)) - 0.15) for k in range(n)]
    acc = _src_Acc()
    frames = []
    for t in pos:
        if time.time() - t0 > _src_TIME_BUDGET:
            break
        left = _src_TIME_BUDGET + 3.0 - (time.time() - t0)
        frames.extend(_src_grab(path, ffmpeg, t, W, H, _src_NFRAMES, timeout=min(20.0, max(4.0, left))))
    if not frames:
        reasons.append("could not decode frames (ffmpeg failed), assuming DVD")
        return done("dvd", "low")
    Hf = H // 2
    fields = []
    for buf in frames:
        for base in (0, Hf):
            fields.append((lambda r, buf=buf, base=base: buf[(base + r) * W:(base + r + 1) * W], Hf))
    for row, hf in fields:
        _src_hs_field(acc, row, hf)
        _src_border_cols(acc, row, hf, W)
    acc.fields = len(fields)
    # borders: leading dark columns (column medians over all sampled lines)
    bws = []
    for side in (0, 1):
        med = [_median(c) for c in acc.colvals[side]]
        edge = _median(med[1:4])
        inner = _median(med[24:40])
        bw = 0
        if inner - edge >= 12 and edge < 40:          # a dark band at the frame edge
            thr = min(40, (edge + inner) / 2)
            while bw < 40 and med[bw] < thr:
                bw += 1
        bws.append(bw)
    bstat = []
    for side in (0, 1):
        bw = bws[side]
        if 4 <= bw <= 30:
            bstat.append(_src_border_stats(fields, W, bw, side))
        else:
            bstat.append((None, None))
    x0 = max(bws[0] if bws[0] <= 30 else 0, 8) + 8
    x1 = W - max(bws[1] if bws[1] <= 30 else 0, 8) - 8
    for row, hf in fields:
        _src_detail_field(acc, row, hf, x0, x1)

    # ---- features ----
    big = lambda v: sum(1 for x in v if abs(x) >= 4) / len(v)
    hs_frac = big(acc.hs) if len(acc.hs) >= 12 else None
    ref_frac = big(acc.ref) if len(acc.ref) >= 12 else None
    hs_c = None if hs_frac is None else hs_frac - (ref_frac or 0.0)
    bigs = [x for x in acc.hs if abs(x) >= 4]
    hs_sign = abs(sum(1 if x > 0 else -1 for x in bigs)) / len(bigs) if len(bigs) >= 6 else None
    hsrd = (acc.rd_bot / acc.n_rd + 0.5) / (acc.rd_body / acc.n_rd + 0.5) if acc.n_rd else None
    h1h4 = acc.e1 / acc.e4 if acc.e4 > 0 else None
    sc.update(fields=acc.fields, hs_rows=len(acc.hs),
              hs_sign=None if hs_sign is None else round(hs_sign, 2), hs_median=_median(acc.hs),
              hs_skew=None if hs_frac is None else round(hs_frac, 3),
              ref_skew=None if ref_frac is None else round(ref_frac, 3),
              hs_contrast=None if hs_c is None else round(hs_c, 3),
              hs_rowdiff=None if hsrd is None else round(hsrd, 2),
              border_w=bws,
              border_noise=[None if b[0] is None else round(b[0], 2) for b in bstat],
              border_wobble=[None if b[1] is None else round(b[1], 3) for b in bstat],
              detail_h1h4=None if h1h4 is None else round(h1h4, 3))
    score, ev = _src_score(sc, prior)
    sc["score"] = round(score, 2)
    sc["evidence"] = ev
    reasons.extend(ev["text"])
    del ev["text"]
    kinds = ev["hs"] > 0, ev["border"] > 0
    reasons.append("tape score %.1f (vhs needs %.1f plus both a head-switching band and ragged "
                   "tape borders)" % (score, _src_VHS_SCORE))
    if veto:
        reasons.insert(0, veto)
        return done("dvd", "high")
    if score >= _src_VHS_SCORE and all(kinds):
        return done("vhs", "high" if score >= _src_VHS_HIGH else "medium")
    if score >= _src_VHS_SCORE - 2 and any(kinds):
        reasons.insert(0, "some VHS signs, but not enough to be sure: if this is a tape, use "
                          "--type vhs")
        return done("dvd", "low")
    return done("dvd", "high" if score <= -2 else "medium")


_src_VHS_SCORE = 4.5       # vhs needs this score AND both head switching and tape borders
_src_VHS_HIGH = 6.5


def _src_score(sc, prior):
    """Points for/against a tape from the measured features. Returns (score, evidence)."""
    text, pts = [], {"prior": prior}
    # head switching: bottom lines skewed against the picture (hs_contrast) and much more
    # different from the line above than picture lines are (hs_rowdiff)
    hsc, rd = sc.get("hs_contrast"), sc.get("hs_rowdiff")
    hs = 0.0
    if hsc is not None and rd is not None and hsc >= 0.2 and rd >= 2.0:
        hs = 3.0
        text.append("head-switching band: bottom lines skewed sideways in %.0f%% of fields, "
                    "%.1fx the line-to-line change of the picture" % (100 * sc["hs_skew"], rd))
    elif rd is not None and ((hsc is not None and hsc >= 0.2 and rd >= 1.6) or
                             (rd >= 2.5 and (hsc is None or hsc >= 0.0))):
        hs = 1.0
        text.append("bottom lines unusual (%.1fx line-to-line change%s)" %
                    (rd, "" if hsc is None else ", skewed in %.0f%% of fields" %
                     (100 * max(0.0, sc["hs_skew"] or 0.0))))
    elif rd is not None and rd < 1.3 and (hsc is None or hsc < 0.1):
        hs = -1.0
        text.append("no head-switching band at the bottom")
    pts["hs"] = hs
    # side borders: dark blanking whose edge wobbles from line to line (time-base error),
    # stronger when its black is noisy; clean straight borders are what DVDs have
    bd, sides = 0.0, []
    for k in (0, 1):
        n, w = sc["border_noise"][k], sc["border_wobble"][k]
        if n is None or w is None:
            sides.append("none")
        elif n <= 14 and w >= 0.09:
            sides.append("tape" if n >= 1.5 else "wobbly")
        elif n < 1.0 and w < 0.06:
            sides.append("clean")
        else:
            sides.append("other")
    val = {"tape": 1.5, "wobbly": 1.0, "none": 0.0, "other": 0.0, "clean": -0.5}
    bd = val[sides[0]] + val[sides[1]]
    if "tape" in sides or "wobbly" in sides:
        text.append("ragged side borders (left %s, right %s; edge wobble %s px, black noise %s)"
                    % (sides[0], sides[1], sc["border_wobble"], sc["border_noise"]))
    elif sides == ["clean", "clean"]:
        text.append("clean straight side borders")
    pts["border"] = bd
    pts["sides"] = sides
    # detail: VHS keeps about half a DVD's horizontal resolution
    d = sc.get("detail_h1h4")
    det = 0.0
    if d is not None:
        if d > 0.20:
            det = -3.0
            text.append("fine horizontal detail (%.3f) a tape cannot hold" % d)
        elif d > 0.17:
            det = -1.5
            text.append("more fine detail than a tape usually holds (%.3f)" % d)
        elif d <= 0.155:
            det = 0.5
    pts["detail"] = det
    score = prior + hs + bd + det
    pts["text"] = text
    return score, pts

def detect_content(path, ffmpeg="ffmpeg", ffprobe="ffprobe", info=None):
    """What's in the movie, from frames across all of it: two independent checks, one on single
    pictures and one on motion. Anime settings only when both say "drawn" and the pictures have
    the large flat colour areas of drawn animation: the anime model flattens faces and textures,
    while the live-action model on a cartoon is only a little softer. (Measured on 67 DVD clips,
    47 tapes and 27 further unseen public-domain films; silent films fooled both checks until
    the flat-area floor and the held-drawing rhythm were added.) cgi vs live is a label only:
    they get the same settings."""
    t0 = time.time()
    a = _cs_detect_content(path, ffmpeg, ffprobe, info)
    b = _ct_detect_content(path, ffmpeg, ffprobe, info)
    sa, sb = a.get("scores") or {}, b.get("scores") or {}
    flat = sa.get("flat_share", 0) or 0
    drawn, held = a["content"] == "anime", b["content"] == "anime"
    if drawn:
        look = "looks drawn (flat colour areas %.0f%%%s)" % (
            100 * flat, ", dark outlines" if sa.get("dark_line_ratio", 0) >= _cs_BW_INK else "")
    elif "undecided" in a.get("reasons", ""):
        look = "pictures inconclusive"
    else:
        look = "looks filmed or rendered, not drawn"
    alt = sb.get("alt")
    if held:
        move = ("drawings held for 2-3 frames (%.0f%% of moving frames)" % (100 * alt)
                if isinstance(alt, (int, float)) and alt >= 0.06
                else "the motion check also sees flat drawn pictures")
    elif "too little motion" in b.get("reasons", ""):
        move = "too little motion to judge"
    else:
        move = "a new picture every frame"
    rank = {"low": 0, "medium": 1, "high": 2}
    lower = min(a["confidence"], b["confidence"], key=rank.get)
    if drawn and held and flat >= 0.5:
        content, conf, why = "anime", "high" if lower == "high" else "medium", f"{look}; {move}"
    elif drawn and held:
        # silent films: frame repeats and sharp B&W restorations fool both checks; drawn
        # animation has more flat colour (0.64-0.94 measured, those films 0.41-0.48)
        content, conf = "live", "low"
        why = f"partly: {look}, {move}, but too few flat colour areas for a drawing"
    elif drawn or held:
        content, conf, why = "live", "low", f"partly: {look}, but {move}"     # the safe choice
    else:
        same = a["content"] == b["content"]
        content = "cgi" if same and a["content"] == "cgi" and lower != "low" else "live"
        conf = lower if same else min(lower, "medium", key=rank.get)
        why = f"{look}; {move}" + (
            "; clean computer-rendered look" if content == "cgi" else
            "" if same else "; live action or 3D animation (same settings either way)")
    return dict(content=content, confidence=conf, reasons=why,
                scores=dict(pictures=a, motion=b), seconds=round(time.time() - t0, 1))


def detect_source(path, ffmpeg="ffmpeg", ffprobe="ffprobe", info=None):
    """VHS capture (or VHS-to-DVD transfer) or DVD. Only "vhs" when the picture shows both tape
    signs: a false "vhs" on a DVD would crop it to 4:3 and blur it, a missed tape just gets the
    DVD handling. (Measured: no DVD of 85 called "vhs"; 32 of 37 simulated tapes found.)"""
    r = _src_detect_source(path, ffmpeg, ffprobe, info)
    ev = (r.get("scores") or {}).get("evidence") or {}
    signs = [t for ok, t in ((ev.get("hs", 0) > 0, "a noise band at the bottom from the tape heads"),
                             (ev.get("border", 0) > 0, "ragged, noisy side edges")) if ok]
    if r["source"] == "vhs":
        r["summary"] = " and ".join(signs) or "tape signs"
    elif r["confidence"] == "low":
        r["hint"] = (("some VHS signs (" + " and ".join(signs) + ")" if signs else "some VHS signs")
                     + ", not enough to be sure: if this is a tape, use --type vhs or the vhs "
                     "folder")
    return r


def previous_settings(a):
    """settings.json of an earlier run of this same movie in the work folder, else None."""
    work = Path(a.work + ("_test" if a.test else ""))
    try:
        old = json.loads((work / "settings.json").read_text())
        st = Path(a.input).stat()
    except (OSError, ValueError, AttributeError):
        return None
    if (Path(old.get("input", "")).name != Path(a.input).name or old.get("size") != st.st_size
            or old.get("mtime") != int(st.st_mtime)):
        return None
    return old


def previous_type(a):
    """The type a resumed movie started with, from its work folder, so it keeps its settings:
    detected.json, or for a run started by an older version, its settings.json (the preset's
    own denoise tells them apart even when --model was changed)."""
    old = previous_settings(a)
    if old is None:
        return None
    try:
        d = json.loads((Path(a.work + ("_test" if a.test else "")) / "detected.json").read_text())
        if d.get("type") in TYPE_NAMES:
            return d["type"]
    except (OSError, ValueError):
        pass
    if "vhs" in old:
        return "vhs"
    return {"2:1.5:3:2.5": "anime", "2:1.5:6:5": "live"}.get(
        old.get("denoise"), "anime" if "animevideov3" in str(old.get("model")) else "live")


def resolve_type(a, find):
    """--type auto: VHS tape or DVD first, then for a DVD anime, CGI or live action. Sets a.type
    to anime/live/cgi/vhs and prints what it found and why. A resumed movie keeps its type."""
    for t in ("ffmpeg", "ffprobe"):
        if not find(t):
            sys.exit(f"Missing tool: {t}")
    a.detected = {}
    if a.type != "auto":
        return
    prev = previous_type(a)
    tape_opts = [o for o, v, d in (("--mask", a.mask, "8:8:2:12"),
                                   ("--chroma-delay", a.chroma_delay, "auto")) if v != d]
    if prev:
        a.type = prev
        print(f"Type: {TYPE_NAMES[a.type]} (kept from the earlier run in "
              f"'{a.work + ('_test' if a.test else '')}'; delete that folder to detect again)")
        return
    folder = Path(a.input).resolve().parent.name
    if folder.lower() in TYPE_DIRS:
        a.type = TYPE_DIRS[folder.lower()]
        print(f"Type: {TYPE_NAMES[a.type]} (it's in the '{folder}' folder, as --all does)")
        return
    tag = type_from_name(a.input)
    if tag:
        a.type = tag
        print(f"Type: {TYPE_NAMES[a.type]} (its name starts with "
              f"'{Path(a.input).name[:len(re.match(r'[^_-]*', Path(a.input).name)[0]) + 1]}')")
        return
    try:
        in_folder = bool(json.loads(os.environ.get("DVD_UPSCALE_QUEUE") or "{}").get("folder"))
    except (ValueError, AttributeError):
        in_folder = False
    if tape_opts and not in_folder:
        # (with --all they go to every movie: there only the tapes - the vhs folder, or loose
        # movies detected as tapes - use them, a DVD isn't turned into a tape)
        a.type = "vhs"
        print(f"Type: VHS tape ({' and '.join(tape_opts)} given: those are for tapes)")
        return
    info = probe_or_exit(a.input)
    ff, fp = find("ffmpeg"), find("ffprobe")
    print("Checking what's in the file (frames from all through the movie, 10-20 s)...",
          flush=True)
    try:
        src = detect_source(a.input, ff, fp, info)
    except Exception as e:              # never stop a movie over the detection
        src = dict(source="dvd", confidence="low", reasons=f"couldn't measure it ({e})")
    a.detected["source_check"] = src
    if src["source"] == "vhs":
        # (cartoon tapes too: on simulated Dragon Ball Z tapes the movie-tape settings, with
        # realesrgan-x2plus, kept line weight and colours closer than the anime model, which
        # thickened outlines and shifted flat colours)
        a.type = "vhs"
        print(f"Tape check: VHS tape ({'sure' if src['confidence'] == 'high' else 'likely'}: "
              f"{src.get('summary', src['reasons'])})")
        if src["confidence"] != "high":
            print("  If it isn't a tape: --type anime / live / cgi")
        return
    if src.get("hint"):
        print(f"Tape check: DVD ({src['hint']})")
    try:
        c = detect_content(a.input, ff, fp, info)
    except Exception as e:
        c = dict(content="live", confidence="low", reasons=f"couldn't measure it ({e})")
    a.detected["content_check"] = c
    a.type = c["content"]
    conf = c["confidence"]
    if src.get("hint") and a.type == "anime":
        # possibly a tape (that verdict is rare on real DVDs): not the anime preset, which did
        # worst on tape material
        a.type, conf = "live", "low"
    sure = {"high": "sure", "medium": "likely", "low": "not sure"}[conf]
    print(f"Type: {TYPE_NAMES[a.type]} ({sure}: {c['reasons']})")
    if conf == "low":
        print("  If that's wrong: --type anime / live / cgi"
              + (" / vhs" if src.get("hint") else "")
              + ", or put the movie in a folder of that name for --all")


def default_output(src, base=None, height=1080):
    """Where a movie goes when no output name is given: "1080p Upscale\\<its subfolder>\\<name>
    1080p.mkv" under the current (or queue) folder, the layout --all uses, so either way counts
    as done for the other. A movie outside that folder: a "1080p Upscale" folder next to it."""
    src, base = Path(src).resolve(), Path(base or Path.cwd()).resolve()
    name = f"{src.stem} {height}p.mkv"
    try:
        return base / f"{height}p Upscale" / src.parent.relative_to(base) / name
    except ValueError:
        return src.parent / f"{height}p Upscale" / name


GPU_VRAM_MB = {}        # Vulkan GPU index -> the memory ncnn may use there (MB), from the last probe


def probe_vulkan_gpus():
    """[(index, name)] of the Vulkan GPUs, as ncnn numbers them (the numbers -g takes). Asked
    in a process of its own: ncnn loaded into this one would crash it when it exits (the
    0xC0000005 the upscaler worker has to dodge, see the end of the file). [] if unknown."""
    if not ncnn_available():
        return []
    code = ("import ncnn\n"
            "for i in range(ncnn.get_gpu_count()):\n"
            "    try:\n"
            "        mb = int(ncnn.get_gpu_device(i).get_heap_budget())\n"
            "    except Exception:\n"
            "        mb = 0\n"
            "    print(i, ncnn.get_gpu_info(i).device_name(), '|', mb)\n")
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=60,
                           stdin=subprocess.DEVNULL, text=True, errors="replace")
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for x in r.stdout.splitlines():
        m = re.fullmatch(r"(\d+) (.+?) \| (\d+)", x.strip())
        if m:
            found.append((int(m[1]), m[2].strip()))
            GPU_VRAM_MB[int(m[1])] = int(m[3])
    return found


def nvidia_driver_major():
    """The NVIDIA driver's major version (610 for 610.88), None if there is no nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=10, stdin=subprocess.DEVNULL)
        return int(r.stdout.strip().splitlines()[0].split(".")[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def gpu_class(name):
    """high / mid / low from a GPU's name: what tile size and settings suit it."""
    n = name.lower()
    if re.search(r"\b(graphics|uhd|iris|vega|basic|microsoft)\b", n) and "rtx" not in n:
        return "low"            # integrated graphics, a software adapter
    if re.search(r"rtx\s*(30|40|50)\d\d|rtx\s*a\d|\brx\s*[679]\d{3}", n):
        return "high"
    if re.search(r"gtx|rtx|radeon|\brx\b|arc", n):
        return "mid"
    return "low"


def print_gpu_recommendations(kind, drv):
    """What suits a GPU of this class: models, tile, and what to do when Windows resets it."""
    print("\nRecommendations for a " + {"high": "high-performance dedicated GPU",
                                        "mid": "mid-range or older dedicated GPU",
                                        "low": "integrated or low-power GPU"}[kind] + ":")
    if kind == "high":
        print("  Anime and cartoons: realesr-animevideov3 (the small model: very fast).")
        print("  Live action and 3D CGI: realesrgan-x2plus (the sharpest and most natural; the")
        print("    default for those types). Whole frames (--tile 1024) are quicker than tiles.")
        print("  One upscaler at a time: on an RTX 3060 laptop two were slower, not faster.")
    elif kind == "mid":
        print("  Anime and cartoons: realesr-animevideov3.")
        print("  Live action and CGI: realesrgan-x2plus works; watch for GPU resets at first.")
        print("  If Windows resets the driver during chunks: --tile 200, then --tile 100.")
    else:
        print("  Anime and cartoons: realesr-animevideov3 (compact, light on the GPU).")
        print("  VHS camcorder tapes: realesr-general-dn50-x4v3 (much faster here).")
        print("  Live action and CGI: realesrgan-x2plus is heavy for this GPU: expect days for a")
        print("    movie. Use --tile 64 (or 32), or --fast for a quick draft without the AI.")
    if drv and drv >= 570:
        print(f"  NVIDIA driver {drv}: drivers from 570 on can reset the GPU with ncnn's default")
        print("    options: the profile's stress test finds the ones that survive (once).")
    if not ncnn_available():
        print(f"  The current ncnn isn't installed: {ncnn_install_hint()}")


def gpu_detect_main(argv):
    """python dvd_upscale.py --gpu-detect [--redo]: the GPUs found, how each is classed, the
    driver, and the saved profile; on a computer with no profile yet, the profile is made now
    (as the first run would). --redo: forget the saved profile (not the files of the
    movies) and profile again, including the stress test."""
    p = argparse.ArgumentParser(prog="dvd_upscale.py --gpu-detect")
    p.add_argument("--redo", action="store_true")
    w = p.parse_args(argv)
    if w.redo:
        try:
            ncnn_opts_file().unlink()
            print(f"Forgot the saved profile ({ncnn_opts_file().name}).")
        except OSError:
            pass
    drv = nvidia_driver_major()
    print(f"NVIDIA driver: {drv if drv else 'none found (no nvidia-smi)'}"
          + (" (570 or newer: the stress test is needed)" if drv and drv >= 570 else ""))
    for idx, name in probe_vulkan_gpus():
        mb = GPU_VRAM_MB.get(idx, 0)
        print(f"  Vulkan GPU {idx}: {name} -> {gpu_class(name)}"
              + (f", {mb / 1024:.1f} GB for the upscaler" if mb else ""))
    inject_gpu_hardware_profile(types.SimpleNamespace(gpu=None, gpu_given=False))
    print("Saved profile: " + (json.dumps(ncnn_saved()) if ncnn_saved() else "none"))
    gpus = probe_vulkan_gpus()
    main = ncnn_saved().get("gpu")
    name = dict(gpus).get(main if isinstance(main, int) else (gpus[0][0] if gpus else -1), "")
    if name:
        print_gpu_recommendations(gpu_class(name), drv)
    return 0


def inject_gpu_hardware_profile(a):
    """First run on a computer: find the GPUs, pick the settings that suit the main one, test
    that it survives the upscaler, and save it all in ncnn_opts.json (one profile for every
    later run: nothing is asked again, and an old profile is never overwritten). The same file
    --ncnn-stress / --ncnn-bench / --ncnn-bench-gpu write, and the upscaler reads."""
    saved = ncnn_saved()
    if "gpu" in saved and "tile" in saved:
        say(f"Hardware profile: GPU {saved['gpu']}"
            + (f" + helper {saved['helper']}" if saved.get("helper") is not None else "")
            + f", tile {saved['tile'] or 'automatic'}, set '{saved.get('set', 'base')}' "
            f"(saved in {ncnn_opts_file().name}; delete it to profile again)")
        return
    gpus = probe_vulkan_gpus()
    if not gpus:
        return                  # (nothing learned: nothing saved, the defaults stay)
    say("Hardware profile (first run on this computer):")
    for idx, name in gpus:
        mb = GPU_VRAM_MB.get(idx, 0)
        say(f"  GPU {idx}: {name} ({gpu_class(name)}"
            + (f", {mb / 1024:.1f} GB for the upscaler" if mb else "") + ")")
    # the GPU the movie runs on: the one given, else the first that isn't integrated, else 0
    if a.gpu_given:
        main_gpu = int(str(a.gpu).split(",")[0])
    else:
        main_gpu = next((i for i, n in gpus if gpu_class(n) != "low"), gpus[0][0])
    name = dict(gpus).get(main_gpu, "")
    kind = gpu_class(name)
    tile = {"high": 1024, "mid": 200, "low": 64}[kind]
    mb = GPU_VRAM_MB.get(main_gpu, 0)
    if kind == "high" and 0 < mb < 3000:
        tile = 200          # (a whole frame of the big models needs about 1.5 GB: too little here)
        say(f"  -> only {mb / 1024:.1f} GB of video memory: 200-pixel tiles instead of whole frames")
    else:
        say(f"  -> GPU {main_gpu} ({kind}): "
            + {1024: "whole frames", 200: "200-pixel tiles", 64: "64-pixel tiles"}[tile])
    new = {**saved, "gpu": main_gpu, "tile": tile}
    # (no helper GPU: a weak second GPU is slower than none and some drivers fault on it;
    # --ncnn-bench-gpu measures it and saves one if it is worth having)
    new.pop("helper", None)
    write_durably(ncnn_opts_file(), json.dumps(new, indent=1))
    a.gpu = str(main_gpu) if not a.gpu_given else a.gpu
    # an NVIDIA GPU on a driver from 570 on gets reset by ncnn's default options (they need
    # robust buffer access that its code doesn't ask for): the ones that survive are found
    # once, here. Older drivers don't need it (and the test is skipped: minutes saved)
    drv = nvidia_driver_major()
    if "nvidia" in name.lower() and "set" not in saved and (drv is None or drv >= 570):
        say(f"  NVIDIA driver {drv if drv else '(unknown version)'}: drivers from 570 on can reset "
            "the GPU with ncnn's default options.")
        say("  Testing which ncnn settings this GPU survives (a few minutes, once)...")
        try:
            if ncnn_stress_main(["--gpu", str(main_gpu)]) != 0:
                say("  (no setting survived: see above; the run goes on with the defaults)")
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            say(f"  (the test couldn't run: {e})")
    say(f"  Saved in {ncnn_opts_file().name}.")


def check_values(a):
    """Option values that would only fail later (after detection, or on every movie of a batch)."""
    if a.tile is not None:
        try:
            tile = int(a.tile)
        except (TypeError, ValueError):
            sys.exit("--tile must be 0 (automatic) or a positive number of pixels")
        if tile < 0:
            sys.exit("--tile must be 0 (automatic) or a positive number of pixels")
        a.tile = str(tile)
    a.gpu_given = a.gpu is not None
    if a.gpu is None:
        # (the faster GPU found by --ncnn-bench-gpu, else 0; --gpu on the command line wins)
        saved_gpu, helper = ncnn_saved().get("gpu"), ncnn_saved().get("helper")
        a.gpu = str(saved_gpu) if isinstance(saved_gpu, int) and saved_gpu >= 0 else DEFAULT_GPU
        if isinstance(helper, int) and helper >= 0 and str(helper) != a.gpu:
            a.gpu += f",{helper}"       # (the slower GPU worth having: it takes whole chunks)
    if a.gpu is not None:
        devices = re.sub(r"\s+", "", str(a.gpu)).split(",")
        if not devices or any(not re.fullmatch(r"[0-9]+", device) for device in devices):
            sys.exit("--gpu must be a GPU index or comma-separated indices, e.g. 0 or 0,1")
        if len(set(devices)) != len(devices):
            sys.exit("--gpu cannot list the same GPU more than once")
        a.gpu = ",".join(str(int(device)) for device in devices)
    if a.ai_blend is not None and (
            not math.isfinite(a.ai_blend) or not 0 <= a.ai_blend <= 1):
        sys.exit("--ai-blend must be between 0 and 1")
    invalid_smooth = a.smooth is not None and (
        not math.isfinite(a.smooth) or a.smooth < 0)
    invalid_sharpen = a.sharpen is not None and (
        not math.isfinite(a.sharpen) or not 0 <= a.sharpen <= 2)
    if invalid_smooth or invalid_sharpen:
        sys.exit("--smooth must be 0 or more, --sharpen between 0 and 2")
    if a.dar and frac(a.dar.replace(":", "/")) <= 0:
        sys.exit(f"Invalid --dar '{a.dar}', use e.g. 16:9 or 4:3")
    if a.fps and frac(a.fps) <= 0:
        sys.exit(f"Invalid --fps '{a.fps}', use e.g. 24000/1001 or 25")
    if a.height < 2 or a.height % 2:
        sys.exit("--height must be an even number")
    if a.chunk_frames is not None and a.chunk_frames < 1:
        sys.exit("--chunk-frames must be at least 1")
    if a.test < 0:
        sys.exit("--test must be a number of seconds (0 = the whole movie)")
    if a.gpu_threads is not None and not 1 <= a.gpu_threads <= 16:
        sys.exit("--gpu-threads must be between 1 and 16")
    if a.encode_jobs is None:
        # (a small processor: the encode of one chunk competes with the upscaler's own Python
        # thread; more cores don't need more: on a 3060 laptop the encode takes 7 s of 176)
        a.encode_jobs = 1 if (os.cpu_count() or 4) <= 4 else 2
    if not 1 <= a.encode_jobs <= 4:
        sys.exit("--encode-jobs must be between 1 and 4")
    if a.gpu_jobs is not None and not 1 <= a.gpu_jobs <= 4:
        sys.exit("--gpu-jobs must be between 1 and 4")
    if a.busy_blend is not None and (not math.isfinite(a.busy_blend)
                                     or not 0 <= a.busy_blend <= 1):
        sys.exit("--busy-blend must be between 0 and 1")
    if a.faces is not None and (not math.isfinite(a.faces) or not 0 < a.faces <= 1):
        sys.exit("--faces must be more than 0 and at most 1 (e.g. --faces 0.6)")


def build_parser():
    p = argparse.ArgumentParser(
        epilog="Everyday commands with examples: python dvd_upscale.py --commands. Several movies: "
               "--all [folder] does every movie in a folder, --queue [queue.txt] runs a list.")
    p.add_argument("input")
    p.add_argument("output", nargs="?", default=None,
                   help='where to save it (default: "1080p Upscale\\<name> 1080p.mkv" in the current '
                        "folder, as --all does; missing folders are created)")
    p.add_argument("--mode", choices=["auto", "telecine", "progressive", "interlaced"],
                   default="auto", help="source type (default: auto-detect)")
    p.add_argument("--telecine", action="store_true", help="same as --mode telecine")
    p.add_argument("--analyze", action="store_true", help="only report detection, then exit")
    p.add_argument("--fast", action="store_true", help="no AI, ffmpeg filters only")
    p.add_argument("--cpu", action="store_true", help="encode on the CPU instead of NVENC")
    p.add_argument("--hevc", action="store_true",
                   help="HEVC 10-bit instead of H.264: smaller files, but won't play on PS5, many "
                        "budget TVs or older Chromecasts, and Windows needs the paid HEVC extension")
    p.add_argument("--type", choices=["auto", "anime", "live", "cgi", "vhs"], default="auto",
                   help="auto (default): detect it from frames across the movie. anime: drawn "
                        "animation model. live: live-action model (needs the realesrgan-x2plus "
                        "files, else falls back to realesrgan-x4plus). cgi: 3D animation, same "
                        "settings as live. vhs: VHS captures; movie tapes -> 23.976 fps with "
                        "realesrgan-x2plus, camcorder tapes -> 59.94 fps with "
                        "realesr-general-dn50-x4v3, always 1440x1080")
    p.add_argument("--chroma-delay", default="auto",
                   help="vhs: how far the colour sits right:down of the picture, moved back "
                        "before the upscale. auto (default: measured, 1 line down assumed), off, "
                        "or e.g. 3:1 (3 pixels right, 1 line down)")
    p.add_argument("--mask", default="8:8:2:12",
                   help="vhs: pixels blacked out at the left:right:top:bottom of the 704x480 "
                        "(PAL 704x576) "
                        "picture, over the tape's ragged edges and head-switching noise "
                        "(default 8:8:2:12)")
    p.add_argument("--model", default=None, help="override model name")
    p.add_argument("--scale", type=int, default=None, help="override model scale")
    p.add_argument("--trained", action="store_true",
                   help="use the model made by upscale_training.py for this kind of movie "
                        "(ai-anime-x2, ai-cgi-x2, ai-live-x2 or ai-vhs-x2, by the detected --type; else the "
                        f"older {TRAINED_MODEL}). From a 4K Blu-ray it makes ai-<type>-x4: --trained --scale 4 "
                        "--height 2160 (with --height over 1080 the x4 model is picked by itself)")
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--dar", default=None, help="force aspect, e.g. 16:9 or 4:3")
    p.add_argument("--fps", default=None, help="override output fps, e.g. 24000/1001")
    p.add_argument("--chunk-frames", type=int, default=None)
    p.add_argument("--test", type=int, default=0, help="only process first N seconds")
    p.add_argument("--phone", nargs="?", type=int, const=PHONE_PORT, metavar="PORT",
                   help="the progress page for a phone is on by itself: open the address the run "
                        "prints in the phone's browser on the same Wi-Fi. --phone also asks "
                        "Windows (an admin prompt) to open the firewall for it for this run, for "
                        "Tailscale and the home network; --phone PORT: another port (default "
                        f"{PHONE_PORT})")
    p.add_argument("--no-phone", action="store_true", help="don't serve the phone page")
    p.add_argument("--no-guard", action="store_true",
                   help="Windows: leave the console's QuickEdit mode on and don't block a "
                        "shutdown or restart while it runs (sleep is still prevented)")
    p.add_argument("--esrgan", default=ESRGAN_DEFAULT)
    p.add_argument("--engine", choices=("auto", "exe", "ncnn"), default="auto", dest="engine_choice",
                   help="what runs the big models (x2plus, x4plus): ncnn, the current ncnn from "
                        f"pip ({NCNN_INSTALL}; fixes the GPU resets of current NVIDIA drivers), "
                        "or exe, realesrgan-ncnn-vulkan's own (ncnn from 2022). auto (default): "
                        "ncnn when it is installed (and --esrgan names no upscaler of its own)")
    p.add_argument("--gpu", default=None,
                   help="Vulkan GPU index for the upscaler (-g; default 0, or the faster one found by --ncnn-bench-gpu). Several, e.g. 0,1 (a laptop's "
                        "NVIDIA plus the processor's built-in graphics): the first works through "
                        "the movie, the others upscale whole chunks alongside it")
    p.add_argument("--tile", default=None,
                   help="tile size in pixels (-t; default: whole frames for anime, the "
                        "upscaler's own 200 for live action). Smaller tiles need less GPU memory "
                        "and are shorter pieces of GPU work; they change the picture a little. "
                        "Without --tile, a chunk the GPU fails on is tried again with fewer "
                        "frames at once and then with 100-, 64- and 32-pixel tiles")
    p.add_argument("--gpu-threads", type=int, default=None,
                   help="frames the GPU upscales at once (default 8 for anime, 2 for live "
                        "action): more can keep a GPU busier, fewer need less GPU memory. The "
                        "picture is the same either way. When a chunk fails with the default, it "
                        "is tried again one step lower (anime 6, 4, 2, 1; live action 1), and "
                        "the rest of the run keeps that")
    p.add_argument("--no-profile", action="store_true",
                   help="don't profile the GPUs on the first run (the profile is saved in "
                        "ncnn_opts.json, and an existing one is never overwritten)")
    p.add_argument("--no-step-down", action="store_true",
                   help="after a GPU reset, don't lower the GPU's load (fewer frames at once, "
                        "smaller tiles) for the rest of the run: the chunk is tried again with "
                        "the same settings. Keeps full speed if resets are rare; a GPU that keeps "
                        "resetting then stops the run")
    p.add_argument("--best-quality", action="store_true",
                   help="upscale live action / 3D animation with the model that "
                        "--ncnn-models MOVIE --save found truest to the original (not used "
                        "otherwise: the usual model stays the default)")
    p.add_argument("--no-skip-black", dest="no_skip_black", action="store_true",
                   help="upscale black frames too (by default a frame that is black through and "
                        "through is not sent to the upscaler: its upscale is black as well)")
    p.add_argument("--no-detail-blend", action="store_true",
                   help="use --ai-blend everywhere (by default crowds of small people, grass and "
                        "other dense small detail get less of the AI, about 0.4: it paints them "
                        "flat and makes them shimmer; smooth areas keep --ai-blend)")
    p.add_argument("--busy-blend", type=float, default=None, metavar="SHARE",
                   help="the AI's share in crowds, grass and other busy detail, 0-1 (default "
                        f"{DETAIL_BLEND['low_blend']:g}; at most --ai-blend). Higher: crisper "
                        "but invented detail that shimmers; lower: truer and steadier, softer. "
                        "E.g. 0.55 for CGI films whose grass and fur look too soft")
    p.add_argument("--no-crop", action="store_true",
                   help="don't cut the black bars of a widescreen movie off before the upscale "
                        "(they are looked for by default: the upscaler then does only the "
                        "picture, about a quarter less work for 2.39:1)")
    p.add_argument("--encode-jobs", type=int, default=None, metavar="N",
                   help="chunks encoded at once while the next is upscaled (default 2, 1 on a "
                        "processor with 4 threads or fewer): the CPU "
                        "filters and encode of a chunk can take longer than its upscale, and the "
                        "GPU then waits")
    p.add_argument("--gpu-jobs", type=int, default=None,
                   help="upscalers running at once on each GPU (default 1: the anime and "
                        "camcorder models' 8 frames at once keep the GPU busy already, two of "
                        "those were 5.5x slower on a 6 GB laptop GPU, and on a GPU that the big "
                        "models get reset, more work at once makes a reset likelier: 2 frames "
                        "at once failed several times sooner than 1). A second one keeps the "
                        "GPU busy while the other starts up, checks its frames or waits for the "
                        "next ones. The picture is the same either way; if two turn out slower "
                        "than one, it goes back to one by itself")
    p.add_argument("--ai-blend", type=float, default=None,
                   help="share of the AI result mixed with a plain upscale, 0-1 "
                        "(lower = less flicker, less detail; default 1 anime, 0.75 live/vhs)")
    p.add_argument("--smooth", type=float, default=None,
                   help="temporal smoothing after the upscale, 0 = off "
                        "(default 0 anime, 4 live, vhs 4 movie tapes / 6 camcorder tapes; "
                        "higher calms flicker but can ghost)")
    p.add_argument("--sharpen", type=float, default=None,
                   help="final sharpening amount (default 0.6 anime, 0.3 live, vhs 0.3 movie "
                        "tapes / 0 camcorder tapes)")
    p.add_argument("--stabilize", nargs="?", const="normal", choices=["normal", "strong"],
                   default=None,
                   help="steady a shaky camcorder or home video (VHS tapes, camera clips): the "
                        "camera shake is measured once, then smoothed out, with the picture "
                        "zoomed in 4%% so no moving edges show (strong: 8%%, for very shaky "
                        "video). Not for films, whose camera moves are meant")
    p.add_argument("--faces", nargs="?", type=float, const=FACE_STRENGTH, default=None,
                   metavar="STRENGTH",
                   help="restore faces after the upscale (GFPGAN or CodeFormer redraws each "
                        "face with real detail: eyes, teeth, skin). On by itself for live action "
                        "and home video when installed; give it for 3D animation to use it there "
                        f"(never for anime). STRENGTH 0-1 (default "
                        f"{FACE_STRENGTH:g}): the redrawn face's share of the final picture, at "
                        "most the AI frames' share (--ai-blend: 0.75 live and VHS). Needs "
                        "pip install -U onnxruntime-directml (Windows; onnxruntime-gpu for NVIDIA "
                        "with CUDA and cuDNN) opencv-python-headless numpy, and its model files "
                        "in a face_models folder next to this script (the run says where to get "
                        "them)")
    p.add_argument("--no-faces", action="store_true",
                   help="no face restoration (it is on by itself for live action and VHS when its "
                        "packages and model files are installed)")
    p.add_argument("--face-model", choices=sorted(FACE_MODEL_FILES), default=FACE_MODEL,
                   help=f"--faces: gfpgan (GFPGAN 1.4, the default) or codeformer (CodeFormer, "
                        f"fidelity {FACE_FIDELITY:g}; its licence, S-Lab 1.0, allows "
                        "non-commercial use only)")
    p.add_argument("--face-models", default=None, metavar="DIR",
                   help="--faces: the folder with its model files (default: face_models next "
                        "to this script)")
    p.add_argument("--fix-combed", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--work", default=None,
                   help='folder for the temporary files (default: "<movie name>_work" next to '
                        "the movie, the one --all uses; a --test run adds _test)")
    return p


COMMANDS = r"""
DVD -> near-Blu-ray upscaler: everyday commands
(type them in PowerShell, in the folder with dvd_upscale.py; put names with spaces in "quotes")

ONE MOVIE (it works out by itself whether it's anime, live action, 3D CGI or a VHS tape,
and prints what it found and why)
  python dvd_upscale.py "Movie.mkv"                       detect, then upscale; saved as
                                                          "1080p Upscale\Movie 1080p.mkv" (folder made
                                                          for you, the same place --all uses)
  python dvd_upscale.py "Movie.mkv" --analyze             only show what it detects
  python dvd_upscale.py "Movie.mkv" --test 60             60-second preview first
                                                          ("1080p Upscale\Movie 1080p test.mkv")
  python dvd_upscale.py "Movie.mkv" "D:\Out\Movie.mkv"    choose the name/folder yourself
  python dvd_upscale.py "Movie.mkv" "Movie 1080p.mp4"     .mp4 for iPhone/Apple TV/browsers
                                                          (DVD subtitles can't go in .mp4)
  If it picks the wrong type, say which (add to any command):
    --type anime     anime, cartoons, hand-drawn Disney
    --type live      live action, stop-motion
    --type cgi       3D animation (Pixar, DreamWorks...): same settings as live
    --type vhs       a VHS tape capture (or a VHS-to-DVD transfer)

VHS TAPES (captured from a VCR: .mpg, .avi, .mkv ...; always 1440x1080)
  python dvd_upscale.py "Tape.mpg" "Tape 1080p.mkv" --analyze              VHS? movie or camcorder
                                                                           tape?
  python dvd_upscale.py "Tape.mpg" "Tape 1080p.mkv"                        movie tapes 23.976 fps,
                                                                           camcorder tapes 59.94 fps
  python dvd_upscale.py "Tape.mpg" "Tape 1080p.mkv" --type vhs --mode telecine
                                                     a movie tape it took for camcorder video
  python dvd_upscale.py "Tape.mpg" "Tape 1080p.mkv" --stabilize       steady a shaky camcorder
                                                                           tape (zooms in 4%;
                                                                           --stabilize strong: 8%)
  (for camcorder tapes put realesr-general-dn50-x4v3.param/.bin in the upscaler's models folder)

EVERY MOVIE IN A FOLDER
  python dvd_upscale.py --all                          all movies in this folder (each detected)
  python dvd_upscale.py --all --type anime             ...all loose ones as anime
  python dvd_upscale.py --all "D:\Movies"              all movies in another folder
  python dvd_upscale.py --all --delete-originals       remove each original once its copy is checked
  python dvd_upscale.py --all --shutdown               turn the PC off when everything is done
    - subfolders named "anime" / "live" / "cgi" / "vhs" force that type for the movies in them
    - finished movies go into "1080p Upscale"; finished, short and HD files are skipped
    - you can add movies while it runs; queue_log.txt says what each movie was done as

A LIST OF MOVIES (each with its own options)
  python dvd_upscale.py --queue                        runs queue.txt (one movie per line, e.g.
                                                       "Movie.mkv" "Movie 1080p.mkv" --type live)
  python dvd_upscale.py --queue --shutdown

CUT A SHORT CLIP (e.g. to send a sample; nothing is re-encoded, so it takes seconds)
  python dvd_upscale.py --clip "Movie.mkv" 15:00 30          30 seconds from 15:00, saved as
                                                              "Movie clip 15m00s.mkv" (picture + sound)
  python dvd_upscale.py --clip "Movie.mkv" 1:05:00 20 --video-only     picture only (smallest file)

TEST ONE SPOT OF A MOVIE (see what the full run will make of it, in a few minutes)
  python dvd_upscale.py --clip "Movie.mkv" 15:00 10 --upscale
                     cuts 10 seconds from 15:00 ("Movie clip 15m00s.mkv", next to the movie) and
                     upscales just that, the way the full run does that movie (it checks the
                     whole movie first) -> "1080p Upscale\Movie clip 15m00s 1080p.mkv"
  add options as usual, e.g. --type anime or --hevc
  the same with ffmpeg itself:
  ffmpeg -ss 15:00 -i "Movie.mkv" -t 30 -map 0:v:0 -c copy "clip.mkv"
    -ss 15:00   where to start (minutes:seconds, or hours:minutes:seconds like 1:05:00)
    -t 30       how many seconds
    -map 0:v:0  picture only; -map 0:v:0 -map 0:a gives picture and sound
    -c copy     no re-encoding: the clip starts at the nearest keyframe (up to ~1 s early)

WATCH IT ON YOUR PHONE (iPhone or Android, any browser, same Wi-Fi as the PC)
  It's ON BY ITSELF for every run (one movie, --all, --queue); --no-phone turns it off
    1. At the start the run prints an address, e.g.  Phone: ... open  http://192.168.1.20:8642
    2. Type that address into Safari / Chrome on the phone (Share > Add to Home Screen keeps it
       one tap away)
    3. The page shows the movie, % done, time left for this movie and for all of them, the step
       it's on, the latest output, and the NVIDIA GPU's load, temperature, power, memory and
       clock (a warning when heat or the power cap slows it down); it refreshes every 3 s
  The first time, Windows asks whether Python may use the network: Allow (private networks).
  Page won't load? Settings > Network > your Wi-Fi > Network profile type = Private, and the
  phone on the same Wi-Fi (not mobile data). Port taken? --phone 8650
  --phone: also asks Windows for admin rights (the "allow changes" prompt) to open the page's
  port in the firewall for this run only (home network and Tailscale), and closes it again
  when the run ends. Use it when the page won't load, e.g. through Tailscale.
  TAILSCALE: if Tailscale runs on the PC, the run also prints its Tailscale address
  (e.g. http://laptop.tail1234.ts.net:8642): with the Tailscale app on the phone, that one works
  from anywhere (mobile data, work, a trip), not only at home. It never changes.
  The Wi-Fi address stays the same from run to run (unless the router gives the PC a new one), so a
  home-screen shortcut keeps working.
  "offline" on the page: the run finished or was stopped, or the PC is asleep.

STOP / RESUME
  Ctrl+C                       stop (finished chunks are kept)
  run the same command again   continue where it stopped (with the same type as before)
  shutdown /a                  cancel a --shutdown countdown

USEFUL EXTRAS (add to any command above)
  --hevc             smaller files, but needs a newer TV/player (H.264 is the default)
  --height 720       720p instead of 1080p
  --dar 16:9         fix a squeezed/stretched picture (or --dar 4:3)
  --ai-blend 0.5     gentler AI (less "painted" look; default 0.75 live, 1 anime). Crowds,
                     grass and other busy detail get less AI by themselves (about 0.4): the AI
                     smears small people together there; --no-detail-blend turns that off
  --busy-blend 0.55  a little more AI in that busy detail (crisper grass and fur in CGI films,
                     but more shimmer; 0.4 default, at most --ai-blend)
  --fast             no AI: much quicker, ordinary resize
  --faces            face restoration: ON BY ITSELF for live action and VHS once its extras are
                     installed (the run says which, and goes on without them until then);
                     --faces 0.8 for more (at most the --ai-blend share, 0.75); add --faces to
                     a 3D-animation (CGI) movie to use it there too (check a clip: its model
                     knows photos, faces can turn photographic); never for anime
  --no-faces         no face restoration
  --cpu              encode without an NVIDIA GPU (slow)
  --tile 128         smaller pieces of GPU work, if the GPU runs out of memory or is reset
                     (lowered by itself to 100, 64, 32 if fewer frames at once didn't help)
  --gpu 1 / 0,1      another GPU, or both (the default is GPU 0; see GPU SET-UP AND SPEED)
  --gpu-threads 4    frames the GPU works on at once (anime/camcorder default 8, lowered to
                     6, 4, 2, 1 by itself if a chunk fails; live action 2, then 1); same picture
  --no-profile       don't profile the GPUs on the first run on a computer (the profile, the
                     settings the GPU survives and the tile, is saved in ncnn_opts.json once)
  --no-skip-black    upscale black frames too (by default a frame that is exactly all black is
                     not sent to the upscaler: black in, black out; fades are still upscaled)
  --no-crop          don't cut the black bars of a widescreen movie off before the upscale (by
                     default they are found, cut, upscaled without and put back: 2.39:1 movies
                     take about a quarter less GPU work, the same picture; a movie already
                     started keeps what it started with)
  --encode-jobs 2    chunks encoded at once while the next is upscaled (default 2; 1 = the old
                     way). The CPU part of a chunk (filters, encode) can take longer than its
                     upscale: the progress line then shows "(upscale 140s, encode 175s)" and the
                     GPU waits. Up to 4 if the encode is still the longer one.
  --gpu-jobs 2       two upscalers at once (default 1): can fill a GPU that waits between
                     chunks, but on an RTX 3060 laptop with the current ncnn it was SLOWER (204
                     against 179 s a chunk): time a --test 60 before using it. Up to 4
  --engine exe       live action/CGI on realesrgan-ncnn-vulkan's own engine (2022) even when
                     the current one is installed (python -m pip install --no-deps ncnn numpy)

SPEED CHECKLIST (what holds a laptop GPU back; the script can't change these, you can)
  1. Plug in the charger (the GPU's power limit drops a lot on battery).
  2. Windows 11: Settings > System > Power > Power mode = "Best performance".
  3. The laptop maker's app (Armoury Crate, Legion Vantage, MSI Center, OMEN ...): Turbo /
     Performance mode and the fans on max: this raises the GPU's power and heat limits.
  4. NVIDIA Control Panel > Manage 3D settings > Power management mode = "Prefer maximum
     performance" (Windows: Settings > System > Display > Graphics: python.exe = High performance).
  5. Cooling: hard flat surface, raised at the back, vents clear, a cooling pad. The run shows
     "slowed by: power cap, thermal" while the GPU is held back by power or heat.
  6. Close what uses the GPU (browser video, games, overlays, AMD software), and pause
     antivirus scans while a movie runs.
  The script already keeps the PC awake and runs above-normal priority. Thermal and power
  limits protect the GPU and can't (or shouldn't) be switched off.

  python dvd_upscale.py --gpu-detect          the GPUs, their class, the driver and the saved profile
                                              (made now if there is none; --redo profiles again)

THE GPU STEP-DOWN (what happens when Windows resets the graphics card: "failed -4")
  The chunk is tried again, and the GPU is given less to do at once for the rest of the run:
  each step is a smaller tile (512, 256, 128, 64, 32 pixels; less for the old engine: fewer
  frames at once, then tiles). Smaller tiles are slower: the overlap between them is done twice.
  Extra work on a DVD frame against whole frames: 512 +2%, 256 +11%, 128 +27%, 64 +66%, 32 +149%.
  The picture changes very slightly, so a reset after
  a long good stretch (120+ frames) doesn't step down. Where it went is kept in gpu_steps.json
  next to the script, so the next run starts there. It costs speed ONLY after a reset: with
  settings the GPU survives (--ncnn-stress) it never happens.
  --no-step-down          never lower the load: a reset retries the chunk with the same settings
                          (and ignores gpu_steps.json). Full speed if resets are rare; a GPU that
                          keeps resetting then stops the run instead of going on slower
  gpu_steps.json          delete it to go back to the full load (after a driver update, say)

GPU SET-UP AND SPEED (run these once from the script's folder; each saves what it finds in
ncnn_opts.json next to dvd_upscale.py, and every later run uses it)
  python dvd_upscale.py --ncnn-auto "CGI\Movie.mkv"
        ALL OF THE BELOW IN ONE GO (about 15-25 minutes, then nothing more to do): clears the old
        saved settings, runs --ncnn-stress, --ncnn-bench, --ncnn-bench-gpu and --ncnn-models, times a 60-second
        test run while watching the GPU, and prints a summary with the hours the movie will take.
        Add --gpu N to set up another GPU, --skip-gpu-test, --skip-model-test or --no-test-run. Use it after a
        driver update, or when GPU resets ("failed -4") come back. The pieces, one by one:
  python dvd_upscale.py --ncnn-models "CGI\Movie.mkv" --save
        FIND THE BEST-QUALITY MODEL: the same test, and the model that is clearly truer to the
        original than realesrgan-x2plus (and not much slower) is saved as best_model for live
        action and 3D animation; else x2plus stays. It changes nothing by itself: upscale with
        --best-quality to use it (python dvd_upscale.py "Movie.mkv" --best-quality). Also
        run by --ncnn-auto. A movie already started keeps its model.
  python dvd_upscale.py --ncnn-models "CGI\Movie.mkv"
        tests EVERY model in the models folder on the same frames of the movie (shrunk to half,
        upscaled back, compared with the original: SSIM/PSNR against a plain resize), prints a
        table with the speed of each, and writes model_compare.png next to the script (the same
        part of the frame from each model side by side, to judge by eye). A higher score means a
        truer picture, not always a nicer one. It changes nothing.
  python dvd_upscale.py --ncnn-stress --gpu 0
        finds the ncnn options your GPU survives (the GPU resets with "vkWaitForFences failed -4"
        or "vkQueueSubmit failed -4" mean it doesn't). Saves e.g. {"set": "nowinograd"}.
        Takes a few minutes; run it again after a driver update or if resets come back.
  python dvd_upscale.py --ncnn-bench "CGI\Movie.mkv" --gpu 0
        the fastest settings that keep the same picture: whole frames instead of 200-pixel tiles
        (about 50% faster, no tile seams) and fp16 (kept only if the picture matches). Saved as
        "set" and "tile". Takes a few minutes.
  python dvd_upscale.py --ncnn-winograd "CGI\Movie.mkv"
        the faster way to run the 3x3 convolutions (winograd, three variants) was switched off
        by --ncnn-stress because the GPU got reset with all three on. This tries each one on its
        own on 300 frames, and saves the fastest that gets through all of them with the same
        picture. About 8 minutes; --ncnn-stress first. (Part of --ncnn-auto.)
  python dvd_upscale.py --ncnn-bench-gpu "CGI\Movie.mkv"
        times every GPU except integrated graphics (--gpus 0,1,2 to choose), saves the fastest
        as the default GPU and, if
        the other is at least 20% as fast, as its helper (it then upscales whole chunks too,
        like --gpu 0,1; "--gpu 0" on its own runs without it)
  --gpu N / --gpu 0,1     one GPU (default 0, or the one --ncnn-bench-gpu saved), or both: the
                          first works through the movie, the second upscales whole chunks too
  --tile 384              a tile size by hand (overrides the saved one; 0 = automatic)
  gpu_steps.json          next to the script: the slower settings (fewer frames at once, small
                          tiles) the script went down to after GPU resets, so the next run
                          starts there. DELETE it after fixing the GPU (stress test, new driver).
  ncnn_opts.json          the saved set/tile/gpu. Delete it to go back to the defaults; it can be
                          edited: {"set": "nowinograd", "tile": 1024, "gpu": 0}
  Time a run first:  python dvd_upscale.py "Movie.mkv" --test 60   (note the "NNNs/chunk" line)
  Watch the GPU:     nvidia-smi -l 2   (utilization well under 90% = something else is slowing it)
  Troubleshooting:   "NOTE: the current ncnn didn't work here" = it fell back to the old engine
                     (slow, resets); the lines above it say why. A resumed movie keeps its
                     settings; gpu_errors.log in the movie's _work folder has the GPU details.

MORE
  python dvd_upscale.py --help          every option, briefly
  python dvd_upscale.py --all --help    folder-mode options
  queue_log.txt                         what happened in --all / --queue runs
"""


QUICK_START = r"""
QUICK START: the ones to remember (everything else below is the detail)
  python dvd_upscale.py "Movie.mkv"                  upscale one movie: everything is automatic
  python dvd_upscale.py --all                        every movie in this folder (each detected)
  python dvd_upscale.py --ncnn-auto "Movie.mkv"      set the GPU up and test it (once, ~20 min):
                                                     the settings it survives, the fastest, the
                                                     best model; saved and used from then on
  python dvd_upscale.py "Movie.mkv" --best-quality   upscale with the model --ncnn-models --save found
  (every run)                                        watch it on your phone: open the address it
                                                     prints at the start (same Wi-Fi)
  python dvd_upscale.py --gpu-detect                 the GPUs, the driver, the saved profile
  python dvd_upscale.py --commands                   this list

ALL THE SPECIAL COMMANDS, ONE LINE EACH (details below)
  --ncnn-auto MOVIE        everything below in one go, with a summary at the end
  --ncnn-stress            the ncnn options the GPU survives (no "failed -4" resets)
  --ncnn-bench MOVIE       the fastest settings with the same picture (whole frames, fp16)
  --ncnn-bench-gpu MOVIE   which GPU is faster (saved as the default)
  --ncnn-models MOVIE      every model on your frames, with a picture to judge by eye
      --save               ...and save the best one (an upscale uses it only with --best-quality)
  --ncnn-winograd MOVIE    the winograd variants (none was faster on an RTX 3060 laptop)
  --gpu-detect [--redo]    GPUs, class, driver, memory, saved profile, recommendations
  --clip MOVIE START SECS  cut a sample (add --upscale to upscale just that piece)
  --queue / --all          several movies (a list of lines / every movie in a folder)
  --analyze                only show what it detects      --test 60   a 60-second preview
  Turn a built-in automatic step off: --no-crop (black bars) --no-skip-black (black frames)
  --no-detail-blend (less AI in crowds/busy detail)
  --no-phone (don't serve the progress page for a phone)  --no-faces (no face restoration)
  --no-profile (first-run GPU profile) --encode-jobs 1 --gpu-jobs 1 --engine exe
  --no-step-down (don't slow the GPU settings down after a reset)  --best-quality (use the saved model)
"""


def commands_text():
    """The cheat sheet: the quick start first, then the detail (COMMANDS), then every option of
    the parser with its first sentence, made from the parser itself so it can't fall behind."""
    p = build_parser()
    rows = []
    for act in p._actions:
        names = [o for o in act.option_strings if o.startswith("--") and o != "--help"]
        if not names or not act.help or act.help == argparse.SUPPRESS:
            continue
        text = " ".join(str(act.help).split()).replace("e.g.", "e.g.,").replace("i.e.", "i.e.,")
        first = re.split(r"(?<=[a-z0-9)]\.)\s", text, maxsplit=1)[0]
        if len(first) > 110:
            first = first[:110].rsplit(" ", 1)[0] + " ..."
        rows.append((", ".join(names) + (f" {act.metavar or act.dest.upper()}"
                                         if act.nargs != 0 and act.const is None
                                         and not isinstance(act, argparse._StoreTrueAction)
                                         else ""), first))
    width = max((len(n) for n, _ in rows), default=0) + 2
    lines = ["", "EVERY OPTION (from --help; add to any command above)"]
    lines += [f"  {n:<{width}}{h[:150]}" for n, h in sorted(rows)]
    lines += ["", "python dvd_upscale.py --self-test   checks the script's own logic (a few seconds, "
              "no GPU; after changing the script)",
              "(--faces-worker and --ncnn-upscaler are used by the script itself: not for typing)"]
    return QUICK_START + COMMANDS + "\n".join(lines) + "\n"


def start_run(a):
    """Work folder, input/output sanity, the tools next to this script; returns find(tool)."""
    phone_movie(Path(a.input).name)
    check_values(a)
    a.combed = a.fix_combed     # (--clip --upscale: as detected for the whole movie)
    if a.work is None:
        # one work folder per movie, next to it (where --all puts it, so either can continue a
        # movie the other started); a movie started in the old shared default continues there
        a.work = "upscale_work"
        if previous_settings(a) is None:
            a.work = str(Path(a.input).with_name(Path(a.input).stem + "_work"))
    a.work = str(Path(a.work))          # "upscale_work\" (tab completion) -> "upscale_work"
    if not a.analyze:
        src, out = Path(a.input).resolve(), Path(a.output or "").resolve()
        if a.output and os.path.normcase(str(src)) == os.path.normcase(str(out)):
            sys.exit("The output can't be the input file itself: that would replace the original.")
    here = str(Path(__file__).resolve().parent)
    # tools next to this script are found too, so --all/--queue on another folder still works
    find = lambda t: shutil.which(t) or shutil.which(t, path=here)
    if find("ffmpeg"):
        check_ffmpeg()
    resolve_type(a, find)
    a.trained_pick = a.trained and a.model is None     # (the model for the type, once its folder is known)
    a.trained_scale = a.scale                           # (--scale 2 / 4 as given: an x2 or x4 trained model)
    if a.trained:
        a.model = a.model or TRAINED_MODEL
        a.scale = a.scale or 2
    return find


def pick_preset(a, find):
    """The type's preset (model, scale, chunk size, denoise, AI blend, smoothing, sharpening), with
    what was given on the command line kept; returns what the user gave, and the VHS probe."""
    vhs_info = None
    if not a.fast and not a.analyze and a.type != "vhs" and not a.no_profile:
        inject_gpu_hardware_profile(a)
    # model, scale, chunk frames, pre-denoise (hqdn3d), ai blend, post smoothing, sharpen
    presets = {"anime": ("realesr-animevideov3", 2, 1440, "2:1.5:3:2.5", 1.0, 0, 0.6),
               # x2plus: on small DVD faces x4plus draws eyes as black outlined almonds and
               # teeth as outlined boxes; x2plus keeps them natural and flickers less.
               # It isn't in the official ncnn download; without it x4plus is used (see below).
               "live": ("realesrgan-x2plus", 2, 480, "2:1.5:6:5", 0.75, 4, 0.3),
               # VHS movie tapes: the live settings measured best on simulated tapes too (LPIPS
               # 0.157 vs 0.302 for a plain upscale); stronger denoise, as part of vhs_prefilter
               "vhs": ("realesrgan-x2plus", 2, 480, "4:8:9:14", 0.75, 4, 0.3)}
    presets["cgi"] = presets["live"]        # 3D animation: textures, like live action
    if a.type == "vhs":
        vhs_info = vhs_setup(a, find)
        if a.mode != "telecine":
            presets["vhs"] = VHS_VIDEO
    pm, ps, pc, a.denoise, pb, psm, psh = presets[a.type]
    user_model, user_blend, user_sharpen = a.model, a.ai_blend, a.sharpen
    user_chunk, user_scale = a.chunk_frames, a.scale
    a.ai_blend = pb if a.ai_blend is None else a.ai_blend
    a.smooth = psm if a.smooth is None else a.smooth
    a.sharpen = psh if a.sharpen is None else a.sharpen
    a.model = a.model or pm
    a.scale = a.scale or (4 if "x4plus" in a.model or "x4v3" in a.model else
                          2 if "x2plus" in a.model else ps)
    # x4 models: 4x the temporary PNGs of x2, so at most the 480 frames of the live preset
    a.chunk_frames = a.chunk_frames or (pc if a.scale <= 2 else min(pc, 480))
    old = previous_settings(a)
    if user_chunk is None and old and (old.get("model"), old.get("scale")) == (a.model, a.scale):
        a.chunk_frames = old.get("chunk") or a.chunk_frames      # as the movie was started
    if a.telecine:
        a.mode = "telecine"
    if a.dar:
        a.dar = a.dar.replace(":", "/")     # (checked in check_values)
    return (user_model, user_blend, user_sharpen, user_chunk, user_scale), vhs_info


def check_output(a):
    """Output name, folder and free space, checked before hours of work."""
    if a.output is None:
        out = default_output(a.input, height=a.height)
        # a preview gets its own name: it must never replace the finished movie
        a.output = str(out.with_name(out.stem + " test.mkv") if a.test else out)
    if not a.analyze and Path(a.output).suffix.lower() not in (".mkv", ".mp4", ".m4v"):
        sys.exit("Output must be .mkv (keeps everything) or .mp4 (for Apple devices/browsers).")
    if a.test and not a.analyze and Path(a.output).exists() and \
            (media_duration(Path(a.output)) or 0) > a.test + 5:
        sys.exit(f"'{a.output}' is already a longer movie (a finished upscale?) and a {a.test} s "
                 "preview would replace it. Give the preview its own name, or leave the output "
                 "name out (then it's saved as '... test.mkv').")
    if not a.analyze and Path(a.output).is_file():
        was = made_from(a.output)
        if was is None:
            sys.exit(f"'{a.output}' already exists and wasn't made by this script: give the "
                     "output another name (or move that file away first).")
        if was and was != source_id(a.input):
            sys.exit(f"'{a.output}' is the finished upscale of another file ({was}): give this "
                     "one another output name (or delete that file first if it should be "
                     "replaced).")
    if not a.analyze:
        out_dir = Path(a.output).resolve().parent
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            sys.exit(f"Can't create the output folder '{Path(a.output).parent}': {e}")
        # the output file is only written at the very end, after the chunks are joined: a
        # folder that can't be written to would fail after hours of upscaling. A small file made
        # and removed now shows it (read-only folder, a drive that went away, a locked share)
        probe_file = out_dir / f".write_test_{os.getpid()}.tmp"
        try:
            probe_file.write_text("test", encoding="utf-8")
        except OSError as e:
            sys.exit(f"Can't write to the output folder '{out_dir}': {e}\n"
                     "Give the output a name in another folder (the second argument, "
                     'for example "D:\\Movies\\Movie 1080p.mkv").')
        finally:
            probe_file.unlink(missing_ok=True)
        # the finished movie is about 10 Mbit/s (more for --hevc 10-bit: less): its size is
        # a guess, a warning only
        try:
            secs = a.test if a.test else (media_duration(Path(a.input)) or 0)
            need = max(1e9, secs * 1.25e6 * 1.3)
            free = shutil.disk_usage(out_dir).free
            if free < need:
                print(f"WARNING: the drive of the output folder has {free / 1e9:.1f} GB free; "
                      f"this movie needs about {need / 1e9:.1f} GB there at the end.")
        except (OSError, TypeError, ValueError):
            pass            # (a share that can't say how much is free)
        print(f"Saving to: {a.output}")


def pick_model(a, find, user):
    """The upscaler model actually used: the preset's, or the next best installed one."""
    user_model, user_blend, user_sharpen, user_chunk, user_scale = user
    tools = ["ffmpeg", "ffprobe"] + ([] if a.fast or a.analyze else [a.esrgan])
    for t in tools:
        if not find(t):
            sys.exit(f"Missing tool: {t}")
    if not a.fast and not a.analyze:
        a.esrgan_path = find(a.esrgan)
        if a.trained_pick:
            a.model = trained_model(a)
            if a.scale == 4 and user_chunk is None:
                a.chunk_frames = min(a.chunk_frames, 480)      # (x4 frames: as for the other x4 models)
        best = ncnn_saved().get("best_model")
        if a.best_quality and not best:
            print("NOTE: --best-quality: no best model is saved yet: run "
                  "python dvd_upscale.py --ncnn-models \"Movie.mkv\" --save first. "
                  "Using the usual model.")
        if a.best_quality and best and user_model is None and a.type in ("cgi", "live") \
                and best != a.model and previous_settings(a) is None:
            # (--ncnn-models --save found a model truer to the original for these types)
            keep = (a.model, a.scale)
            a.model, a.scale = best, 2 if "x2plus" in best else 4
            if model_installed(a):
                print(f"Best-quality model for {TYPE_NAMES[a.type]}: {best} "
                      f"(found by --ncnn-models --save, used because of --best-quality)")
                if "x4plus" in best:           # tame its face artifacts, as for the fallback
                    a.ai_blend = 0.5 if user_blend is None else user_blend
                    a.sharpen = 0.2 if user_sharpen is None else user_sharpen
                    a.chunk_frames = user_chunk or 480
            else:
                a.model, a.scale = keep
        if a.model == "realesr-animevideov3" and user_model is None and not model_installed(a):
            # a folder set up for live action only: the next best model that is there
            for m in ("realesrgan-x2plus", "realesrgan-x4plus"):
                a.model, a.scale = m, 2 if "x2plus" in m else 4
                if model_installed(a):
                    print(f"NOTE: realesr-animevideov3-x2.param/.bin not found in {models_dir(a)} "
                          "(they come with the official Real-ESRGAN ncnn download: copy them in). "
                          f"Using {m} until then.")
                    if m == "realesrgan-x4plus":          # tame its face artifacts, as for live
                        a.ai_blend = 0.5 if user_blend is None else user_blend
                        a.sharpen = 0.2 if user_sharpen is None else user_sharpen
                        # x4 frames: 3x the temporary PNGs of x2, so the live preset's chunk size
                        a.chunk_frames = user_chunk or 480
                    break
            else:
                a.model, a.scale = "realesr-animevideov3", 2     # reported as missing below
        if a.type == "vhs" and user_model is None and not model_installed(a):
            # missing model files: the next best ones that are there
            wanted = a.model
            for m in ((["realesr-general-x4v3"] if wanted == VHS_VIDEO[0] else [])
                      + ["realesrgan-x2plus", "realesrgan-x4plus"]):
                a.model, a.scale = m, 2 if "x2plus" in m else 4
                if model_installed(a):
                    break
            else:
                sys.exit(f"No upscaler model for VHS in {models_dir(a)}: put "
                         "realesrgan-x2plus.param/.bin (from realesrgan-x2plus.zip, for movie "
                         "tapes) and realesr-general-dn50-x4v3.param/.bin (from "
                         "realesr-general-dn50-x4v3.zip, for camcorder tapes) in that folder.")
            print(f"NOTE: {wanted}.param/.bin not found in {models_dir(a)}: unzip "
                  f"{VHS_MODEL_ZIPS[wanted]} into that folder. Using {a.model} until then.")
            if a.model == "realesrgan-x4plus":          # tame its face artifacts, as for live
                a.ai_blend = 0.5 if user_blend is None else user_blend
                a.sharpen = 0.2 if user_sharpen is None else user_sharpen
        if not model_installed(a):
            if a.type in ("live", "cgi") and user_model is None:
                # x2plus files missing: use x4plus with settings that tame its face artifacts
                print(f"NOTE: {a.model} model files not found in {models_dir(a)}, using "
                      "realesrgan-x4plus instead (faces look better with x2plus: put "
                      "realesrgan-x2plus.param/.bin in that folder).")
                a.model, a.scale = "realesrgan-x4plus", 4
                a.ai_blend = 0.5 if user_blend is None else user_blend
                a.sharpen = 0.2 if user_sharpen is None else user_sharpen
            if not model_installed(a):
                sys.exit(f"Model '{a.model}' (x{a.scale}) not found in {models_dir(a)}")
        old = previous_settings(a)
        if (old and user_model is None and user_scale is None and old.get("model")
                and old["model"] != a.model and old.get("denoise") == a.denoise):
            # started with a substitute model (or the other way round): finish with that one,
            # as chunks from two models would differ
            now = a.model, a.scale
            a.model, a.scale = old["model"], old.get("scale") or a.scale
            if model_installed(a):
                a.ai_blend = old.get("ai_blend", a.ai_blend) if user_blend is None else a.ai_blend
                a.sharpen = old.get("sharpen", a.sharpen) if user_sharpen is None else a.sharpen
                a.chunk_frames = user_chunk or old.get("chunk") or a.chunk_frames
                print(f"Keeping {a.model}, which this movie was started with (a new movie, or "
                      f"this one started over, gets {now[0]})")
            else:
                a.model, a.scale = now
        if "x4plus" in a.model and a.scale != 4:
            sys.exit("realesrgan-x4plus models need --scale 4")
        if "x2plus" in a.model and a.scale != 2:
            sys.exit("realesrgan-x2plus needs --scale 2")


def pick_engine(a):
    """Which engine runs the model, and how much the GPU is given at once."""
    if not a.fast and not a.analyze:
        # the big models: the current ncnn if it is there (see ncnn_upscaler_main); the small
        # anime/camcorder ones stay on realesrgan-ncnn-vulkan, which runs them fine
        a.engine = "exe"
        if not compact_model(a) and a.engine_choice != "exe":
            if ncnn_available() and (a.engine_choice == "ncnn"
                                     or a.esrgan == ESRGAN_DEFAULT):
                a.engine = "ncnn"
            elif a.engine_choice == "ncnn":
                sys.exit("--engine ncnn: the ncnn Python module isn't installed: "
                         + ncnn_install_hint())
            elif a.esrgan == ESRGAN_DEFAULT:
                print(f"NOTE: for {a.model}, '{ncnn_install_hint()}' gives the upscaler a current "
                      "GPU engine: the one built into realesrgan-ncnn-vulkan (2022) makes NVIDIA "
                      "drivers from 570 on reset the GPU now and then (\"vkQueueSubmit failed -4\")")
    if not a.fast:
        kind = (a.type if a.type != "vhs" else
                f"vhs {'movie tape' if a.mode == 'telecine' else 'camcorder tape'}")
        print(f"Upscaler: {a.model} x{a.scale} ({kind} preset)"
              + (", run by the current ncnn (pip)" if getattr(a, "engine", "exe") == "ncnn"
                 else ""))
        load_gpu_step(a)        # (as far down as earlier runs had to go, see gpu_step_file)
        if a.gpu_jobs is None:          # (see --gpu-jobs)
            # one: the small models keep the GPU busy at 8 frames at once (two anime upscalers
            # starved NVENC on a laptop), and where the big ones get the GPU reset, more work at
            # once makes it likelier (2 frames at once failed several times sooner than 1)
            # (measured on an RTX 3060 laptop with the current ncnn, whole frames: 179 s a chunk
            # with one upscaler, 204 with two: --gpu-jobs 2 is slower there)
            a.gpu_jobs = 1
        jobs_note = (f"up to {a.gpu_jobs} upscalers per GPU, a second kept only if faster"
                     if a.gpu_jobs > 1 else "one upscaler per GPU")
        print(f"GPU settings: {gpu_load_text(a)}; {jobs_note}"
              + (f" (what this GPU needed before; to try more again, delete "
                 f"{gpu_step_file().name} next to {Path(__file__).name})" if GPU_STEP[0] else ""))


def pick_faces(a):
    """Face restoration on or off for this movie."""
    # face restoration: on by itself for live action and tapes (when its packages and model
    # files are installed: else the movie goes on without, see faces_check), only when asked
    # for 3D animation (its model was trained on photos: animated faces can turn photographic),
    # never for anime. A movie started without it keeps going without it (its chunks must match)
    a.faces_auto = False
    if a.no_faces:
        a.faces = None
    elif a.faces is None and a.type in ("live", "vhs") and not a.fast and not a.analyze \
            and a.ai_blend > 0:
        prev = previous_settings(a)
        if prev is None or "faces" in prev:
            a.faces, a.faces_auto = FACE_STRENGTH, True
    if a.faces is not None and (a.fast or a.type == "anime" or a.ai_blend == 0):
        # (--all --faces on a folder of all kinds of movies: anime gets none)
        if not a.analyze:
            print("NOTE: --faces isn't used " + (
                "with --fast (it works on the AI-upscaled frames)" if a.fast else
                f"for {TYPE_NAMES[a.type]}: it would turn drawn faces into photographic ones"
                if a.type == "anime" else
                "with --ai-blend 0 (it works on the AI-upscaled frames, none of which are used "
                "then)"))
        a.faces = None
    elif a.faces is not None and a.type == "cgi" and not a.analyze:
        print("NOTE: --faces on 3D animation: its model was trained on photos, so characters' "
              "faces can come out photographic. Check a clip first (--clip ... --upscale)")


def read_source(a, vhs_info):
    """The source's size, aspect and frame rate; the output size from them."""
    info = vhs_info if a.type == "vhs" else probe_or_exit(a.input)
    sar_txt = f"{info['sar'].numerator}:{info['sar'].denominator}"
    print(f"Source: {info['w']}x{info['h']}, SAR {sar_txt}, "
          f"{float(info['fps']):.3f} fps, {info['duration']/60:.1f} min")
    a.matrix = "bt601-6-625" if info["h"] == 576 else "bt601-6-525"
    a.src_hd, a.pal = info["h"] > 576, info["h"] in (288, 576)
    if a.src_hd and info["h"] >= a.height:
        msg = (f"This video is already {info['h']} lines tall: upscaling it to {a.height}p "
               "would only re-encode it (and cost quality).")
        if not a.analyze:
            sys.exit(msg + " Nothing to do.")
        print("NOTE: " + msg)
    if a.type == "vhs":
        # 4:3 is the 704-sample active width (BT.601), whatever aspect the capture is flagged with
        a.dar = a.dar or "4/3"
        a.vhs_crop, a.vhs_sw, a.vhs_sh = vhs_geometry(info, a.dar)
        if not a.fast and a.scale == 4:
            # x4 model: 3/4 size in (480x360 -> 1920x1440 -> 1440x1080). Tape has no detail that
            # fine: measured better (LPIPS 0.249 vs 0.302), 0.56x the AI work and temp PNGs
            a.vhs_sw, a.vhs_sh = a.vhs_sw * 3 // 4 // 2 * 2, a.vhs_sh * 3 // 4 // 2 * 2
        a.vhs_h, a.vhs_trim = info["h"], 0

    out_ratio = (float(Fraction(a.dar)) if a.dar else info["w"] * float(info["sar"]) / info["h"])
    a.out_w = max(8, round(a.height * out_ratio / 8) * 8)
    if a.out_w > 1920 and a.height == 1080:
        # wider than 16:9 (e.g. ITU 40:33 pixels): stay 1920 wide, the H.264 level 4.1 and TV
        # limit, by giving up a few lines of height
        a.out_w, a.height = 1920, round(1920 / out_ratio / 2) * 2
    a.level = "4.1" if -(-a.out_w // 16) * -(-a.height // 16) <= 8192 else "5.1"
    if min(abs(out_ratio - 4 / 3), abs(out_ratio - 16 / 9)) > 0.05 * out_ratio:
        print(f"WARNING: output aspect would be {out_ratio:.2f}:1, which is odd. "
              "If the picture looks squeezed/stretched, pass --dar 16:9 or --dar 4:3")
    return info


def detect_frames(a, info):
    """Film or video (telecine / progressive / interlaced), and the real frame rate."""
    measured, kept, regular = (None, 1.0, 0.0) if a.type == "vhs" else cadence(a, info)
    if measured:
        print(f"Actual frames in file: {float(measured):.3f} fps")
    if a.type == "vhs":
        what = {"telecine": "movie tape (film with 3:2 pulldown), inverse telecine",
                "interlaced": "camcorder/TV video, one frame per field",
                "progressive": "not interlaced (deinterlaced before?), used as it is"}[a.mode]
        print(f"Detected: {a.mode} - {what} ({a.vhs_detail})")
        if a.mode == "interlaced" and a.vhs_auto:
            print("  (a movie or cartoon tape whose sample is mostly still pictures looks like "
                  "this too: for a movie or cartoon, use --mode telecine)")
        if a.chroma_delay == "auto":
            dx = vhs_chroma_delay(a, info)
            # undone in whole pixels, within what tapes show; vertically the deck's comb filter
            # puts the colour about a line low on every tape (too faint to measure reliably)
            # (not measurable: a typical 3 pixels of a 704-wide picture)
            dflt = max(1, round(3 * min(info["w"], 704) / 704))
            sx = -dflt if dx is None else -max(-2, min(6, round(dx)))
            a.chroma_shift = f"cbh={sx}:crh={sx}:cbv=-1:crv=-1"
            got = f"not measurable in the sample, assuming {dflt}" if dx is None else f"{dx:.1f}"
            print(f"Colour offset: {got} px right, 1 line down -> chromashift={a.chroma_shift}")
    elif a.mode == "auto":
        a.mode, detail = detect_mode(a, info)
        if (a.mode in ("progressive", "interlaced") and measured == Fraction(30000, 1001)
                and 0.74 <= kept <= 0.86):
            if regular >= 0.7:
                a.mode = "telecine"       # 24p padded to 30p with repeated frames
                detail += f", {1 - kept:.0%} of frames are repeats in a steady 1-in-5 pattern"
            else:
                detail += (f", {1 - kept:.0%} of frames are repeats but irregular "
                           "(held animation frames?) - if motion judders, try --mode telecine")
        inter = re.search(r"interlaced frames (\d+)%", detail)
        combed = bool(inter) and int(inter.group(1)) >= 3
        if a.mode != "telecine" and not a.pal and abs(float(info["fps"]) - 30000 / 1001) < 0.1:
            # 29.97 fps: hard-telecined film (TV animation, many TV series) can look like video
            # or even progressive in one sample: a soft picture or held drawings hide idet's
            # repeated fields, and a quiet minute shows little combing. The field-by-field
            # cadence check the VHS handling uses tells film from video; tried at up to three
            # points of the movie (measured on Avatar discs: episodes of one season were
            # sampled as telecine, as 29% interlaced and as 0% interlaced)
            d = info["duration"]
            for at in [sample_start(info)] + ([d * 0.5, d * 0.8] if d > 600 else []):
                film, _, why = vhs_detect(a, info, at)
                if film == "telecine":
                    a.mode = "telecine"
                    detail += f"; at {at / 60:.0f} min: " + why.split("), ")[-1]
                    break
                m = re.search(r"idet (\d+) TFF / (\d+) BFF / (\d+) progressive", why)
                if m and int(m[1]) + int(m[2]) >= 0.03 * max(1, sum(map(int, m.groups()))):
                    combed = True               # interlaced stretches seen there
        if not a.pal and a.mode != "telecine":
            mix = pulldown_mix(a.input)
            if mix and mix[0] >= 0.01 and mix[1] >= 10:
                a.mode = "telecine"
                detail += (f"; the whole file mixes film stored with soft pulldown and "
                           f"{mix[1] / 60:.1f} min of 29.97 fps (hard-telecined film or video): "
                           "inverse telecine for all of it")
        if a.mode == "progressive" and info["fps"] > Fraction(241, 10) and combed:
            # TV animation and the like: progressive pictures, but pans, zooms and dissolves
            # done as interlaced video. Left alone those frames keep comb lines (sawtooth edges)
            # that the AI upscale makes worse
            a.combed = True
            detail += "; the interlaced stretches get deinterlaced, the rest is left untouched"
        old = previous_settings(a)
        if old and old.get("mode") in ("telecine", "progressive", "interlaced") \
                and old["mode"] != a.mode:
            print(f"Detected now: {a.mode} ({detail})")
            print(f"Mode: {old['mode']}, kept from the earlier run in "
                  f"'{a.work + ('_test' if a.test else '')}' so the movie stays the same "
                  f"throughout. To redo the whole movie as {a.mode} (likely better), delete "
                  "that folder.")
            a.mode = old["mode"]
        else:
            print(f"Detected: {a.mode} ({detail})")
    else:
        print(f"Mode: {a.mode} (forced)")
    if a.type != "vhs" and a.mode == "telecine" and not a.pal:
        mix = pulldown_mix(a.input)         # (None: not MPEG-2, or not readable: repeatfields stays)
        a.no_soft_pulldown = mix is not None and mix[0] == 0
    return measured


def check_ai(a):
    """The upscaler and the face restoration, tried before the movie starts."""
    if not a.fast:
        check_upscaler(a)       # (exits if the model's files are broken or the GPU fails)
    if a.faces:
        try:
            faces_check(a)      # (exits with what to install or download if something's missing)
        except FacesUnavailable as e:
            # (turned on by itself: the movie goes on without it, and says how to get it)
            first, _, rest = str(e).partition("\n")
            print("NOTE: face restoration is skipped for this movie: "
                  + first.replace("--faces needs", "it needs").rstrip(".") + "."
                  + (f"\n{rest}" if rest else "")
                  + "\n  (This movie goes on without faces, and is saved that way: to add them "
                    "later, install what is missing and start the movie over by deleting its "
                    "_work folder. --no-faces hides this note.)", flush=True)
            a.faces = a.faces_auto = None


def pick_fps(a, info, measured):
    """The output frame rate (and the VHS chunk rounding and warm-up)."""
    if a.fps is None and a.type == "vhs":
        if a.mode == "telecine":
            fps = a.vhs_fin * 4 / 5 if a.vhs_fin > 26 else a.vhs_fin    # PAL film is 2:2
        elif a.mode == "interlaced":
            fps = a.vhs_fin * 2       # one frame per field, no halving: motion as smooth as on tape
        else:
            fps = a.vhs_fin
    elif a.fps is None:
        fps = ((Fraction(25) if a.pal else Fraction(24000, 1001)) if a.mode == "telecine"
               else (measured or info["fps"]))
        if fps > 31:
            fps /= 2
        if fps <= 0:
            fps = Fraction(24000, 1001) if a.mode == "progressive" else Fraction(30000, 1001)
        fps = fps.limit_denominator(1001)
        old = previous_settings(a)
        if old and old.get("mode") == a.mode and frac(old.get("fps")) > 0 \
                and frac(old.get("fps")) != fps:
            # started by an older version that picked another rate: finish it the same way
            # (chunks at two frame rates can't be joined)
            print(f"Frame rate: {float(frac(old['fps'])):.3f} fps, kept from the earlier run "
                  f"(this version picks {float(fps):.3f}; delete the work folder to redo it)")
            fps = frac(old["fps"])
    else:
        fps = given = frac(a.fps)
        for exact in (Fraction(24000, 1001), Fraction(30000, 1001), Fraction(60000, 1001)):
            if fps != exact and abs(fps - exact) < Fraction(1, 100):
                fps = exact                 # 23.976 / 29.97 / 59.94 typed as decimals
        old = previous_settings(a)
        if old and frac(old.get("fps")) == given:
            fps = given                     # started that way by an older version
    a.fps = f"{fps.numerator}/{fps.denominator}"
    if fps > 30 and a.level == "4.1":
        # 1440x1080p59.94 is 366,833 macroblocks/s, level 4.1 allows 245,760 (camcorder tapes,
        # or --fps 50/59.94 on a DVD: 1080p59.94 is 489,110)
        a.level = "4.2"
    if a.type == "vhs":
        # every chunk must start on a whole tape frame (4 film frames = 5 tape frames,
        # 2 fields = 1 tape frame), else the cut repeats or drops part of one
        unit = (fps / a.vhs_fin).numerator
        if unit <= 8 and a.chunk_frames % unit:
            a.chunk_frames = max(unit, a.chunk_frames // unit * unit)
            print(f"Chunk size rounded to {a.chunk_frames} frames (whole tape frames)")
        # chunk warm-up (Chunk): two whole 5-frame cycles for the IVTC, else 6 tape frames
        pre = 10 if a.mode == "telecine" else 6
        trim = pre * fps / a.vhs_fin
        a.vhs_warm = (pre, int(trim)) if trim.denominator == 1 else (0, 0)
    return fps


def setup_stabilize(a, info, fps):
    """--stabilize: the smoothing and zoom settings."""
    if a.stabilize:
        if "vidstabtransform" not in capture(["ffmpeg", "-hide_banner", "-filters"])[1]:
            sys.exit("--stabilize needs an ffmpeg with vid.stab (the vidstabdetect and "
                     "vidstabtransform filters): on Windows the gyan.dev \"full\" build has "
                     "them, the \"essentials\" one doesn't")
        if a.type != "vhs":
            print("NOTE: --stabilize is meant for camcorder and home videos: a film's camera "
                  "moves are wanted, and steadying them zooms into the picture")
        # smoothing over half a second each way (strong: a second), zoomed in just enough that
        # the largest correction allowed (maxshift) shows no edge; crop=black and a fixed zoom
        # keep nothing from one frame to the next, so each chunk comes out as one run would
        secs, zoom, angle = {"normal": (0.5, 4, 0.02), "strong": (1.0, 8, 0.04)}[a.stabilize]
        mu = max(1, round(secs * float(fps)))
        width = (a.vhs_sw if a.type == "vhs" else
                 int(info["h"] * float(Fraction(a.dar)) / 2) * 2 if a.dar else
                 int(info["w"] * float(info["sar"]) / 2) * 2)
        a.stab_tf = (f"smoothing={mu}:optalgo=gauss:optzoom=0:zoom={zoom}:crop=black:"
                     f"interpol=bicubic:maxshift={round(width * zoom / 200)}:maxangle={angle}")
        # frames read before (and motion taken after) each chunk: more than the smoothing's
        # reach, in whole multiples of 4 (whole 3:2 cycles, whole tape frames)
        a.stab_margin = -(-(mu + 2) // 4) * 4


def pick_encoder(a, fps):
    """NVENC or the CPU, and the encoder settings."""
    a.gop = round(4 * float(fps))
    old = previous_settings(a)
    if old and not a.cpu and old.get("enc") == " ".join(cpu_args(a)):
        # chunks from the two encoders can't be joined into one video
        a.cpu = True
        print("Encoding on the CPU, as this movie was started (NVENC wasn't usable then).")
    a.enc = None if a.cpu else pick_nvenc(a)
    if a.enc is None:
        if not a.cpu:
            print("NVENC not usable here, falling back to CPU encoding (much slower).")
            if a.nvenc_error:
                print(f"  reason: {a.nvenc_error}\n  (an old NVIDIA driver is the usual cause; "
                      "also check the laptop isn't in iGPU/Eco mode)")
        a.cpu, a.enc = True, cpu_args(a)
    elif a.enc != nvenc_args(a)[0]:
        print("Using simpler NVENC settings (this GPU doesn't support all quality options).")
    rate = f", {float(fps):.3f} fps"
    print(f"Output: {a.out_w}x{a.height} {'HEVC 10-bit' if a.hevc else 'H.264'}{rate}, "
          f"keyframe every {a.gop} frames")


def open_work(a, info):
    """The work folder: locked, its settings.json checked against this run (a resumed movie must
    be made the same way) and written; returns the folder and the source's stat."""
    work = Path(a.work + ("_test" if a.test else ""))
    work.mkdir(parents=True, exist_ok=True)
    a.lock = lock_work(work)
    protect_run(Path(a.input).name)
    st = Path(a.input).stat()
    # encoder args are included so a resumed run never mixes chunks from different encoder
    # settings (their stream headers differ, and joining them breaks playback)
    # colour interpolated into the RGB pictures for the upscaler (see Chunk.extract); a movie
    # started before that keeps its way, so its chunks match
    prev = previous_settings(a)
    a.rgb_interp = not a.fast and not (prev and "rgb" not in prev)
    # less AI in crowds and other busy detail (see DETAIL_BLEND): not for drawn animation (its
    # ink lines are dense detail the anime model is made for), and a movie started before keeps
    # its way, so its chunks match
    a.detail_blend = (not a.fast and a.type != "anime" and a.ai_blend < 1
                      and not a.no_detail_blend and not (prev and "detail_blend" not in prev))
    # black bars of a widescreen movie: cut off before the upscale, put back after it (see
    # detect_bars). A movie started earlier keeps what it started with (its chunks must match)
    a.crop_rows, a.crop_src_h = 0, info["h"]
    if not a.fast and a.type != "vhs" and not a.no_crop and info["h"] <= 576:
        if prev is not None:
            a.crop_rows = int(prev.get("crop", 0))
            if a.crop_rows and not bar_rows_valid(a, info["h"], a.crop_rows):
                # (--height changed since: the settings then differ, and the run says so)
                a.crop_rows = 0
        else:
            status_line("  looking for black bars...")
            a.crop_rows = detect_bars(a, info)
            status_line()
            if a.crop_rows:
                print(f"Black bars: {a.crop_rows} rows at the top and at the bottom are cut "
                      f"before the upscale and put back after it ({info['h'] - 2 * a.crop_rows} "
                      f"of {info['h']} rows upscaled: faster, the same picture; --no-crop turns "
                      "it off)")
            else:
                print("Black bars: none found (the picture fills the frame)")
    fp = dict(input=str(Path(a.input).resolve()), size=st.st_size, mtime=int(st.st_mtime),
              mode=a.mode, fps=a.fps, model=a.model, scale=a.scale, height=a.height,
              chunk=a.chunk_frames, fast=a.fast, dar=a.dar, test=a.test, enc=" ".join(a.enc), w=a.out_w,
              denoise=a.denoise, ai_blend=a.ai_blend, smooth=a.smooth, sharpen=a.sharpen,
              **({"rgb": "interp"} if a.rgb_interp else {}),
              **({"detail_blend": "{sigma}:{lo}:{hi}:{low}".format(
                  **{**DETAIL_BLEND, "low": DETAIL_BLEND["low_blend"]
                     if a.busy_blend is None else a.busy_blend})}
                 if a.detail_blend else {}),
              **({"crop": a.crop_rows} if a.crop_rows else {}),
              **({"stabilize": a.stab_tf} if a.stabilize else {}),
              # (CodeFormer's fidelity: GFPGAN has none)
              **({"faces": f"{a.face_model} {a.faces:g}" + (
                  f" {FACE_FIDELITY:g}" if a.face_model == "codeformer" else "")}
                 if a.faces else {}),
              # VHS: field order, crop, colour shift, mask, sizes, warm-up (anime/live unchanged)
              **({"vhs": f"{prefilter(a)} warm={a.vhs_warm[0]}:{a.vhs_warm[1]}"}
                 if a.type == "vhs" else {}))
    sf = work / "settings.json"
    try:
        old = json.loads(sf.read_text()) if sf.exists() else None
    except ValueError:
        sys.exit(f"'{sf}' is damaged (empty?): delete the folder '{work}' to start this movie "
                 "over.")
    if old and old.get("input") != fp["input"] and \
            Path(old.get("input", "")).name == Path(fp["input"]).name and \
            (old.get("size"), old.get("mtime")) == (fp["size"], fp["mtime"]):
        old["input"] = fp["input"]          # the same file, moved (or another drive letter)
    if old is not None and old != fp:
        # (both ways: a setting only the first run had, such as --stabilize, counts too)
        changed = [k for k in [*fp, *(k for k in old if k not in fp)]
                   if old.get(k) != fp.get(k)]

        def codec(enc):
            w = str(enc).split()
            return w[w.index("-c:v") + 1] if "-c:v" in w[:-1] else enc
        was, now = codec(old.get("enc")), codec(fp["enc"])
        hevc = lambda c: "hevc" in str(c) or "265" in str(c)
        hint = ""
        if "enc" in changed and hevc(was) != hevc(now):
            hint = (f" (the first run was {'HEVC' if hevc(was) else 'H.264'}: "
                    f"{'add' if hevc(was) else 'leave out'} --hevc to continue it)")
        elif "enc" in changed and ("nvenc" in was) != ("nvenc" in now):
            hint = (" (the first run used the "
                    f"{'NVIDIA' if 'nvenc' in was else 'CPU'} encoder: "
                    + ("NVENC isn't usable right now - check the driver / that the laptop isn't "
                       "in iGPU mode" if "nvenc" in was else "leave out --cpu") + ")")
        queue = os.environ.get("DVD_UPSCALE_QUEUE")
        if "mode" in changed and old.get("mode") in ("telecine", "interlaced", "progressive") \
                and not queue:
            hint += f" (to continue it as it was started, add --mode {old['mode']})"
        if "faces" in changed:
            was_faces = str(old.get("faces") or "").split()
            hint += (" (it was started without face restoration: leave out --faces, or add "
                     "--no-faces)" if not was_faces else
                     f" (it was started with face restoration (--faces {was_faces[1]}"
                     + (f" --face-model {was_faces[0]}" if was_faces[0] != FACE_MODEL else "")
                     + (f", CodeFormer fidelity {was_faces[2]}" if len(was_faces) > 2
                        and float(was_faces[2]) != FACE_FIDELITY else "")
                     + "): to continue it, leave out --no-faces and any other --faces / "
                       "--face-model, with its packages and model files working)")
        if "detail_blend" in changed:
            was_db = str(old.get("detail_blend") or "").split(":")
            hint += (" (it was started without the crowd/busy-detail blend: add "
                     "--no-detail-blend)" if len(was_db) < 4 else
                     f" (it was started with --busy-blend {was_db[3]})"
                     if fp.get("detail_blend") else
                     " (it was started with the crowd/busy-detail blend: leave out "
                     "--no-detail-blend)")
        sys.exit(f"Settings or input changed since the last run in '{work}': "
                 f"{', '.join(changed)}{hint}. "
                 + (f"Delete the folder '{work.resolve()}' to start this movie fresh." if queue
                    else "Delete that folder (or use --work NEWNAME) to start fresh."))
    write_durably(sf, json.dumps(fp))
    # what it was detected as: a resumed run keeps it, and the --all/--queue log shows it
    if a.detected or not (work / "detected.json").exists():
        write_durably(work / "detected.json", json.dumps(dict(type=a.type, **a.detected)))
    return work, st


def plan_chunks(a, info, fps):
    """The chunks: (index, start, length, frames) on the video's own timeline."""
    # video may start later than the container (audio-first files): cut on the video's own grid
    vo = info["vstart"] - info["cstart"]
    eps = 0.5 / float(info["fps"]) if info["fps"] > 0 else 0.015   # half a source frame
    vlen = info["vduration"] or max(0.0, info["duration"] - vo)    # prefer the video's own length
    total = Fraction(vlen).limit_denominator(1000)
    if a.test:
        total = min(Fraction(a.test), total)
    step = Fraction(a.chunk_frames) / fps
    plan = []
    for i in range(math.ceil(total / step)):
        t = step * i
        length = min(step, total - t)
        expected = a.chunk_frames if length == step else int(round(length * fps))
        if expected >= 1:
            plan.append((i, t, length, expected))
    return plan, total, vo, eps, vlen


def chunk_source(a, info, work, st, plan, total, vo, eps, vlen):
    """What the chunks are cut from (an exact-seek copy of an MPEG file's video), and --stabilize's
    camera-motion file; returns where the video starts in it."""
    # MPEG program/transport streams (capture cards, DVD recorders, .vob): they have no index, so
    # -ss can land frames off (measured on 100 s MPEG-2 captures: up to 15 frames late with
    # ffmpeg 6.1, up to 12 with a 2026 build), repeating frames at chunk starts and moving the
    # picture against the sound. Chunks are cut from a stream copy of the video in .mkv instead,
    # which seeks exactly; its timeline starts at the source's first video frame.
    # (full paths: a --stabilize chunk's ffmpeg runs in its own folder)
    a.chunk_input, cut = str(Path(a.input).resolve()), vo
    if Path(a.input).suffix.lower() in MPEG_EXT:
        idx = work / "video_index.mkv"
        if not idx.exists():
            gb = st.st_size / 1e9 * (min(1, (a.test + 10) / vlen) if a.test and vlen else 1)
            print(f"Copying the MPEG file's video into the work folder for exact cuts "
                  f"(about {gb:.1f} GB)...")
            tmp = work / "video_index.part.mkv"
            run_step(["ffmpeg", "-y", "-v", "error", "-fflags", "+genpts", "-i", a.input,
                      "-map", "0:v:0", "-c", "copy", *(["-t", a.test + 10] if a.test else []),
                      tmp], min(vlen, a.test + 10) if a.test else vlen, "copying the video")
            flush_to_disk(tmp)
            replace_file(tmp, idx)
        a.chunk_input, cut = str(idx.resolve()), 0.0
    if a.stabilize:
        # the camera motion of the whole video, measured once on the frames the chunks are made
        # of (the same filters, from the same first frame, without the chunks' warm-up), and kept
        # in the work folder for a resumed run
        a.stab_file = work.resolve() / "stab.trf"
        n_frames = sum(e for *_, e in plan)
        if not a.stab_file.exists():
            part = a.stab_file.with_name("stab.part.trf")
            a.vhs_trim = a.dvd_trim = 0
            seek0 = max(0.0, cut - eps)
            print("Measuring the camera shake for --stabilize (once for this movie):", flush=True)
            run_step(["ffmpeg", "-y", "-v", "error",
                      *(["-ss", f"{seek0:.6f}"] if seek0 > 0 else []), "-i", a.chunk_input,
                      "-an", "-sn", "-vf", prefilter(a) + f",vidstabdetect=result={part.name}"
                      ":shakiness=6:accuracy=15:stepsize=6", "-frames:v", n_frames,
                      "-f", "null", "-"], float(total), "measuring the camera shake",
                     cwd=a.stab_file.parent)
            flush_to_disk(part)
            replace_file(part, a.stab_file)
        a.stab_index = stab_index(a.stab_file)
        if len(a.stab_index[1]) < n_frames - 2:
            print(f"NOTE: the camera shake was measured on {len(a.stab_index[1])} of {n_frames} "
                  "frames; the rest is left as it is")
    return cut


def upscale_chunks(a, info, fps, work, plan, cut, eps):
    """Every chunk made (extract, upscale, encode, overlapped; other upscalers alongside);
    returns the chunk files in order."""
    try:            # set by --all / --queue: which movie this is, and how much video comes after
        queue = json.loads(os.environ.get("DVD_UPSCALE_QUEUE") or "null")
        later = float(queue["later_secs"]) if queue else 0.0
    except (ValueError, TypeError, KeyError):
        queue, later = None, 0.0
    # a crash or power cut can leave the chunks written just before it cut short (versions
    # before this one didn't flush them to disk first): the newest three are read through once,
    # and a damaged one is made again (ffmpeg exits 0 on a cut-short .mkv, but reports it)
    for p in sorted((p for p in (work / f"chunk_{i:05d}.mkv" for i, *_ in plan) if p.exists()),
                    key=lambda p: p.stat().st_mtime)[-3:]:
        rc, txt = capture(["ffmpeg", "-v", "error", "-i", p, "-map", "0:v", "-c", "copy",
                           "-f", "null", "-"])
        if rc or txt.strip() or not count_frames(p):
            print(f"{p.name} is damaged (cut short by a crash?): it is made again")
            p.unlink()
    done_before = sum((work / f"chunk_{i:05d}.mkv").exists() for i, *_ in plan)
    if done_before:
        print(f"Resuming: {done_before} of {len(plan)} chunks already done")
    chunks, notes = {}, []
    # three steps per chunk, overlapped: while chunk i is upscaled on the GPU, chunk i+1's frames
    # are read and chunk i-1 is encoded in the background, so the GPU doesn't wait for either.
    # More upscalers (--gpu-jobs, and the other GPUs of --gpu 0,1) each take whole chunks
    # alongside (see helper below). A failure on the main upscaler stops the movie as before
    todo = [entry for entry in plan if not (work / f"chunk_{entry[0]:05d}.mkv").exists()]
    for entry in plan:
        if entry not in todo:
            chunks[entry[0]] = work / f"chunk_{entry[0]:05d}.mkv"     # done in an earlier run
    waiting = list(todo)                    # the chunks no GPU has taken yet
    lock, make_lock = threading.Lock(), threading.Lock()
    devices = gpu_list(a)
    a.stop_lanes = threading.Event()
    a.stopping = threading.Event()          # the movie stops (Ctrl+C, a failure): see below
    a.cancel_lanes = set()                  # helpers whose chunk the main GPU took back
    main_secs, skipped, shown = [], set(), [0]  # main GPU's upscale times; chunks with no video
    progress = {"chunks": 0, "video": 0.0, "helped": 0, "frames": 0}
    t_start = time.time()
    encodings, reading, current = [], None, None    # [(Background, Chunk)] / (Background, Chunk) / Chunk
    last_enc = [0.0]            # seconds the last finished chunk's encode took
    enc_limit = max(1, getattr(a, "encode_jobs", None) or 1)    # encodes at once (--encode-jobs)
    # (with faces, the face restorations of two chunks take turns: FACE_LOCK)

    def claim(keep=0):
        with lock:
            return waiting.pop(0) if len(waiting) > keep else None

    def give_back(entry):
        with lock:
            waiting.insert(0, entry)

    def make(entry, gpu=devices[0], lane=None):
        i, t, length, expected = entry
        stab = None
        if a.stabilize:
            # start `margin` frames early (read, stabilized and cut off again; see Chunk)
            first = i * a.chunk_frames          # this chunk's first frame in the whole video
            margin = min(a.stab_margin, first) // 4 * 4
            t, length = t - Fraction(margin) / fps, length + Fraction(margin) / fps
            stab = (first - margin, first + expected + a.stab_margin, margin)
        # (an AVI with B-frames: after the first chunk, which is read without -ss, seek by
        # ffmpeg's late picture times, else each later chunk started 1-2 frames early)
        seek = max(0.0, cut + float(t) + (info["seek_delay"] if t else 0.0) - eps)
        warm = i > 0 and (a.type == "vhs" and t * a.vhs_fin >= a.vhs_warm[0] + 1
                          or a.type != "vhs" and a.mode == "telecine" and not a.pal and t >= 1)
        with make_lock:             # (it sets this chunk's warm-up trim on a, then reads it)
            job = Chunk(a, i, f"{seek:.6f}", f"{float(length):.6f}", expected, work.resolve(),
                        f"chunk {i + 1}/{len(plan)}", last=(i == plan[-1][0]), warm=warm,
                        stab=stab)
        job.gpu, job.lane = gpu, lane
        job.make_room = None if lane else stop_helpers
        return job

    def counted(job, helped=False):
        with lock:
            progress["chunks"] += 1
            progress["video"] += job.length
            progress["helped"] += helped
            progress["frames"] += job.expected

    def helper(dev, lane, keep, gated):
        """Another upscaler, on another GPU or a second one on the same GPU (--gpu-jobs): whole
        chunks, from reading the frames to the chunk file, alongside the main one. The last
        `keep` chunks are left to the main one (another GPU may be slower: nothing waits for it
        at the end). gated: a second upscaler on a GPU starts once that GPU's speed with one is
        known (see meter). Whatever goes wrong, the chunk goes back to the main upscaler (its
        frames deleted first) and this one stops helping; a retired one stops after its chunk."""
        if gated:
            while not solo_ready[dev].wait(1):
                with lock:
                    nothing_left = len(waiting) <= keep
                if a.stop_lanes.is_set() or lane in a.retire_lanes or nothing_left:
                    return
        while not a.stop_lanes.is_set() and lane not in a.retire_lanes:
            entry = claim(keep=keep)
            if entry is None:
                return
            job = None
            try:
                job = make(entry, dev, lane)
                job.note = notes.append
                n_in, warning = job.extract()
                # (the end of the video: the main GPU deals with it)
                if n_in == 0 or lane_stopped(a, lane):
                    raise HandBack()
                job.upscale(n_in)
                if lane_stopped(a, lane):   # (Ctrl+C: no new encode that would hold it up)
                    raise HandBack()
                path, warning2 = job.finish()
            except BaseException as e:
                if job:
                    job.clear_frames()
                give_back(entry)
                if not isinstance(e, HandBack) and not lane_stopped(a, lane):
                    notes.append(f"  NOTE: {lane} stopped helping ({str(e).rstrip('.')}); "
                                 "the rest is done without it")
                return
            finally:
                LANE_STATUS.pop(lane, None)
            with lock:
                chunks[job.idx] = path
            counted(job, helped=True)
            for msg in (warning, warning2):     # (only now: a chunk handed back warns again)
                if msg:
                    notes.append(msg)

    def stop_helpers(why):
        """why "disk": the main upscaler's chunk doesn't fit on the drive next to the other
        upscalers' frames: they all stop rather than the movie failing. why "retry": the main
        upscaler failed, and the others on its GPU may have taken the memory it needed: they
        stop before it tries again. Their chunks come back to the main upscaler, their frames
        are deleted. True if any were still running."""
        busy = [(lane, h) for (dev, lane, *_), h in zip(lanes, helpers) if h.thread.is_alive()
                and (why == "disk" or dev == devices[0])]
        if not busy:
            return False
        if why == "disk":
            a.stop_lanes.set()
        for lane, _ in busy:
            a.retire_lanes.add(lane)
            a.cancel_lanes.add(lane)        # (stops its upscale now)
        for _, h in busy:
            try:
                h.wait()
            except Exception:
                pass
        names = " and ".join(lane for lane, _ in busy)
        notes.append("  NOTE: the drive with the work folder is too full for two chunks at "
                     f"once: {names} stopped helping, the main upscaler does the rest"
                     if why == "disk" else
                     f"  NOTE: {names} stopped, as it shares the GPU (which may be short of "
                     "memory for two upscalers); the main upscaler goes on alone")
        return True

    def take_back():
        """Nothing left for the main GPU while another GPU is still on a chunk: the main GPU
        takes it back if it would be done with it sooner, or if that GPU stopped making
        progress (a GPU or driver hang)."""
        now = time.time()
        quick = statistics.median(main_secs[-5:]) + 15 if main_secs else None
        for lane, (done, of, started, last_new) in list(LANE_PROGRESS.items()):
            if lane in a.cancel_lanes:
                continue
            stuck = now - last_new > 60 if done else now - started > 120
            left = (of - done) * (now - started) / done if done else None
            if stuck or quick and left is not None and left > 1.5 * quick:
                a.cancel_lanes.add(lane)
                notes.append(f"  NOTE: the main upscaler takes {lane}'s chunk back ("
                             + ("it stopped making progress)" if stuck else
                                "it will be done with it sooner)"))

    def done_encoding(keep=0):
        """Wait for the oldest encodes until at most `keep` are still running."""
        while len(encodings) > keep:
            bg, job = encodings[0]
            while a.faces and not bg.done.wait(0.5):      # --faces: show what it is doing
                if job.stage:
                    status_line(f"  {job.label}: {job.stage}")
            # (cleared only once it has ended: stopped meanwhile, it is waited for, see below)
            chunks[job.idx], warning = bg.wait()
            encodings.pop(0)
            last_enc[0] = getattr(job, "enc_secs", 0.0)
            for msg in (job.face_msg, warning):
                if msg:
                    status_line()
                    print(msg, flush=True)

    def show_notes():
        while notes:
            status_line()
            print(notes.pop(0), flush=True)

    # the upscalers besides the main one, each taking whole chunks: --gpu-jobs on every GPU (a
    # second one keeps the GPU busy while the other starts up, checks its frames or waits for
    # the next ones; each frame is upscaled the same way either way), and the other GPUs of
    # --gpu 0,1. Each: (GPU, name, chunks it leaves to the main one, gated: waits until its
    # GPU's speed with one upscaler is known)
    jobs = 1 if a.fast else a.gpu_jobs or 1

    def lane_name(dev, k):
        return (f"GPU {dev}" if dev is not None else "upscaler") + (f" #{k + 1}" if k else "")
    lanes = [(devices[0], lane_name(devices[0], k), 1, True) for k in range(1, jobs)]
    for d in ([] if a.fast else dict.fromkeys(devices[1:])):
        if d and d != devices[0]:
            lanes += [(d, lane_name(d, k), 3, k > 0) for k in range(jobs)]
    count = Counter([devices[0]] + [dev for dev, *_ in lanes])
    solo_ready = {dev: threading.Event() for dev, c in count.items() if c > 1}
    a.retire_lanes = set()                  # upscalers that stop after their current chunk
    meter_stop = threading.Event()

    def meter():
        """Frames a second on each GPU with one upscaler running and with two, sampled every
        second. A GPU's second upscaler waits until the speed with one is known (30 s of it),
        and is retired after its chunk if two turn out slower than one over the next 60 s (a
        GPU short of video memory, say)."""
        last, rate, decided = {}, {}, set()
        while not meter_stop.wait(1):
            now = dict(UPSCALER_FRAMES)
            new = {}                # GPU -> frames each of its upscalers made in that second
            for k, (dev, n) in now.items():
                # (one that just appeared made its first frames within that second; a lower
                # count than before is the same upscaler on its next chunk: skipped once)
                before = last[k][1] if k in last else 0
                if n >= before:
                    new.setdefault(dev, []).append(n - before)
            last = now
            for dev in solo_ready:
                r = rate.setdefault(dev, {1: [0, 0], 2: [0, 0]})     # frames, seconds
                if dev in new:
                    s = r[min(2, len(new[dev]))]
                    s[0], s[1] = s[0] + sum(new[dev]), s[1] + 1
                if r[1][1] >= 30:
                    solo_ready[dev].set()
                if dev in decided or r[1][1] < 30 or r[2][1] < 60:
                    continue
                decided.add(dev)
                one, two = r[1][0] / r[1][1], r[2][0] / r[2][1]
                if two < 0.95 * one:
                    a.retire_lanes.update(lane for d, lane, _, gated in lanes
                                          if d == dev and gated)
                    notes.append(f"  NOTE: two upscalers at once on "
                                 f"{'the GPU' if dev is None else 'GPU ' + dev} were slower "
                                 f"({two:.1f} frames/s against {one:.1f} for one): one at a "
                                 "time from the next chunk on")

    for line in hardware_text():
        say(line)
    load = LoadWatch()                      # CPU and GPU load: per chunk and for the whole run
    old_ctrl_c = signal.getsignal(signal.SIGINT)
    if lanes and old_ctrl_c is signal.default_int_handler \
            and threading.current_thread() is threading.main_thread():
        def ctrl_c(sig, frame):
            a.stop_lanes.set()      # the other GPUs stop now, before they start anything new
            a.stopping.set()
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT, ctrl_c)
    if count[devices[0]] > 1:
        say(f"Up to {count[devices[0]]} upscalers at once on the GPU (--gpu-jobs; same picture, "
            "the GPU waits less between chunks)")
    helpers = [Background(lambda lane=lane: helper(*lane)) for lane in lanes]
    if solo_ready:
        Background(meter)
    try:
        while True:
            if reading:
                bg, job = reading
                reading = None
                status_line(f"  {job.label}: reading frames")
                n_in, warning = bg.wait()
            else:
                entry = claim()
                if entry is None:
                    show_notes()
                    if any(h.thread.is_alive() for h in helpers):
                        # the other GPUs' last chunks (one they give back is done here)
                        take_back()
                        show_notes()
                        status_line("  waiting for: " + ", ".join(LANE_STATUS.values()))
                        time.sleep(1)
                        continue
                    if waiting:
                        continue
                    break
                job = make(entry)
                if a.fast:
                    status_line(f"  {job.label}: filtering and encoding")
                    n_in, warning = job.fast(), None
                else:
                    status_line(f"  {job.label}: reading frames")
                    n_in, warning = job.extract()
            a.chunk_label = job.label
            current = job
            if warning:
                status_line()
                print(warning, flush=True)
            if n_in == 0:
                status_line()
                job.no_video()                  # (stops here unless it is the last chunk)
                skipped.add(job.idx)
                print(f"  {job.label} is past the end of the video, skipping it")
                current = None
                continue
            if not a.fast:
                nxt = claim()
                if nxt:                         # the next chunk's frames, in the background
                    nxt_job = make(nxt)
                    reading = (Background(nxt_job.extract), nxt_job)
                t_up = time.time()
                job.upscale(n_in)
                main_secs.append(time.time() - t_up)
            done_encoding(enc_limit - 1)        # (the encodes that were running, up to --encode-jobs)
            encodings.append((Background(job.finish), job))
            current = None
            counted(job)
            show_notes()
            elapsed = time.time() - t_start
            per_chunk = elapsed / progress["chunks"]
            remaining = len(todo) - progress["chunks"]
            left = remaining * per_chunk
            movie = f" - movie {queue['n']} of {queue['of']} - this movie:" if queue else " -"
            helped = (f" ({progress['helped']} by the other upscaler{'s' * (len(helpers) > 1)})"
                      if helpers and progress["helped"] else "")
            shown[0] = len(plan) - remaining
            where = (f" (upscale {main_secs[-1]:.0f}s, encode {last_enc[0]:.0f}s)"
                     if main_secs and last_enc[0] else "")
            detail = [x for x in (job.up_info, job.gpu_info) if x]
            say(f"[{len(plan) - remaining}/{len(plan)}] {per_chunk:.0f}s/chunk{where}{helped}"
                + (f"{movie} {eta_text(left)}" if remaining else
                   ", all chunks done, finishing the file..."))
            if detail and not helpers:
                say("        upscale: " + "; ".join(detail))
            chunk_load, run_load = load.stats(since_last=True), load.stats()
            phone_load(chunk_load, run_load)
            for line in load.format(chunk_load):
                say("        " + line)
            if later > 0 and remaining:
                # the movies still to come, at this movie's speed (seconds of work per second
                # of video): rough, a live-action movie takes longer than an anime one
                rest = left + later * elapsed / progress["video"]
                say(f"        all {queue['of']} movies: {eta_text(rest)} (rough)")
        if encodings:
            status_line(f"  {encodings[-1][1].label}: encoding")
        done_encoding()
        status_line()
        show_notes()
        if helpers and progress["chunks"] and shown[0] != len(plan):
            # (another upscaler finished the last chunk to be counted)
            say(f"[{len(plan)}/{len(plan)}] ({progress['helped']} by the other "
                f"upscaler{'s' * (len(helpers) > 1)}), all chunks done, finishing the file...")
        if progress["frames"] and not a.fast:
            took = time.time() - t_start
            say(f"Upscaled {progress['frames']} frames in {short_time(took)} "
                f"({progress['frames'] / took:.1f} frames/s)")
        run_load = load.stats()
        phone_load(None, run_load)
        overall = load.format(run_load)
        if overall:
            say("Load over the whole run (average and peak):")
            for line in overall:
                say("  " + line)
    except BaseException:
        # stopped (Ctrl+C) or failed: let the steps already running in the background end
        # first, so nothing is left writing on its own (a finished encode keeps its chunk; a
        # face restoration is stopped, and its chunk made again next time)
        a.stop_lanes.set()
        a.stopping.set()
        for bg in [x[0] for x in encodings] + [x[0] for x in (reading,) if x] + helpers:
            try:
                bg.wait()
            except BaseException:
                pass
        for job in (current, reading and reading[1]):
            if job:
                job.clear_frames()              # made again next time
        raise
    finally:
        meter_stop.set()
        load.stop()
        if signal.getsignal(signal.SIGINT) is not old_ctrl_c:
            signal.signal(signal.SIGINT, old_ctrl_c)
    # every chunk must be there (a chunk lost between GPUs would otherwise just be missing
    # from the movie)
    missing = [i + 1 for i, *_ in plan if i not in chunks and i not in skipped]
    if missing:
        raise RuntimeError(f"chunk {missing[0]} of {len(plan)} wasn't made (an internal error), "
                           "so the movie wasn't finished")
    chunks = [chunks[k] for k in sorted(chunks)]
    if not chunks:
        sys.exit("No video was produced (is the input's video stream empty?)")
    return chunks


def finish_movie(a, info, work, chunks, total):
    """The chunks joined, with the source's audio, subtitles and chapters, into the output."""
    lst = work / "list.txt"
    lst.write_text("".join("file '{}'\n".format(c.resolve().as_posix().replace("'", "'\\''"))
                           for c in chunks), encoding="utf-8")
    joined = work / "video_joined.mkv"
    print("Finishing the file:", flush=True)
    run_step(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
              "-i", lst, "-c", "copy", joined], float(total), "joining the chunks")

    # -copyts everywhere below: without it ffmpeg re-bases every input on the first timestamp of
    # the streams taken from it, which put subtitles 0.3 s (VOB: seconds) early and moved VOB
    # audio that starts after the video. Instead everything keeps the source's own timeline
    # (minus its start), and the video is placed where the source's video starts.
    vo = info["vstart"] - info["cstart"]
    # Subtitles are added in a second pass. Copying a sparse subtitle track in the same pass as
    # the audio makes ffmpeg store the audio far from its video (measured: 15 s typical, up to
    # 50 s), and players such as VLC then lose the sound after a skip.
    # .vob/.mpg: subtitle (and some audio) streams can start minutes in; look further for them
    more = (["-probesize", "100M", "-analyzeduration", "100M"]
            if Path(a.input).suffix.lower() in MPEG_EXT else [])
    streams = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", *more, "-show_entries", "stream=codec_type,codec_name,channels",
         "-of", "json", str(a.input)], stdin=subprocess.DEVNULL,
        timeout=LONG_PROBE_TIMEOUT))["streams"]
    audio = [s for s in streams if s["codec_type"] == "audio"]
    subs = [s for s in streams if s["codec_type"] == "subtitle"]
    mp4 = Path(a.output).suffix.lower() in (".mp4", ".m4v")
    amap, aopts = [], []
    if audio and audio[0]["codec_name"] != "aac":
        # AAC copy of the main track first, as the default: the one audio format every player,
        # TV, phone, console and browser decodes (Windows 11 24H2 Media Player, PS5, Android and
        # Firefox can't play Dolby Digital or DTS). The original tracks follow for home theater.
        ch = audio[0].get("channels") or 2
        print(f"Main audio is {audio[0]['codec_name']}: adding an AAC copy as the default track "
              "so every player has sound")
        amap += ["-map", "1:a:0"]
        aopts += ["-c:a:0", "aac", "-b:a:0", "384k" if ch > 2 else "192k",
                  # standard 5.1, not the DVD's 5.1(side): else the encoder writes a custom
                  # channel map (PCE) that hardware decoders often can't read
                  *(["-filter:a:0", "aformat=channel_layouts=5.1"] if ch > 2 else []),
                  "-metadata:s:a:0", "title=AAC"]
    if audio:
        aopts += ["-disposition:a", "-default", "-disposition:a:0", "default"]
    for i, s in enumerate(audio):
        if mp4 and s["codec_name"] not in MP4_AUDIO:
            print(f"Note: audio track {i + 1} ({s['codec_name']}) can't go in .mp4, left out "
                  "(use .mkv to keep it)")
            continue
        if s["codec_name"] in ("pcm_dvd", "pcm_bluray"):     # .mkv can't hold these: lossless FLAC
            aopts += [f"-c:a:{len(amap) // 2}", "flac"]
        amap += ["-map", f"1:a:{i}"]
    if mp4:
        # .mp4 only takes text subtitles; DVD subtitles are pictures
        keep = [i for i, s in enumerate(subs) if s["codec_name"] in MP4_SUBS]
        if len(keep) < len(subs):
            print("Note: DVD subtitles can't go in .mp4 and were left out (use .mkv to keep them)")
        smap = [x for i in keep for x in ("-map", f"1:s:{i}")]
        # note: ffmpeg's mp4 muxer still enables the first subtitle track, so some players
        # (mpv, Jellyfin) show it by default; .mkv keeps the disc's own on/off setting
        sopts = ["-c:s", "mov_text", "-disposition:s", "0"]
        # faststart: playable while still downloading/streaming; hvc1: Apple devices need it
        final = ["-f", "mp4", "-movflags", "+faststart", *(["-tag:v", "hvc1"] if a.hevc else [])]
    else:
        keep = [i for i, s in enumerate(subs) if s["codec_name"] in MKV_SUBS]
        if len(keep) < len(subs):
            print("Note: subtitles in a format .mkv can't hold were left out ("
                  + ", ".join(sorted({s["codec_name"] for s in subs} - MKV_SUBS)) + ")")
        smap, final = [x for i in keep for x in ("-map", f"1:s:{i}")], []
        # .mkv can't hold mov_text (subtitles from an .mp4 source): convert those to SRT
        sopts = [x for j, i in enumerate(keep) if subs[i]["codec_name"] == "mov_text"
                 for x in (f"-c:s:{j}", "srt")]
    # a label inside every finished movie, so --all never mistakes it for a movie to upscale
    # (or, with --delete-originals, for an original to delete), even if its original is gone
    final += ["-metadata", f"comment={OUTPUT_TAG} from {source_id(a.input)}"]
    # the movie is written under a temporary name and renamed only when complete, so a
    # half-written file never looks finished (the queue relies on this)
    out = Path(a.output)
    part = out.with_name(out.stem + ".part" + out.suffix)
    av = work / "video_audio.mkv" if smap else part
    aoff = f"{-info['cstart']:.6f}"
    if a.type == "vhs" and a.mode == "telecine" and a.vhs_fin > 26:
        # VHS movie tapes: the IVTC puts the film frames on an even 23.976 grid that starts at
        # the tape's first frame, from 0.5 field before to 2 fields after where the tape showed
        # them, depending on where the 3:2 pattern stood at the start or after an edit
        # (measured on simulated tapes: video 9 ms early to 24 ms late). Audio 0.75 field later
        # centres that: within about +-21 ms instead of -8..+33 ms. (Moving the video earlier
        # instead breaks .mp4 output.)
        aoff = f"{0.75 * 1001 / 60000 - info['cstart']:.6f}"
    run_step(["ffmpeg", "-y", "-v", "error", "-copyts",
         "-itsoffset", f"{vo - video_start(joined):.6f}", "-i", joined,
         *more, "-itsoffset", aoff, "-i", a.input,
         "-map", "0:v", *amap, *([] if smap or mp4 else ["-map", "1:t?"]),
         "-map_chapters", "1", "-map_metadata", "1", "-c", "copy", *aopts,
         "-disposition:v:0", "default", *(["-t", f"{vo + float(total):.3f}"] if a.test else []),
         *([] if smap else final), av], vo + float(total),
         "adding the sound" + (" (making the AAC copy)" if aopts[:2] == ["-c:a:0", "aac"]
                               else ""))
    if smap:
        # the muxer may have moved everything later (it does when the audio starts before the
        # video): put the subtitles on that file's timeline, lined up with its video
        vs = video_start(av)
        # explicit -disposition: otherwise ffmpeg marks the first subtitle track "default" when
        # there are two or more, and players then show subtitles nobody asked for
        run_step(["ffmpeg", "-y", "-v", "error", "-copyts", "-i", av,
             *more, "-itsoffset", f"{vs - info['vstart']:.6f}", "-i", a.input,
             "-map", "0", *smap, *([] if mp4 else ["-map", "1:t?"]),     # + attached fonts
             "-c", "copy", *sopts, "-disposition:v:0", "default",
             *(["-t", f"{vs + float(total):.3f}"] if a.test else []), *final, part],
             vs + float(total), "adding the subtitles")
    try:
        flush_to_disk(part)
        replace_file(part, out)
    except PermissionError:
        sys.exit(f"Can't replace '{out}': it's open in another program (a video player?). "
                 "Close it and run the same command again; only this last step is redone.")
    if smap:
        try:
            av.unlink()
        except OSError:
            pass        # only a temp file in the work folder (Windows may hold it briefly)

    vid = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(joined)], encoding="utf-8", errors="replace",
        stdin=subprocess.DEVNULL, timeout=LONG_PROBE_TIMEOUT).strip())
    diff = vid - float(total)
    print(f"Done: {a.output}\nVideo length {vid:.2f}s vs source {float(total):.2f}s "
          f"(difference {diff:+.2f}s)")
    if abs(diff) > 1.0:
        print("WARNING: length mismatch, check lip sync near the end.")
    try:
        (work / "video_joined.mkv").unlink()      # a second full copy of the video: not needed now
    except OSError:
        pass
    if all(WORK_FILES.fullmatch(f.name) for f in work.iterdir()) and \
            not os.environ.get("DVD_UPSCALE_CLIP_TEST"):            # (that one cleans up itself)
        print("You can delete", work)


def main():
    a = build_parser().parse_args()
    find = start_run(a)
    user, vhs_info = pick_preset(a, find)
    check_output(a)
    pick_model(a, find, user)
    pick_engine(a)
    pick_faces(a)
    info = read_source(a, vhs_info)
    measured = detect_frames(a, info)
    if a.analyze:
        report = os.environ.get("DVD_UPSCALE_REPORT")
        if report:
            Path(report).write_text(json.dumps(dict(type=a.type, mode=a.mode,
                                                    combed=bool(getattr(a, "combed", False)),
                                                    denoise=a.denoise, filters=training_filters(a))))
        return
    check_ai(a)
    fps = pick_fps(a, info, measured)
    setup_stabilize(a, info, fps)
    pick_encoder(a, fps)
    work, st = open_work(a, info)
    plan, total, vo, eps, vlen = plan_chunks(a, info, fps)
    cut = chunk_source(a, info, work, st, plan, total, vo, eps, vlen)
    chunks = upscale_chunks(a, info, fps, work, plan, cut, eps)
    finish_movie(a, info, work, chunks, total)


# ---------------------------------------------------------------------------------------------
# Queue mode: python dvd_upscale.py --queue [queue.txt] [--shutdown]
# Each movie runs as its own run of this script, so one movie crashing can't take the queue
# down, and Ctrl+C, the work-folder lock and resuming behave exactly as for a single movie.

VALUE_OPTS = ("--type", "--mode", "--model", "--scale", "--height", "--dar", "--fps",
              "--chunk-frames", "--test", "--esrgan", "--gpu", "--tile", "--gpu-threads",
              "--gpu-jobs", "--encode-jobs", "--engine",
              "--ai-blend", "--busy-blend", "--smooth",
              "--sharpen", "--work", "--chroma-delay", "--mask", "--face-model", "--face-models")
STOPPED = (130, 3221225786)     # a run stopped by Ctrl+C; Windows "terminated by Ctrl+C"
VIDEO_EXT = (".mkv", ".mp4", ".m4v")
# one word: a 'single-quoted' word ('' = one apostrophe, PowerShell style), or text in which
# "double quotes" group spaces anywhere (Windows names can't contain "), or an unclosed quote
_WORD = re.compile(r"""'((?:[^']|'')*)'(?!\S)|(?:"[^"]*"|[^\s"])+|"[^"]*$""")


def queue_parse(line):
    words = []
    for m in _WORD.finditer(line):
        if m.group(1) is not None:
            words.append(m.group(1).replace("''", "'"))
        elif m.group().startswith("#"):
            break                                   # the rest of the line is a comment
        elif m.group().count('"') % 2:
            raise ValueError("a quote is not closed")
        else:
            words.append(m.group().replace('"', ""))
    return words


def queue_lines(path):
    """queue.txt as Notepad (UTF-8, UTF-8 BOM, UTF-16, ANSI) or PowerShell 5.1 (> and >> write
    UTF-16, Add-Content writes ANSI) may have saved it, even mixed."""
    data = path.read_bytes()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = data.decode("utf-16", errors="replace")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("mbcs" if os.name == "nt" else "latin-1", errors="replace")
    return text.replace("\x00", "").replace("\ufeff", "").splitlines()


def queue_test_secs(args):
    """The line's --test N (seconds), or 0."""
    for i, w in enumerate(args):
        v = w.split("=", 1)[1] if w.startswith("--test=") else (
            args[i + 1] if w == "--test" and i + 1 < len(args) else None)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                return 0
    return 0


def queue_read(path):
    base, jobs, outs = path.parent, [], {}
    claimed = set()                     # work folders a line already continues

    def started_in(folder, args, test):
        """A run of this line's movie (same --test) was started in that work folder."""
        try:
            old = json.loads((base / (folder + ("_test" if test else "")) / "settings.json")
                             .read_text())
        except (OSError, ValueError):
            return False
        return old.get("input") == str((base / args[0]).resolve()) and old.get("test", 0) == test

    for n, line in enumerate(queue_lines(path), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            args = queue_parse(line)
        except ValueError as e:
            jobs.append((n, line, None, f"can't read this line ({e})", True))
            continue
        while args and Path(args[0]).name.lower() in (
                "python", "python.exe", "python3", "py", "py.exe", "dvd_upscale.py"):
            args.pop(0)          # a whole "python dvd_upscale.py ..." command pasted in
        if args and not args[0].startswith("-") and (len(args) == 1 or args[1].startswith("-")):
            # no output name: the default one, under the queue file's folder
            h = "1080"
            for k, w in enumerate(args):        # (the last one counts, as for argparse)
                if w.startswith("--height="):
                    h = w.split("=", 1)[1]
                elif w == "--height" and k + 1 < len(args):
                    h = args[k + 1]
            try:
                out = default_output(base / args[0], base, int(h))
                if queue_test_secs(args):           # a preview never takes the movie's name
                    out = out.with_name(out.stem + " test.mkv")
                try:
                    out = out.relative_to(base.resolve())
                except ValueError:                  # a movie outside the queue's folder
                    pass
                args.insert(1, str(out))
            except (ValueError, OSError):
                pass
        if (len(args) < 2 or args[0].startswith("-")
                or Path(args[1]).suffix.lower() not in VIDEO_EXT):
            jobs.append((n, line, None, "a line must start with the input file (then, if you "
                                        "like, the output file: .mkv or .mp4)", True))
            continue
        key = os.path.normcase(str((base / args[1]).resolve()))
        if "--analyze" in args:
            key = None                      # writes nothing
        elif key in outs:
            jobs.append((n, line, None, f"same output file as line {outs[key]}: give it another "
                                        "name", True))
            continue
        if key:
            outs[key] = n
        if not any(w == "--work" or w.startswith("--work=") for w in args):
            o, src = Path(args[1]), Path(args[0])
            work = str(o.with_name(o.stem + "_work"))      # next to the output: one per output
            # a movie started elsewhere continues there instead of starting over: by hand or
            # with --all ("<movie name>_work" next to the movie), by hand with an older version
            # (upscale_work), or by an older queue ("<output name>_work" next to the queue
            # file). Each folder goes to one line only (two outputs of one movie would clash)
            test = queue_test_secs(args)
            work = next((w for w in (work, str(src.with_name(src.stem + "_work")), "upscale_work",
                                     o.stem + "_work")
                         if w not in claimed and started_in(w, args, test)), work)
            if "--analyze" not in args:
                claimed.add(work)
            args += ["--work", work]
        jobs.append((n, line, args, None, True))
    return jobs


# the quick look at a file's header (length, height, tags): a few hundred ms on a good file. A
# damaged file, one still being copied, or a network drive that stopped answering can make
# ffprobe hang for good, and with it a whole --all/--queue night: given up on after this long
PROBE_TIMEOUT = 60
# the same for a header read inside a movie's run (more patient: a .vob read 100 MB in, a
# drive waking up), and for reading a whole file through (counting its frames)
LONG_PROBE_TIMEOUT = 300
READ_TIMEOUT = 3 * 3600


def ffprobe_out(args, timeout=None):
    """ffprobe's output (text) for a quick header read; "" if it failed, hung, or isn't there.
    timeout: seconds (default PROBE_TIMEOUT)."""
    try:
        return subprocess.run(["ffprobe", *map(str, args)], capture_output=True,
                              stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
                              timeout=PROBE_TIMEOUT if timeout is None else timeout).stdout or ""
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


# one ffprobe per file for the header facts the queue asks about (length, video height, the
# label of a finished movie), kept until the file changes: --all looks at every file in the
# folder again before each movie, which was three ffprobes per file each time (slow on a NAS)
_HEADERS = {}


def file_header(path):
    """{"duration": str or None, "height": str or None, "tags": dict or None} for a file (None
    values: not there or unreadable), from one ffprobe, kept until the file's size or date
    changes."""
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    except OSError:
        key = None
    if key is not None and key in _HEADERS:
        head, failed_at = _HEADERS[key]
        # (a failed read is tried again after 10 minutes: a file that was locked, a drive that
        # was waking up; until then a file that hangs ffprobe costs one wait, not one per look)
        if failed_at is None or time.time() - failed_at < 600:
            return head
    out = ffprobe_out(["-v", "error", "-show_entries", "format=duration:format_tags:stream=height",
                       "-select_streams", "v:0", "-of", "json", path])
    try:
        j = json.loads(out)
        fmt = j.get("format")
        head = {"duration": (fmt or {}).get("duration"),
                "height": next((s.get("height") for s in j.get("streams") or []), None),
                "tags": None if fmt is None else (fmt.get("tags") or {})}
    except (ValueError, AttributeError):
        head = {"duration": None, "height": None, "tags": None}
    if key is not None:
        if len(_HEADERS) > 2000:
            _HEADERS.clear()
        _HEADERS[key] = (head, None if out else time.time())
    return head


def media_duration(path):
    try:
        return float(str(file_header(path)["duration"]).strip())
    except ValueError:
        return None


def queue_finished(base, args):
    """Output exists and has the expected length, so an older, shorter --test preview saved
    under the same name isn't mistaken for the finished movie."""
    out = base / args[1]
    if not out.exists():
        return False
    try:
        if made_from(out) not in (None, "", source_id(base / args[0])):
            return False        # made from another file: not this movie's (main() won't touch it)
    except OSError:
        pass                    # the original is gone: its output is what's left
    d_out, d_src = media_duration(out), media_duration(base / args[0])
    if d_out is None:
        return False
    want, test = (d_src or 0), queue_test_secs(args)
    if test:
        want = min(want, test) if want else test
    return d_out >= 0.95 * want


MOVIE_EXT = (".mkv", ".mp4", ".m4v", ".mpg", ".mpeg", ".ts", ".m2ts", ".mts", ".avi", ".mov",
             ".dv")


def source_id(path):
    """'Movie.mkv (4549558560 bytes)': recorded in every finished movie, so it is never taken
    for, or replaced by, the upscale of another file that gets the same output name."""
    p = Path(path)
    return f"{p.name} ({p.stat().st_size} bytes)"


def made_from(path):
    """The source_id a finished movie records; "" for one made by an older version (nothing
    recorded), None if it isn't one of ours (or can't be read)."""
    tags = file_header(path)["tags"]
    if tags is None:
        return None
    for v in map(str, tags.values()):
        if OUTPUT_TAG in v:
            return v.split(" from ", 1)[1] if " from " in v else ""
    return None


def is_upscaled_output(path):
    """True for a movie this script made: it carries OUTPUT_TAG, or its name ends in " 720p",
    " 1080p", " 1440p"... (DVD rips are 480p/576p, so such a name is never an original)."""
    m = re.search(r" (\d{3,4})p$", Path(path).stem)
    if m and int(m.group(1)) >= 720:
        return True
    tags = file_header(path)["tags"] or {}
    return any(OUTPUT_TAG in str(v) for v in tags.values())


TYPE_DIRS = {"live": "live", "live action": "live", "live-action": "live", "anime": "anime",
             "cgi": "cgi", "3d": "cgi", "vhs": "vhs"}


def type_from_name(path):
    """A type tag at the start of the file name, for movies not in a type folder: cgi_Shrek.mkv,
    anime-DBZ.mkv, live_Spider-Man_dvd.mkv, vhs_Wedding.mpg (anime, live, cgi, 3d or vhs, then
    _ or -). None if it has none."""
    m = re.match(r"(anime|live|cgi|3d|vhs)[_-]", Path(path).name, re.I)
    return TYPE_DIRS[m.group(1).lower()] if m else None


def folder_jobs(folder, extra):
    """Every movie in folder and its "anime" / "live" / "cgi" / "vhs" subfolders as queue jobs, plus files
    that still seem to be copying (changed in the last 2 minutes)."""
    passed = build_parser().parse_known_args(["in.mkv", "out.mkv", *extra])[0]
    height = passed.height
    dirs = [(folder, None)] + [(d, TYPE_DIRS[d.name.lower()]) for d in sorted(folder.iterdir())
                               if d.is_dir() and d.name.lower() in TYPE_DIRS]
    jobs, waiting = [], []
    for d, kind in dirs:
        files = sorted(d.iterdir(), key=lambda x: x.name.lower())
        for f in files:
            if f.is_file() and f.suffix.lower() in (".vob", ".m2v"):
                jobs.append((len(jobs) + 1, str(f.relative_to(folder)), None,
                             "not picked up by --all (pieces of a DVD: rip the disc with MakeMKV "
                             "instead; or run this file on its own)", False))
        videos = [f for f in files if f.is_file() and f.suffix.lower() in MOVIE_EXT]
        stems = {f.stem for f in videos}
        for f in videos:
            stem = f.stem
            made = re.match(r"(.*) \d{3,4}p$", stem)
            if ((made and made.group(1) in stems)        # "Movie 1080p" made from "Movie"
                    or stem.endswith(".part") or stem.lower() == "test"
                    or re.search(r" clip \d+(h\d\d)?m\d\ds$", stem)      # made by --clip
                    or is_upscaled_output(f)):           # ours, even if the original is gone
                continue                 # one of our own outputs, half-written, or a preview
            rel = f.relative_to(folder)
            if sum(g.stem.lower() == stem.lower() for g in videos) > 1:
                jobs.append((len(jobs) + 1, str(rel), None, "another movie here has the same name "
                             "(e.g. .mkv and .mp4); rename one so their outputs don't collide",
                             True))
                continue
            # (abs: a file dated in the future - a wrong clock - would wait until then)
            if abs(time.time() - f.stat().st_mtime) < 120:
                waiting.append(str(rel))
                continue
            own = kind or type_from_name(f)          # (the folder's type, else the name's tag)
            tape = (own or passed.type) == "vhs"
            h = video_height(f)
            if h and h > 576 and not tape:
                jobs.append((len(jobs) + 1, str(rel), None,
                             f"already HD ({h} lines), not a DVD", False))
                continue
            d_src = media_duration(f)
            # (not for VHS: home videos are often captured as short clips)
            if d_src is not None and d_src < 300 and not tape:
                jobs.append((len(jobs) + 1, str(rel), None,
                             "shorter than 5 minutes (a preview or extra?)", False))
                continue
            out = Path(f"{height}p Upscale") / rel.parent / f"{stem} {height}p.mkv"
            args = [str(rel), str(out), *extra]
            # the folder's type wins over a --type given for the loose movies (the last --type
            # given counts)
            if own:
                args += ["--type", own]
            args += ["--work", str(rel.with_name(stem + "_work"))]
            jobs.append((len(jobs) + 1, str(rel), args, None, False,
                         str(rel.with_name(f"{stem} {height}p.mkv"))))
    return jobs, waiting


def video_height(path):
    m = re.match(r"\s*(\d+)", str(file_header(path)["height"] or ""))
    return int(m.group(1)) if m else None


def video_length(path):
    """Length of the video stream itself (not the whole file, which counts the audio too)."""
    out = ffprobe_out(["-v", "error", "-select_streams", "v:0", "-show_entries",
                       "stream=duration:stream_tags", "-of", "json", path])
    try:
        st = json.loads(out)["streams"][0]
    except (ValueError, KeyError, IndexError):
        return None
    d = hms(st.get("duration"))
    if d is None:
        d = hms(next((v for k, v in st.get("tags", {}).items()
                      if k.upper().startswith("DURATION")), None))
    return d


def check_finished_movie(src, out):
    """Before an original is deleted: is the new file complete? Returns a reason if not."""
    def tracks(path):
        more = (["-probesize", "100M", "-analyzeduration", "100M"]
                if Path(path).suffix.lower() in MPEG_EXT else [])
        j = json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", *more, "-show_entries", "stream=codec_type,codec_name",
             "-of", "json", str(path)], stdin=subprocess.DEVNULL, timeout=LONG_PROBE_TIMEOUT))
        return [s.get("codec_type") for s in j["streams"]], j["streams"]
    try:
        (t_src, s_src), (t_out, _) = tracks(src), tracks(out)
    except (subprocess.SubprocessError, ValueError, KeyError):
        return "it couldn't be read"
    d_src, d_out = media_duration(src), media_duration(out)
    if d_src is None or d_out is None or d_out < 0.98 * d_src:
        return "it is shorter than the original"
    # the source's real length, measured by reading it through (some DVD files misreport their
    # length), against the frames really in the new file (catches missing/damaged chunks)
    rc, txt = capture(["ffmpeg", "-hide_banner", "-nostats", "-i", src, "-map", "0:v:0",
                       "-c", "copy", "-f", "null", "-"])
    m = re.findall(r"time=(\d+):(\d+):([\d.]+)", txt)
    want = int(m[-1][0]) * 3600 + int(m[-1][1]) * 60 + float(m[-1][2]) if m else 0
    v_src = video_length(src)
    want = max(want, v_src or 0)
    try:
        fps = float(probe(out)["fps"])
        have = count_frames(out) / fps if fps > 0 else 0
    except (subprocess.SubprocessError, ValueError, KeyError, ZeroDivisionError):
        have = 0
    if not want or have < 0.98 * want:
        return "its video is shorter than the original's"
    # the tracks the new file can hold (the rules it was made with: an .mp4 leaves out DVD
    # subtitles, fonts and audio .mp4 players don't take), plus the AAC copy of the main track
    mp4 = Path(out).suffix.lower() in (".mp4", ".m4v")
    a_src = [s.get("codec_name") for s in s_src if s.get("codec_type") == "audio"]
    want_audio = (sum(1 for c in a_src if not mp4 or c in MP4_AUDIO)
                  + (1 if a_src and a_src[0] != "aac" else 0))
    want_subs = sum(1 for s in s_src if s.get("codec_type") == "subtitle"
                    and s.get("codec_name") in (MP4_SUBS if mp4 else MKV_SUBS))
    if t_out.count("video") < 1 or t_out.count("audio") < want_audio:
        return "it doesn't have all of the original's audio tracks"
    if t_out.count("subtitle") < want_subs:
        return "it doesn't have all of the original's subtitles"
    if not mp4 and t_out.count("attachment") < t_src.count("attachment"):
        return "it doesn't have the original's attached fonts"
    # the picture itself, at points through the movie: about as bright as the original's
    # (a black or garbled stretch from a GPU fault fails this)
    def brightness(path, t):
        rc, txt = capture(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", path, "-t", "1",
                           "-map", "0:v:0", "-vf", "format=yuv420p,signalstats,"
                           "metadata=mode=print:key=lavfi.signalstats.YAVG:file=-",
                           "-f", "null", "-"])
        ys = [float(x) for x in re.findall(r"YAVG=([\d.]+)", txt)]
        return sum(ys) / len(ys) if ys else None
    off = 0
    for k in range(1, 10):
        x, y = brightness(src, want * k / 10), brightness(out, want * k / 10)
        off += x is not None and y is not None and abs(x - y) > 25
    if off >= 2:
        return "its picture doesn't match the original's in places (black or garbled?)"
    # play the whole new file through once (a few minutes; tiny next to the upscale)
    rc, txt, _ = ffmpeg_progress(["ffmpeg", "-v", "error", "-i", out, "-map", "0:v:0",
                                  "-map", "0:a?", "-f", "null", "-"], d_out,
                                 "checking the new file plays all the way through")
    if rc or [l for l in txt.splitlines() if l.strip() and not l.startswith("[null @")]:
        return "it doesn't play cleanly all the way through"
    return None


WORK_FILES = re.compile(r"chunk_\d{5}(\.part)?\.mkv|tmp_\d{5}|list\.txt|"
                        r"video_(joined|audio)\.mkv|video_index(\.part)?\.mkv|"
                        r"(settings|detected)\.json(\.tmp)?|stab(\.part)?\.trf|\.lock")


def clean_work_folder(work):
    """Delete only the files this script puts in a work folder, then the folder if that leaves
    it empty: a --work pointing at a folder with other things in it never loses them."""
    for f in work.iterdir():
        if WORK_FILES.fullmatch(f.name):
            try:
                if f.is_dir() and not f.is_symlink():
                    shutil.rmtree(f, ignore_errors=True)
                else:
                    f.unlink()
            except OSError:
                pass
    try:
        work.rmdir()
    except OSError:
        pass                    # something else is in it: leave it


def to_recycle_bin(path):
    """Windows: move a file to the Recycle Bin (Windows deletes it for good if it's too big for
    the bin). Elsewhere: delete it. Returns True if the file is gone."""
    if os.name != "nt":
        path.unlink()
        return not path.exists()
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
                    ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
                    ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]
    FO_DELETE, FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 3, 4, 16, 64, 1024
    op = SHFILEOPSTRUCTW(None, FO_DELETE, os.path.abspath(path) + "\0", None,  # list ends \0\0
                         FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI,
                         False, None, None)
    rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    return rc == 0 and not op.fAnyOperationsAborted and not path.exists()


def detected_type(base, args):
    """What a finished movie was upscaled as (for the log), from its work folder."""
    try:
        o = build_parser().parse_known_args(args)[0]
        d = json.loads((base / (o.work + ("_test" if o.test else "")) / "detected.json").read_text())
        return d["type"]
    except (OSError, ValueError, KeyError, TypeError, SystemExit):
        return None


def queue_main(argv):
    folder_mode = any(w == "--all" or w.startswith("--all=") for w in argv)
    p = argparse.ArgumentParser(
        prog="dvd_upscale.py --all" if folder_mode else "dvd_upscale.py --queue",
        description="Upscale every movie in a folder" if folder_mode else
                    "Upscale the movies listed in a queue file, one after another.")
    if folder_mode:
        p.add_argument("--all", nargs="?", const=".", metavar="FOLDER",
                       help="folder with the movies (default: the current folder)")
    else:
        p.add_argument("--queue", nargs="?", const="queue.txt", default="queue.txt",
                       help="queue file (default queue.txt)")
    p.add_argument("--shutdown", action="store_true",
                   help="turn the PC off when everything is finished")
    p.add_argument("--delete-originals", action="store_true",
                   help="after each movie is finished and checked, move the original to the "
                        "Recycle Bin and delete its work folder (never for --test runs)")
    if folder_mode:
        a, extra = p.parse_known_args(argv)
        # folder-valued options are relative to the folder typed in, but each movie runs in
        # its own folder
        # (a value left out, the next option there instead, is left for the check below to
        # report, and an empty one to mean the default folder)
        for i, w in enumerate(extra):
            if w == "--face-models" and i + 1 < len(extra) and extra[i + 1] \
                    and not extra[i + 1].startswith("-"):
                extra[i + 1] = str(Path(extra[i + 1]).resolve())
            elif w.startswith("--face-models=") and w.split("=", 1)[1]:
                extra[i] = "--face-models=" + str(Path(w.split("=", 1)[1]).resolve())
        # the folder given anywhere ("--all --shutdown D:\Movies"), not only right after --all,
        # but never the value of an option such as --face-models
        dirs = [w for k, w in enumerate(extra) if not w.startswith("-") and Path(w).is_dir()
                and not (k > 0 and extra[k - 1] in VALUE_OPTS)]
        if a.all == "." and len(dirs) == 1:
            a.all = dirs[0]
            extra.remove(dirs[0])
        if any(w == "--work" or w.startswith("--work=") for w in extra):
            sys.exit("--work can't be used with --all: every movie gets its own work folder.")
        check_values(build_parser().parse_args(["in.mkv", "out.mkv", *extra]))   # the options
        base = Path(a.all).resolve()
        if not base.is_dir():
            sys.exit(f"Folder not found: {base}")
        get_jobs, what = (lambda: folder_jobs(base, extra)), f"Folder: {base}"
    else:
        a = p.parse_args(argv)
        qfile = Path(a.queue or "queue.txt").resolve()
        if qfile.is_dir():
            qfile = qfile / "queue.txt"
        if not qfile.exists():
            sys.exit(f"Queue file not found: {qfile}\nCreate it with one movie per line, e.g.\n"
                     '  "Movie.mkv" "Movie 1080p.mkv" --type live --work movie\n'
                     "or use --all to do every movie in a folder.")
        base = qfile.parent
        get_jobs, what = (lambda: (queue_read(qfile), [])), f"Queue: {qfile}"
    script, log = Path(__file__).resolve(), base / "queue_log.txt"
    qlock = hold_lock(base / ".queue.lock",      # noqa: F841 (held until the queue exits)
                      "A queue is already running for this folder in another window. Stop that "
                      "one first (new movies / lines are picked up by the running one).")
    protect_run()

    # a marker in front of each kind of line on the screen (the log file keeps the plain text,
    # so it can be searched for DONE / FAILED)
    MARKERS = (("Started", "▶️ "), ("START", "▶️ "), ("DONE", "✅ "), ("Finished", "🏁 "), ("FAILED", "❌ "),
               ("SKIPPED", "❌ "), ("STOPPED", "⏹️ "), ("KEPT", "⚠️ "), ("Waiting", "⏳ "))

    def note(msg):
        marker = next((m for word, m in MARKERS if msg.startswith(word)), "")
        if not marker and "WARNING" in msg.upper():
            marker = "⚠️ "
        status_line()           # (the progress line, finished, before a line that stays)
        print(marker + msg, flush=True)
        if msg.startswith(("START", "Started")):     # (shown if Windows is asked to shut down)
            shutdown_reason(f"dvd_upscale.py is working through its list ({msg[:150]}): "
                            "shutting down now loses the chunk in progress")
        try:
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")
        except OSError as e:    # (a log file another program has open must not stop the queue)
            print(f"Couldn't write to {log.name}: {e}", flush=True)

    tried, done, failed, told_waiting, written = set(), [], [], set(), set()
    analyzed = 0

    def delete_original(n, args):
        """--delete-originals, after a movie is done: the original to the Recycle Bin and the
        work folder deleted, only when the new file passes every check."""
        src, out = base / args[0], base / args[1]
        here = os.path.normcase(str(src.resolve()))
        still = [j for j in get_jobs()[0] if j[2] and j[0] != n and "--analyze" not in j[2]
                 and not queue_finished(base, j[2])
                 and os.path.normcase(str((base / j[2][0]).resolve())) == here]
        if still:
            note(f"KEPT the original {args[0]} for now: line {still[0][0]} of the queue still "
                 "needs it")
            return
        try:
            problem = check_finished_movie(src, out)
        except KeyboardInterrupt:
            note(f"STOPPED during the check of {args[1]}: the original {args[0]} was kept (run "
                 "the same command again to check and delete it)")
            sys.exit(130)
        if not problem and (src.resolve() == out.resolve() or is_upscaled_output(src)):
            problem = "the 'original' is itself an upscaled movie"
        if problem:
            note(f"KEPT the original {args[0]}: the new file looks incomplete ({problem})")
            return
        work = base / build_parser().parse_known_args(args)[0].work
        try:
            gone = to_recycle_bin(src)
        except OSError:
            gone = False
        if gone:
            if work.is_dir():
                clean_work_folder(work)
            note(f"Original deleted ({'to the Recycle Bin, if its drive has one' if os.name == 'nt' else 'permanently: there is no Recycle Bin here'}): {args[0]}"
                 + ("" if work.exists() else f" (work folder {work.name} deleted)"))
        else:
            note(f"KEPT the original {args[0]}: it couldn't be moved to the Recycle Bin")
    note(f"Started. {what}")
    while True:
        job = None
        jobs, waiting = get_jobs()                       # re-read: movies/lines may be added
        for n, key, args, err, is_failure, *legacy in jobs:
            if key in tried:
                continue
            tried.add(key)
            if err:
                if is_failure:
                    failed.append(key)
                note(f"SKIPPED {key}: {err}")
                continue
            if os.path.normcase(str((base / args[1]).resolve())) in written:
                failed.append(key)
                note(f"SKIPPED {key}: another movie was already saved as {args[1]} in this run")
                continue
            if queue_finished(base, args):
                note(f"Already done, skipping: {args[1]}")
                # finished in an earlier run without --delete-originals: its original goes now,
                # but only when the movie records being made from this very file
                src = base / args[0]
                if a.delete_originals and not queue_test_secs(args) and src.is_file() and \
                        made_from(base / args[1]) == source_id(src):
                    delete_original(n, args)
                continue
            if legacy and queue_finished(base, [args[0], legacy[0], *args[2:]]):
                note(f"Already done (next to the original), skipping: {legacy[0]}")
                continue
            job = (n, args)
            break
        if job is None:
            if waiting:                                  # a movie is still being copied in
                for w in set(waiting) - told_waiting:
                    note(f"Waiting for {w} to finish copying...")
                told_waiting |= set(waiting)
                time.sleep(30)
                continue
            break
        n, args = job
        # which movie this is, and how much video is left after it (for the overall ETA)
        valid = [j for j in jobs if not j[3]]
        later, later_secs = 0, 0.0
        for _, key2, args2, _, _, *leg2 in valid:
            if (key2 in tried or queue_finished(base, args2)
                    or (leg2 and queue_finished(base, [args2[0], leg2[0], *args2[2:]]))):
                continue
            later += 1
            d = media_duration(base / args2[0]) or 0.0
            test = queue_test_secs(args2)
            later_secs += min(d, test) if test else d
        pos = len(valid) - later
        print(f"\n=== Movie {pos} of {len(valid)}: {args[0]} ===", flush=True)
        analyze = "--analyze" in args
        note((f"ANALYZE {args[0]}" if analyze else f"START {args[1]}")
             + ("" if folder_mode else f"  (line {n})")
             + f"  (movie {pos} of {len(valid)})")
        if folder_mode and not analyze:
            (base / args[1]).parent.mkdir(parents=True, exist_ok=True)   # "1080p Upscale"
        t0 = time.time()
        env = dict(os.environ, DVD_UPSCALE_QUEUE=json.dumps(
            dict(n=pos, of=len(valid), later_secs=later_secs, folder=folder_mode)))
        p = subprocess.Popen([sys.executable, str(script), *args], cwd=base, env=env)
        try:
            rc = p.wait()
        except KeyboardInterrupt:
            # the movie's run got the Ctrl+C too: let it stop as a single movie does (its
            # background steps end, its GBs of frames are deleted); a second Ctrl+C ends it now
            # (subprocess.run gave it 0.25 s, then killed it mid-cleanup)
            try:
                p.wait(timeout=300)
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                p.kill()
                p.wait()
            rc = STOPPED[0]
        mins = (time.time() - t0) / 60
        if rc in STOPPED:
            note(f"STOPPED during {args[1]} after {mins:.0f} min. "
                 "Run the same command again to continue from its last finished chunk.")
            sys.exit(130)
        if rc == 0 and analyze:
            analyzed += 1
            continue
        if rc == 0 and (base / args[1]).exists():
            done.append(args[1])
            written.add(os.path.normcase(str((base / args[1]).resolve())))
            kind = detected_type(base, args)
            note(f"DONE  {args[1]}  ({mins:.0f} min{', as ' + kind if kind else ''})")
            if a.delete_originals and not queue_test_secs(args):
                delete_original(n, args)
        else:
            failed.append(" ".join(args[:2]))
            note(f"FAILED {args[1]} (exit code {rc}, {mins:.0f} min; see the error above in this "
                 "window), moving on. Run the same command again to retry it.")

    note(f"Finished: {len(done)} done, {len(failed)} failed"
         + (f", {analyzed} analyzed (nothing upscaled)" if analyzed else ""))
    for line in failed:
        print(f"  failed: {line}")
    if a.shutdown:
        note("Shutting down in 60 s. To cancel, type:  "
             + ("shutdown /a" if sys.platform == "win32" else "shutdown -c"))
        cmd = (["shutdown", "/s", "/t", "60"] if sys.platform == "win32"
               else ["shutdown", "-h", "+1"])
        allow_shutdown()                    # (this run's own shutdown block goes first)
        subprocess.run(cmd)


def hms_text(secs):
    m, s = divmod(int(secs), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}"


def has_dvd_pcm(path):
    out = ffprobe_out(["-v", "error", "-select_streams", "a", "-show_entries",
                       "stream=codec_name", "-of", "csv=p=0", path])
    return any(c.strip() in ("pcm_dvd", "pcm_bluray") for c in out.splitlines())


def clip_main(argv):
    """--clip: a short piece of a video, copied without re-encoding (e.g. to send a sample)."""
    p = argparse.ArgumentParser(
        prog="dvd_upscale.py --clip",
        description="Cut a short clip out of a video without re-encoding it (so it starts at the "
                    "nearest keyframe, up to about a second early). Saved next to the video as "
                    "'<name> clip <start>.mkv'.")
    p.add_argument("input")
    p.add_argument("start", nargs="?", default="0",
                   help="where to start: seconds, minutes:seconds or hours:minutes:seconds "
                        "(default 0)")
    p.add_argument("seconds", nargs="?", default="30", help="how many seconds (default 30)")
    p.add_argument("--video-only", action="store_true",
                   help="picture only, no sound (smallest file)")
    p.add_argument("--upscale", action="store_true",
                   help="then upscale the clip, with what is detected for the whole movie (any "
                        "other options, e.g. --type anime or --hevc, go to that upscale)")
    a, extra = p.parse_known_args(argv)
    if extra and not a.upscale:
        p.error("unrecognized arguments: " + " ".join(extra))
    src = Path(a.input)
    if not src.is_file():
        sys.exit(f"Not found: {src}")
    start = 0.0 if a.start.replace(":", "").replace(".", "").strip("0") == "" else hms(a.start)
    length = hms(a.seconds)
    if start is None or length is None:
        sys.exit("Start must look like 90, 15:00 or 1:05:00, and the length like 30 (seconds)")
    if not shutil.which("ffmpeg"):
        sys.exit("Missing tool: ffmpeg")
    m, sec = divmod(int(start), 60)
    tag = f"{m // 60}h{m % 60:02d}m{sec:02d}s" if m >= 60 else f"{m}m{sec:02d}s"
    if start != int(start):
        tag += f"{round((start - int(start)) * 10):d}"    # (a start of 1.5 s is not a start of 1 s)
    out = src.with_name(f"{src.stem} clip {tag}.mkv")
    dur = media_duration(src)
    if dur and start >= dur:
        sys.exit(f"The video is only {hms_text(dur)} long: pick an earlier start.")
    part = out.with_name(out.stem + ".part.mkv")       # (renamed only once it is complete)
    # genpts: an .mpg/.vob or .avi cut mid-stream has packets without timestamps, which .mkv
    # refuses (the cut then failed, or came out a fraction of a second long)
    rc = subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-fflags", "+genpts",
                         "-ss", f"{start:.3f}",
                         "-i", str(src), "-t", f"{length:.3f}", "-map", "0:v:0",
                         *([] if a.video_only else ["-map", "0:a?"]),
                         # DVD/recorder PCM can't go in .mkv as it is: lossless FLAC instead
                         "-c", "copy", *([] if a.video_only else ["-c:a", "flac"]
                                         if has_dvd_pcm(src) else []),
                         "-metadata:s", "DURATION-eng=", str(part)],
                        stdin=subprocess.DEVNULL).returncode
    if rc or not part.exists() or not (media_duration(part) or 0) > 0:
        part.unlink(missing_ok=True)
        sys.exit("Couldn't cut the clip (see the ffmpeg error above).")
    replace_file(part, out)
    print(f"Saved: {out} ({out.stat().st_size / 1e6:.0f} MB)")
    if a.upscale:
        clip_upscale(src, out, extra)


def clip_upscale(src, clip, extra):
    """--clip --upscale: what the full run would make of this stretch. The film/video check and
    the type come from the whole movie (a few seconds on their own can look different), then
    only the clip is upscaled with them."""
    script = str(Path(__file__).resolve())
    print(f"\nChecking the whole movie, as the full run would ({src.name})...", flush=True)
    fd, report = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    rc = 1
    try:
        rc = subprocess.run([sys.executable, script, str(src), "--analyze", *extra],
                            env=dict(os.environ, DVD_UPSCALE_REPORT=report)).returncode
        found = json.loads(Path(report).read_text() or "null")
    except (OSError, ValueError):
        rc, found = rc or 1, None
    finally:
        Path(report).unlink(missing_ok=True)
    if rc or not found:
        sys.exit("Couldn't check the movie (see above).")
    height = build_parser().parse_known_args([str(clip), *extra])[0].height
    out = default_output(clip, height=height)
    work = clip.with_name(clip.stem + "_work")
    # a test made from this clip name before: it is replaced (the clip may have been cut anew)
    if out.is_file() and made_from(out) is not None:
        out.unlink()
    if work.is_dir():
        clean_work_folder(work)
    print(f"\nUpscaling the clip as {TYPE_NAMES[found['type']]}, {found['mode']}"
          + (" (interlaced stretches deinterlaced)" if found.get("combed") else "") + "...",
          flush=True)
    rc = subprocess.run([sys.executable, script, str(clip), str(out), *extra,
                         "--type", found["type"], "--mode", found["mode"],
                         *(["--fix-combed"] if found.get("combed") else []),
                         "--work", str(work)],
                        env=dict(os.environ, DVD_UPSCALE_CLIP_TEST="1")).returncode
    if rc:
        sys.exit(rc)
    if work.is_dir():
        clean_work_folder(work)
    print(f"\nTest done. The original piece: {clip}\nUpscaled: {out}")


# ---------------------------------------------------------------------------------------------
# Self-test: python dvd_upscale.py --self-test
# Quick checks of this script's own logic (no GPU, no movie needed; a few seconds): the queue
# file, the folder scan, the detection decision, face tracking, GPU steps, the PNG helpers and the
# timeouts. Tests that need ffmpeg/ffprobe are skipped when those aren't found. Run it after
# changing the script: it catches a broken queue parser or folder scan before a night's queue
# does. (Defined in here so nothing of it is loaded on a normal run.)

def self_test_main(argv):
    import unittest
    from types import SimpleNamespace
    from unittest import mock
    du = sys.modules[__name__]              # (this script, as the tests see it)

    HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


    def make_clip(path, secs=2, size="320x240", rate="24000/1001", extra=()):
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                        "-i", f"testsrc2=s={size}:r={rate}", "-t", str(secs), *extra,
                        "-c:v", "mpeg2video", str(path)], check=True)


    class TempDir(unittest.TestCase):
        def setUp(self):
            self.dir = Path(tempfile.mkdtemp(prefix="dvd_upscale_test_"))

        def tearDown(self):
            shutil.rmtree(self.dir, ignore_errors=True)


    class SmallHelpers(unittest.TestCase):
        def test_frac(self):
            self.assertEqual(du.frac("30000/1001"), Fraction(30000, 1001))
            self.assertEqual(du.frac("16:9"), Fraction(16, 9))
            self.assertEqual(du.frac("N/A"), Fraction(0))
            self.assertEqual(du.frac("1/0", Fraction(5)), Fraction(5))

        def test_hms(self):
            self.assertAlmostEqual(du.hms("01:23:45.678000000"), 5025.678)
            self.assertEqual(du.hms("12.5"), 12.5)
            self.assertIsNone(du.hms("0"))
            self.assertIsNone(du.hms("N/A"))
            self.assertIsNone(du.hms(None))

        def test_short_time(self):
            self.assertEqual(du.short_time(0.2), "1 s")
            self.assertEqual(du.short_time(59), "59 s")
            self.assertEqual(du.short_time(150), "2 min")      # (rounded: 2.5 -> 2)
            self.assertEqual(du.short_time(3 * 3600 + 5 * 60), "3 h 5 min")

        def test_eta_text(self):
            self.assertTrue(du.eta_text(90).startswith("2 min left, done ~"))
            self.assertTrue(du.eta_text(2 * 3600).startswith("2 h 0 min left"))
            self.assertTrue(du.eta_text(0).startswith("1 min left"))   # never "0 min"

        def test_median(self):
            self.assertEqual(du._median([3, 1, 2]), 2)
            self.assertEqual(du._median([4, 1, 3, 2]), 2.5)
            self.assertIsNone(du._median([]))
            self.assertEqual(du._median([], 0.0), 0.0)
            self.assertEqual(du._median(x for x in (5, 7)), 6)      # any iterable

        def test_quiet_run_never_raises(self):
            self.assertEqual(du._quiet_run(["surely-not-a-program-xyz"]), (-1, b""))
            rc, out = du._quiet_run([sys.executable, "-c", "print('hi')"])
            self.assertEqual((rc, out.strip()), (0, b"hi"))
            rc, out = du._quiet_run([sys.executable, "-c", "import time; time.sleep(5)"],
                                    timeout=0.5)
            self.assertEqual((rc, out), (-1, b""))

        def test_ffprobe_out_gives_up(self):
            with mock.patch.object(du.subprocess, "run",
                                   side_effect=subprocess.TimeoutExpired("ffprobe", 60)):
                self.assertEqual(du.ffprobe_out(["x"]), "")
            with mock.patch.object(du.subprocess, "run", side_effect=FileNotFoundError):
                self.assertEqual(du.ffprobe_out(["x"]), "")

        def test_hung_file_is_waited_for_once(self):
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / "Stuck.mkv"
                f.write_bytes(b"x")
                with mock.patch.object(du, "ffprobe_out", return_value="") as probe:
                    for _ in range(3):
                        self.assertIsNone(du.media_duration(f))
                        self.assertIsNone(du.video_height(f))
                    self.assertEqual(probe.call_count, 1)
                    with mock.patch.object(du.time, "time", return_value=time.time() + 601):
                        du.media_duration(f)              # tried again after 10 minutes
                    self.assertEqual(probe.call_count, 2)


    class QueueFile(TempDir):
        def test_words(self):
            self.assertEqual(du.queue_parse('"My Movie.mkv" out.mkv --type live'),
                             ["My Movie.mkv", "out.mkv", "--type", "live"])
            self.assertEqual(du.queue_parse('a.mkv b.mkv   # a comment "with quotes'),
                             ["a.mkv", "b.mkv"])
            self.assertEqual(du.queue_parse("'It''s.mkv' x.mkv"), ["It's.mkv", "x.mkv"])
            self.assertEqual(du.queue_parse(r'"C:\Movies\A B.mkv"'), [r"C:\Movies\A B.mkv"])
            self.assertEqual(du.queue_parse("# whole line"), [])
            self.assertEqual(du.queue_parse(""), [])

        def test_unclosed_quote(self):
            with self.assertRaises(ValueError):
                du.queue_parse('"Movie.mkv out.mkv')

        def test_test_secs(self):
            self.assertEqual(du.queue_test_secs(["a", "b", "--test", "60"]), 60)
            self.assertEqual(du.queue_test_secs(["a", "--test=30"]), 30)
            self.assertEqual(du.queue_test_secs(["a", "--test", "x"]), 0)
            self.assertEqual(du.queue_test_secs(["a", "b"]), 0)
            self.assertEqual(du.queue_test_secs(["a", "--test"]), 0)

        def test_encodings(self):
            text = '"Été.mkv" "Été 1080p.mkv"\n# comment\n'
            for enc, bom in (("utf-8", b""), ("utf-8", b"\xef\xbb\xbf"), ("utf-16-le", b"\xff\xfe")):
                p = self.dir / f"q_{enc}_{len(bom)}.txt"
                p.write_bytes(bom + text.encode(enc))
                lines = du.queue_lines(p)
                self.assertEqual(du.queue_parse(lines[0]), ["Été.mkv", "Été 1080p.mkv"], enc)


    class Pngs(TempDir):
        def test_black_png_round_trip(self):
            p = self.dir / "000001.png"
            du.write_black_png(p, 64, 36)
            self.assertEqual(du.png_size(p), (64, 36))
            self.assertTrue(du.png_is_black(p))

        def test_not_black(self):
            import numpy as np
            img = np.zeros((36, 64, 3), np.uint8)
            img[20, 30] = (0, 1, 0)                     # one pixel off black
            p = self.dir / "sub" / "000001.png"
            p.parent.mkdir()
            du.write_png(p, img)
            self.assertEqual(du.png_size(p), (64, 36))
            self.assertFalse(du.png_is_black(p))
            du.write_png(p, np.zeros((36, 64, 3), np.uint8))
            self.assertTrue(du.png_is_black(p))

        def test_garbage_is_not_black(self):
            p = self.dir / "x.png"
            p.write_bytes(b"not a png at all")
            self.assertFalse(du.png_is_black(p))
            self.assertFalse(du.png_is_black(self.dir / "missing.png"))

        @unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg")
        def test_write_png_reads_back_exactly(self):
            import numpy as np
            rng = np.random.default_rng(1)
            img = rng.integers(0, 256, (24, 40, 3), dtype=np.uint8)
            p = self.dir / "f" / "000001.png"
            p.parent.mkdir()
            du.write_png(p, img)
            raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(p), "-f", "rawvideo",
                                  "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
            self.assertEqual(raw, img.tobytes())


    class Faces(unittest.TestCase):
        @staticmethod
        def face(x, y):
            """A detection (landmarks, score) of a face big enough for full strength: eyes
            between FACE_FULL_EYES and FACE_TAPER_EYES source pixels apart (frames at scale 2)."""
            import numpy as np       # 5 landmarks: eyes, nose, mouth corners
            e = (du.FACE_FULL_EYES + du.FACE_TAPER_EYES)       # (2x the middle of that range)
            return (np.array([[x, y], [x + e, y], [x + e / 2, y + e / 2], [x + e / 5, y + e],
                              [x + 4 * e / 5, y + e]], float), 0.9)

        def test_steady_face_everywhere(self):
            n = 12
            plan = du.face_plan([[self.face(100, 50)] for _ in range(n)], n, 0.6)
            self.assertEqual(len(plan), n)
            self.assertTrue(all(len(p) == 1 for p in plan))
            # reaches both chunk edges: no fade-in/out there (no pulse at the seams)
            self.assertAlmostEqual(plan[0][0][1], 0.6)
            self.assertAlmostEqual(plan[-1][0][1], 0.6)

        def test_flash_of_a_face_is_dropped(self):
            n = 20
            dets = [[] for _ in range(n)]
            dets[9], dets[10] = [self.face(100, 50)], [self.face(101, 50)]
            self.assertTrue(all(p == [] for p in du.face_plan(dets, n, 0.6)))

        def test_one_missed_detection_is_filled(self):
            n = 16
            dets = [[self.face(100, 50)] for _ in range(n)]
            dets[7] = []
            plan = du.face_plan(dets, n, 0.6)
            self.assertEqual(len(plan[7]), 1)


    class DetectionDecision(unittest.TestCase):
        """detect_content: how the picture check and the motion check are combined."""

        def run_with(self, pictures, motion, flat):
            a = dict(content=pictures, confidence="high", reasons="", scores={"flat_share": flat})
            b = dict(content=motion, confidence="high", reasons="", scores={"alt": 0.2})
            with mock.patch.object(du, "_cs_detect_content", return_value=a), \
                    mock.patch.object(du, "_ct_detect_content", return_value=b):
                return du.detect_content("x.mkv")

        def test_anime_needs_both_and_flat_areas(self):
            self.assertEqual(self.run_with("anime", "anime", 0.8)["content"], "anime")
            r = self.run_with("anime", "anime", 0.45)          # silent-film look
            self.assertEqual((r["content"], r["confidence"]), ("live", "low"))
            self.assertEqual(self.run_with("anime", "live", 0.8)["content"], "live")
            self.assertEqual(self.run_with("live", "anime", 0.8)["content"], "live")

        def test_cgi_only_when_both_agree(self):
            self.assertEqual(self.run_with("cgi", "cgi", 0.2)["content"], "cgi")
            self.assertEqual(self.run_with("cgi", "live", 0.2)["content"], "live")
            self.assertEqual(self.run_with("live", "live", 0.2)["content"], "live")


    class GpuSteps(TempDir):
        def setUp(self):
            super().setUp()
            self.env = mock.patch.dict(os.environ, {"DVD_UPSCALE_GPU_STEPS": str(self.dir / "s.json")})
            self.env.start()
            du.GPU_STEP[0], du.FRAMES_OK[0] = 0, 0

        def tearDown(self):
            self.env.stop()
            du.GPU_STEP[0], du.FRAMES_OK[0] = 0, 0
            super().tearDown()

        def args(self, model="realesrgan-x2plus", **kw):
            return SimpleNamespace(model=model, gpu=None, gpu_threads=None, tile=None, engine="exe",
                                   **kw)

        def test_steps_down_and_is_remembered(self):
            a = self.args()
            self.assertEqual(du.gpu_load(a), (2, None))
            self.assertTrue(du.lower_gpu_load(a))
            self.assertEqual(du.gpu_load(a), (1, None))
            du.GPU_STEP[0] = 0                          # a new run
            self.assertTrue(du.load_gpu_step(a))
            self.assertEqual(du.gpu_load(a), (1, None))

        def test_last_step(self):
            a = self.args()
            while du.lower_gpu_load(a):
                pass
            self.assertEqual(du.gpu_load(a), (1, 32))
            self.assertFalse(du.lower_gpu_load(a))

        def test_user_settings_kept(self):
            a = self.args(model="realesr-animevideov3")
            a.tile = 128
            self.assertEqual(du.gpu_load(a), (8, 128))
            self.assertIn("128-pixel tiles", du.gpu_load_text(a))

        def test_no_step_down(self):
            a = self.args(no_step_down=True)
            self.assertFalse(du.lower_gpu_load(a))


    class WorkFolder(TempDir):
        def test_clean_keeps_foreign_files(self):
            w = self.dir / "Movie_work"
            (w / "tmp_00003").mkdir(parents=True)
            for name in ("chunk_00001.mkv", "chunk_00002.part.mkv", "settings.json", "list.txt",
                         "my notes.txt"):
                (w / name).write_text("x")
            du.clean_work_folder(w)
            self.assertEqual(sorted(p.name for p in w.iterdir()), ["my notes.txt"])
            (w / "my notes.txt").unlink()
            du.clean_work_folder(w)
            self.assertFalse(w.exists())


    @unittest.skipUnless(HAVE_FFMPEG, "needs ffmpeg and ffprobe")
    class Headers(TempDir):
        def test_header_and_cache(self):
            clip = self.dir / "Clip.mkv"
            make_clip(clip, secs=2)
            self.assertAlmostEqual(du.media_duration(clip), 2.0, delta=0.1)
            self.assertEqual(du.video_height(clip), 240)
            self.assertIsNone(du.made_from(clip))
            self.assertFalse(du.is_upscaled_output(clip))
            with mock.patch.object(du, "ffprobe_out", side_effect=AssertionError("not cached")):
                self.assertEqual(du.video_height(clip), 240)       # read once, kept
            # a changed file is read again
            make_clip(clip, secs=3, size="640x480")
            os.utime(clip, (time.time() + 5, time.time() + 5))
            self.assertEqual(du.video_height(clip), 480)
            self.assertAlmostEqual(du.media_duration(clip), 3.0, delta=0.1)

        def test_our_label(self):
            src, out = self.dir / "Movie.mkv", self.dir / "Out.mkv"
            make_clip(src)
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-c", "copy",
                            "-metadata", f"comment={du.OUTPUT_TAG} from {du.source_id(src)}",
                            str(out)], check=True)
            self.assertEqual(du.made_from(out), du.source_id(src))
            self.assertTrue(du.is_upscaled_output(out))

        def test_unreadable(self):
            bad = self.dir / "bad.mkv"
            bad.write_bytes(b"\x00" * 1000)
            self.assertIsNone(du.media_duration(bad))
            self.assertIsNone(du.video_height(bad))
            self.assertIsNone(du.made_from(bad))
            self.assertIsNone(du.media_duration(self.dir / "missing.mkv"))

        def test_folder_scan(self):
            f = self.dir
            (f / "anime").mkdir()
            make_clip(f / "Short.mkv", secs=2)                  # under 5 minutes: skipped
            make_clip(f / "anime" / "Show.mkv", secs=2)
            make_clip(f / "HD.mkv", secs=2, size="1280x720")
            make_clip(f / "Movie 1080p.mkv", secs=2)              # one of our outputs
            (f / "VTS_01_1.VOB").write_bytes(b"\x00")
            old = time.time() - 3600
            for p in f.rglob("*.*"):
                os.utime(p, (old, old))
            make_clip(f / "Copying.mkv", secs=2)                 # just changed: still copying?
            long = mock.patch.object(du, "media_duration", return_value=600.0)
            with long:
                jobs, waiting = du.folder_jobs(f, [])
            self.assertEqual(waiting, ["Copying.mkv"])
            by_name = {j[1]: j for j in jobs}
            self.assertIn("already HD", by_name["HD.mkv"][3])
            self.assertIn("pieces of a DVD", by_name["VTS_01_1.VOB"][3])
            self.assertNotIn("Movie 1080p.mkv", by_name)
            show = by_name[str(Path("anime") / "Show.mkv")]
            self.assertEqual(show[2][show[2].index("--type") + 1], "anime")
            self.assertEqual(show[2][1], str(Path("1080p Upscale") / "anime" / "Show 1080p.mkv"))
            jobs, _ = du.folder_jobs(f, [])                      # real lengths: 2 s
            self.assertIn("shorter than 5 minutes", {j[1]: j for j in jobs}["Short.mkv"][3])

        def test_queue_finished(self):
            src, out = self.dir / "Movie.mkv", self.dir / "Movie 1080p.mkv"
            make_clip(src, secs=3)
            self.assertFalse(du.queue_finished(self.dir, [src.name, out.name]))
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-t", "1",
                            "-c", "copy", str(out)], check=True)
            self.assertFalse(du.queue_finished(self.dir, [src.name, out.name]))   # too short
            self.assertTrue(du.queue_finished(self.dir, [src.name, out.name, "--test", "1"]))


    class CommandLine(unittest.TestCase):
        def test_parser_knows_every_queue_value_option(self):
            p = du.build_parser()
            known = {s for act in p._actions for s in act.option_strings}
            for opt in du.VALUE_OPTS:
                self.assertIn(opt, known)

        def test_trained(self):
            a = du.build_parser().parse_args(["in.mkv", "--trained"])
            self.assertTrue(a.trained)

        def test_trained_picks_x2_or_x4(self):
            with tempfile.TemporaryDirectory() as d:
                models = Path(d) / "models"
                models.mkdir()
                put = lambda n: [(models / f"{n}.{e}").write_text("x") for e in ("param", "bin")]
                pick = lambda height, scale=None: (lambda a: (du.trained_model(a), a.scale))(SimpleNamespace(
                    type="live", height=height, trained_scale=scale, esrgan_path=str(Path(d) / "esrgan")))
                with mock.patch("builtins.print"):
                    put("ai-live-x2")
                    self.assertEqual(pick(1080), ("ai-live-x2", 2))
                    self.assertEqual(pick(2160), ("ai-live-x2", 2))         # (no x4 model: the x2 one)
                    self.assertEqual(pick(1080, 4), ("ai-live-x4", 4))      # (asked for: reported missing)
                    put("ai-live-x4")
                    self.assertEqual(pick(2160), ("ai-live-x4", 4))
                    self.assertEqual(pick(1080), ("ai-live-x2", 2))
                    self.assertEqual(pick(2160, 2), ("ai-live-x2", 2))

        def test_commands_text(self):
            self.assertIn("--all", du.commands_text())


    class Load(unittest.TestCase):
        def test_lines(self):
            w = du.LoadWatch(start=False)
            self.assertEqual(w.lines(), [])
            g = lambda busy, temp: dict(index="0", name="X", busy=busy, clock=1500.0, temp=temp,
                                        power=None, mem=2048.0, mem_max=8192.0)
            w.add(20.0, [g(90.0, 60.0)])
            w.add(None, [g(100.0, 70.0)])
            w.add(60.0, [])
            self.assertEqual(w.lines(since_last=True), [
                "CPU: 40% avg, 60% peak",
                "GPU: 95% avg, 100% peak | 1500 MHz avg, 1500 MHz peak | 65 C avg, 70 C peak"
                " | VRAM 2.0 GB avg, 2.0 GB peak of 8.0"])
            w.add(10.0, [])
            self.assertEqual(w.lines(since_last=True), ["CPU: 10% avg, 10% peak"])
            self.assertEqual(w.lines()[0], "CPU: 30% avg, 60% peak")    # (the whole run)

        def test_cpu_times(self):
            t = du.cpu_times()
            if t:
                self.assertLessEqual(t[0], t[1])
            self.assertIsInstance(du.hardware_text()[0], str)

    tests = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(case)
        for case in list(locals().values())
        if isinstance(case, type) and issubclass(case, unittest.TestCase))
    verbosity = 1 if "-q" in argv else 2
    ok = unittest.TextTestRunner(verbosity=verbosity).run(tests).wasSuccessful()
    return 0 if ok else 1


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):      # a name the console/log can't show: '?', not a crash
        try:    # (the face worker's output is read by this script, as UTF-8: any path in it)
            stream.reconfigure(errors="replace", **(
                {"encoding": "utf-8"} if sys.argv[1:2] == ["--faces-worker"] else {}))
        except (AttributeError, ValueError):
            pass
    # ffmpeg/ffprobe/the upscaler next to this script work even when the movies are elsewhere
    # (every ffmpeg call uses the bare name, and --all/--queue run each movie in its folder),
    # and come first: a folder set up by setup.bat holds everything, and another ffmpeg on the
    # PATH may be older or lack vid.stab
    _here = str(Path(__file__).resolve().parent)
    if any(shutil.which(t, path=_here) for t in ("ffmpeg", "realesrgan-ncnn-vulkan")):
        os.environ["PATH"] = _here + os.pathsep + os.environ.get("PATH", "")
    # commands that run on their own, first word only (the rest of the line is theirs)
    tools = {
        "--faces-worker": faces_worker_main,        # (a chunk's face restoration: see Chunk)
        "--ncnn-models": ncnn_models_main,          # (every model on the same frames)
        "--gpu-detect": gpu_detect_main,            # (the GPUs, the profile; made if there is none)
        "--ncnn-auto": ncnn_auto_main,              # (all of the GPU set-up and a timed test)
        "--ncnn-winograd": ncnn_winograd_main,      # (the winograd variants the GPU survives)
        "--ncnn-bench-gpu": ncnn_bench_gpu_main,    # (which GPU is faster)
        "--ncnn-bench": ncnn_bench_main,            # (the fastest settings that keep the picture)
        "--ncnn-stress": ncnn_stress_main,          # (find the ncnn options the GPU survives)
        "--self-test": self_test_main,              # (checks of this script's own logic)
    }
    if sys.argv[1:2] and sys.argv[1] in tools:
        sys.exit(tools[sys.argv[1]](sys.argv[2:]))
    if sys.argv[1:2] == ["--ncnn-upscaler"]:         # (the current ncnn: see esrgan_cmd)
        rc = ncnn_upscaler_main(sys.argv[2:])
        # (leave at once, without Python's and ncnn's clean-up: on Windows with the NVIDIA
        # driver the current ncnn can crash (0xC0000005) while it shuts down, after every frame
        # was written, and the exit code then said the upscale had failed)
        sys.stdout.flush()
        sys.stderr.flush()
        if sys.platform == "win32":
            # (os._exit is ExitProcess, which still unloads every DLL, and the crash is there:
            # TerminateProcess ends the process without that)
            try:
                import ctypes
                k32 = ctypes.windll.kernel32
                k32.GetCurrentProcess.restype = ctypes.c_void_p
                k32.TerminateProcess(ctypes.c_void_p(k32.GetCurrentProcess()), int(rc or 0))
            except (OSError, AttributeError, ValueError):
                pass
        os._exit(rc or 0)
    if sys.argv[1:2] == ["--clip"]:
        try:
            clip_main(sys.argv[2:])
        except KeyboardInterrupt:
            sys.exit(130)
        sys.exit(0)
    if sys.argv[1:] in (["--commands"], ["commands"]):
        print(commands_text())      # the cheat sheet: only when asked for
        sys.exit(0)
    if len(sys.argv) == 1:
        print('Usage: python dvd_upscale.py "Movie.mkv"    (upscale one movie)\n'
              '       python dvd_upscale.py --all          (every movie in this folder)\n'
              '       python dvd_upscale.py --commands     (every command and option)')
        sys.exit(0)
    if "--no-guard" in sys.argv:                   # (see protect_run; the movies inherit it)
        while "--no-guard" in sys.argv:
            sys.argv.remove("--no-guard")
        os.environ["DVD_UPSCALE_NO_GUARD"] = "1"
    phone_port = phone_args(sys.argv)              # (the phone page: before anything parses argv)
    queue_mode = any(w in ("--queue", "--all") or w.startswith(("--queue=", "--all="))
                     for w in sys.argv[1:])
    if phone_port:
        phone_start(phone_port, queue_mode)
    phone_setup(queue_mode)
    try:
        if queue_mode:
            queue_main(sys.argv[1:])
        else:
            main()
    except KeyboardInterrupt:
        if not queue_mode:
            print("\nStopped. Re-run the same command to resume.", file=sys.stderr)
        sys.exit(130)       # distinct code: the queue stops instead of moving on
    except (subprocess.SubprocessError, RuntimeError) as e:
        sys.exit(f"\nFailed: {e}\nRe-run the same command to retry from the last finished chunk.")
    except OSError as e:                         # disk full, file in use, missing folder...
        sys.exit(f"\nFailed: {e}\nFix that and re-run the same command: finished chunks are kept.")
