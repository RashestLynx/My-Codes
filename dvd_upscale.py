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
anime/live/cgi/vhs (or a folder of that name with --all) overrides it. A resumed movie keeps the
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

Every movie in a folder, one after another:
  python dvd_upscale.py --all                  # all movies in the current folder
  python dvd_upscale.py --all "D:\Movies" --shutdown
- Movies in a subfolder named "anime", "live", "cgi" or "vhs" get that --type, whatever else you
  pass; movies loose in the folder are detected one by one (or get the --type you give, e.g.
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
- While it runs the PC is kept from going to sleep (single movies too). Keep it plugged in, and if you
  close the lid, set the lid action to "Do nothing" for when it's plugged in.
Progress is logged to queue_log.txt next to the queue file.
"""
import argparse, json, math, operator, os, re, shutil, signal, statistics, subprocess, sys
import tempfile
import threading, time
from bisect import bisect_right as _bisect
from collections import Counter
from operator import add, sub
from fractions import Fraction
from pathlib import Path

OUTPUT_TAG = "Upscaled by dvd_upscale.py"


def nostdin(cmd):
    """ffmpeg reads the console keyboard ('q' quits, 'c'/'d' wait for input): never let it."""
    cmd = [str(c) for c in cmd]
    return cmd[:1] + ["-nostdin"] + cmd[1:] if Path(cmd[0]).stem.lower() == "ffmpeg" else cmd


def run(cmd):
    subprocess.run(nostdin(cmd), check=True, stdin=subprocess.DEVNULL)


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
    if not sys.stdout.isatty():
        return
    width = max(20, shutil.get_terminal_size((80, 24)).columns - 1)  # a full line would wrap
    sys.stdout.write("\r" + msg[:width].ljust(width) + "\r")
    sys.stdout.flush()


def say(msg):
    status_line()
    print(msg, flush=True)


def short_time(secs):
    secs = max(1, round(secs))
    if secs < 60:
        return f"{secs} s"
    m = round(secs / 60)
    return f"{m // 60} h {m % 60} min" if m >= 60 else f"{m} min"


def ffmpeg_progress(cmd, secs, label):
    """Run an ffmpeg command showing label, % done and time left on the progress line; secs is
    the length of what it writes. Returns (exit code, its error output, seconds it took)."""
    cmd = nostdin(cmd)
    cmd = cmd[:1] + ["-progress", "pipe:1", "-nostats"] + cmd[1:]
    t0 = time.time()
    with tempfile.TemporaryFile() as err:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err,
                             encoding="utf-8", errors="replace")
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


def run_step(cmd, secs, label):
    """run() for a long step: progress while it runs, how long it took when done."""
    rc, text, took = ffmpeg_progress(cmd, secs, label)
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
         ":stream_tags:format=start_time,duration", "-of", "json", str(src)]))
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
                           errors="replace")
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
        if getattr(a, "pal", False):
            f += ["fieldmatch", "yadif=deint=interlaced"]   # PAL film is 2:2: nothing to drop
        else:
            f += ["repeatfields", "fieldmatch", "yadif=deint=interlaced", "decimate"]
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
    if getattr(a, "dvd_trim", 0):
        f += [f"trim=start_frame={a.dvd_trim}", "setpts=PTS-STARTPTS"]   # chunk warm-up frames
    f += [f"fps={a.fps}"]
    return ",".join(f)


def postfilter(a):
    # exact width (multiple of 8, e.g. 1920 or 1440): some TVs and hardware decoders reject
    # odd sizes like 1918x1080
    f = [f"scale={a.out_w}:{a.height}:flags=lanczos+accurate_rnd",
         "format=yuv420p10le",
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
         "-of", "csv=p=0", str(path)], encoding="utf-8", errors="replace").strip()
    return float(frac(out))


def count_frames(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)], encoding="utf-8", errors="replace")
    out = out.strip().split(",")[0]
    return int(out) if out.isdigit() else 0


def gpu_list(a):
    """--gpu 0,1 -> ["0", "1"]; no --gpu -> [None] (the upscaler picks the dedicated one)."""
    return str(a.gpu).replace(" ", "").split(",") if a.gpu is not None else [None]


def esrgan_cmd(a, src, dst, size=None, gpu=None):
    compact = any(m in a.model for m in ("animevideov3", "general"))
    gpu = gpu if gpu is not None else gpu_list(a)[0]
    # threads to load:upscale:save frames (default 1:2:2): reading and writing the PNGs is CPU
    # work that otherwise leaves the GPU waiting; the small (compact) models also get 4 frames
    # on the GPU at once (each frame is upscaled on its own: same result); the big x2plus/x4plus
    # models stay at 2, as more could run out of GPU memory
    cmd = [a.esrgan_path, "-i", src, "-o", dst, "-n", a.model, "-s", a.scale, "-f", "png",
           "-j", f"2:{(a.gpu_threads or 4) if compact else 2}:4"]
    models = models_dir(a)
    if models.is_dir():
        cmd += ["-m", models]
    if gpu is not None:
        cmd += ["-g", gpu]
    if a.tile:
        cmd += ["-t", a.tile]
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
                        r"VK_ERROR_DEVICE_LOST|device lost|(en|de)code image .* failed", re.I)


LANE_STATUS = {}     # what each helper GPU is doing, shown on the main progress line
# each helper GPU's upscale while it runs: [frames done, of, started, last new frame] (times)
LANE_PROGRESS = {}
# the chunks being upscaled right now (on any GPU): chunk -> (bytes of frames it writes, frames)
UPSCALING = {}
DISK_LOCK = threading.Lock()


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
    hung = False
    with open(log_path, "wb") as log:
        p = subprocess.Popen([str(c) for c in esrgan_cmd(a, src, dst, size, gpu)],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            # watchdog, counted in 1 s waits rather than clock time so a laptop that slept in
            # between isn't taken for a stall
            last, same, all_there, tick = -1, 0, 0, 0
            while True:
                n = sum(1 for _ in os.scandir(dst))
                if lane:
                    LANE_STATUS[lane] = f"{lane}: {label} {n}/{n_in}"
                    now, tick = time.time(), tick + 1
                    prog = LANE_PROGRESS.setdefault(lane, [0, n_in, now, now])
                    if n > prog[0]:
                        prog[0], prog[3] = n, now
                    if lane_stopped(a, lane):
                        raise RuntimeError("stopped")
                    # a GPU error: the chunk goes back now, not once the upscaler gives up (or
                    # hangs), and the main GPU does it
                    if tick % 5 == 0:
                        bad = [x for x in log_path.read_bytes().decode("utf-8", "replace")
                               .splitlines() if GPU_ERRORS.search(x)]
                        if bad:
                            raise RuntimeError(f"the upscaler reported errors: "
                                               f"{bad[0].strip()[:100]}")
                else:
                    status_line(f"  {label}: upscaling frame {n} of {n_in}"
                                + "".join(f" | {x}" for x in list(LANE_STATUS.values())))
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
        if lane:        # a helper GPU: one line, in the note that it stopped helping
            raise RuntimeError(("the upscaler reported errors" if errors else
                                f"the upscaler failed (exit code {p.returncode})")
                               + (f": {lines[0 if errors else -1].strip()[:100]}"
                                  if lines else ""))
        status_line()
        print("Upscaler output (end):\n" + "\n".join(lines[-25:]), flush=True)
        if re.search(r"memory|vkAllocate", text, re.I):
            print("(the GPU may have run out of memory: try adding --tile 128)", flush=True)
        elif re.search(r"encode image", text):
            print("(it couldn't write the frames: is the work folder's drive full?)", flush=True)
        if errors:
            raise RuntimeError(f"the upscaler reported errors (exit code {p.returncode})")
        raise subprocess.CalledProcessError(p.returncode, f"{a.esrgan} (upscaler)")


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
    if len(before) == len(after):
        for i, (x, y) in zip(picks, zip(before, after)):
            if abs(x - y) > 20:
                raise RuntimeError(f"upscaled frame {i} looks wrong (brightness {y:.0f} instead "
                                   f"of about {x:.0f}: black or garbled - a GPU fault?)")


def models_dir(a):
    return Path(a.esrgan_path).resolve().parent / "models"     # resolve() follows symlinks


def model_installed(a):
    d = models_dir(a)
    if not d.is_dir():
        return True     # unusual install layout: let the upscaler find its own models
    names = [a.model, f"{a.model}-x{a.scale}"]       # animevideov3 files carry the scale
    return any((d / f"{n}.param").exists() and (d / f"{n}.bin").exists() for n in names)


def keep_awake():
    """CPU/GPU work doesn't count as activity for Windows: keep the PC from sleeping while this
    runs (released automatically when it exits; closing the lid still sleeps)."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


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


class Chunk:
    """One chunk of the movie, in three steps: read its frames out of the source (extract), run
    them through the upscaler (upscale, on the GPU) and encode them into the chunk file
    (finish). It is set up in the main thread, where the settings that differ from chunk to
    chunk are fixed, so extract and finish can then run in other threads: the next chunk's
    frames are read, and the previous chunk encoded, while this one is upscaled."""

    def __init__(self, a, idx, start, length, expected, work, label, last=False, warm=False):
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
        self.encode = None

    def clear_frames(self):
        # GBs of frames that a retry makes again anyway: don't leave them filling the drive
        # (upscaler_log.txt stays, for a look at what went wrong)
        for d in ("in", "out"):
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
            # fast PNG compression: the same pixels, written several times faster
            run(["ffmpeg", "-y", "-v", "error", *self.src, "-vf", self.pre,
                 "-frames:v", self.expected, "-compression_level", "1", pngs(tmp / "in")])
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
            frames = n_in * w * h * a.scale ** 2 * 1.8      # measured ~1.7 bytes a pixel
            need = frames + 1e9
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
                        break
                if self.make_room and self.make_room():
                    continue        # the other GPUs stopped and their frames are gone: again
                if self.lane:
                    raise RuntimeError("the drive with the work folder is too full for two "
                                       "chunks at once")
                raise RuntimeError(
                    f"the drive with the work folder needs about {need / 1e9:.0f} GB free for "
                    f"this chunk's upscaled frames and has {free / 1e9:.1f} GB: make room (or "
                    "use --work on another drive)")
            try:
                for attempt in (1, 2):
                    try:
                        run_upscaler(a, tmp / "in", tmp / "out", n_in, label, (w, h),
                                     self.gpu, self.lane)
                        n_out = len(list((tmp / "out").glob("*.png")))
                        if n_out != n_in:
                            raise RuntimeError(f"upscaler produced {n_out} of {n_in} frames "
                                               "(GPU out of memory? try --tile 128)")
                        check_frames(a, tmp, n_in, strict=bool(self.lane))
                        break
                    except (RuntimeError, subprocess.CalledProcessError) as e:
                        # (no second try on a helper GPU: the main GPU redoes its chunk)
                        if attempt == 2 or self.lane:
                            raise
                        # a new upscaler process gets a fresh GPU device (after a driver
                        # reset, say)
                        status_line()
                        print(f"  {label}: {str(e).rstrip('.')} - trying this chunk once more.",
                              flush=True)
                        shutil.rmtree(tmp / "out", ignore_errors=True)
                        (tmp / "out").mkdir()
            finally:
                with DISK_LOCK:
                    UPSCALING.pop(self, None)
            ins = ["-framerate", a.fps, "-i", pngs(tmp / "out")]
            if a.ai_blend < 1:
                # mix the AI frames with a plain upscale of the same input frames, so frames
                # where the model adds detail and frames where it doesn't look less different
                ins += ["-framerate", a.fps, "-i", pngs(tmp / "in")]
                graph = (f"[1:v]scale=iw*{a.scale}:ih*{a.scale}:flags=lanczos,format=gbrp[plain];"
                         f"[0:v]format=gbrp[ai];"
                         f"[ai][plain]blend=all_mode=normal:all_opacity={a.ai_blend}")
            else:
                graph = "[0:v]null"
            k = min(12, n_in - 1)
            if a.smooth > 0 and k > 0:
                # the temporal smoothing would start "cold" on every chunk's first frame (a
                # visible sharp-to-soft breath every chunk); warm it up on the next k frames
                # played backwards, then cut those warm-up frames off again
                graph += (f",split[wa][wb];[wa]trim=start_frame=1:end_frame={k + 1},reverse[wr];"
                          f"[wr][wb]concat=n=2:v=1:a=0,{postfilter(a)},"
                          f"trim=start_frame={k},setpts=PTS-STARTPTS")
            else:
                graph += "," + postfilter(a)
            self.encode = ["ffmpeg", "-y", "-v", "error", *ins, "-filter_complex", graph,
                           *encode_args(a), self.part]
        except BaseException:
            self.clear_frames()
            raise

    def fast(self):
        """--fast: filters only, straight into the chunk file. Returns the frame count."""
        a = self.a
        run(["ffmpeg", "-y", "-v", "error", *self.src,
             "-vf", self.pre + "," + postfilter(a), "-frames:v", self.expected,
             *encode_args(a), self.part])
        try:
            got = count_frames(self.part)
        except subprocess.CalledProcessError:   # encoder got no frames: file is unreadable
            got = 0
        if 0 < got < self.expected and not self.last:
            status_line()
            print(gap_message(self.idx, self.expected - got, self.expected), flush=True)
            run(["ffmpeg", "-y", "-v", "error", *self.src, "-vf", self.pre
                 + ",tpad=stop_mode=clone:stop=-1," + postfilter(a), "-frames:v", self.expected,
                 *encode_args(a), self.part])
        return got

    def finish(self):
        """Encode (unless --fast did already) and keep the chunk. Returns (path, warning)."""
        if self.encode:
            try:
                run(self.encode)
            except BaseException:
                self.clear_frames()
                raise
        n = count_frames(self.part)
        os.replace(self.part, self.out)
        shutil.rmtree(self.tmp, ignore_errors=True)
        return self.out, (f"  WARNING: chunk {self.idx} has {n} frames, expected {self.expected} "
                          "(can cause small sync drift)"
                          if n != self.expected and not self.last else None)


class Background:
    """One step running in another thread; wait() hands back its result or raises its error."""

    def __init__(self, fn):
        self.box = {}

        def work_():
            try:
                self.box["result"] = fn()
            except BaseException as e:
                self.box["error"] = e
        self.thread = threading.Thread(target=work_, daemon=True)
        self.thread.start()

    def wait(self):
        while self.thread.is_alive():
            self.thread.join(0.5)       # (a timeout keeps Ctrl+C working while it waits)
        if "error" in self.box:
            raise self.box["error"]
        return self.box.get("result")


def probe_or_exit(src):
    try:
        return probe(src)
    except subprocess.CalledProcessError:
        sys.exit(f"Could not read '{src}' (see the ffprobe error above)")
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
_cs_all__ = ["detect_content"]

# ---- model (fitted on the training split, see REPORT.md) ---------------------------------------
_cs_W_ANIME = (3.2616312366286953, 2.795601085467864, 0.07764657454479618)   # logit(eshare), logit(eshare2x), noise
_cs_B_ANIME = 9.154252224821379
_cs_W_CGI = (0.5447004345669803, -0.4198416396817829)                        # sat_std, logit(eshare)
_cs_B_CGI = -5.499787115874285

_cs_N_SAMPLES = 32
_cs_BW_INK, _cs_BW_INK_HIGH = 1.2, 1.5               # dark-line ratio; non-anime training clips: max 1.19 (Caminandes 2)
_cs_NPL = 7                                  # planes per sample: Y, median(Y), Laplacian, black/white top-hat, U, V
_cs_ABS = {}


def _cs_logit(x):
    x = min(max(x, 0.005), 0.95)
    return math.log(x / (1.0 - x))


def _cs_sig(z):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))


def _cs_run(cmd, timeout=60):
    try:
        p = subprocess.run([str(c) for c in cmd], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL, timeout=timeout)
        return p.returncode, p.stdout
    except (OSError, subprocess.SubprocessError):
        return -1, b""


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
    rc, out = _cs_run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
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
            rc, pk = _cs_run([ffprobe, "-v", "error", "-read_intervals", "%+20", "-show_entries",
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
    rc, out = _cs_run(cmd, timeout=120)
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

_ct_NOWIN = {}
if os.name == "nt":      # no console window flashing for every ffmpeg call
    _ct_NOWIN["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


def _ct_run(cmd, timeout=60):
    try:
        p = subprocess.run([str(c) for c in cmd], stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           timeout=timeout, **_ct_NOWIN)
        return p.stdout
    except (OSError, subprocess.SubprocessError):
        return b""


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
    out = _ct_run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=avg_frame_rate,r_frame_rate,duration:stream_tags:format=duration,start_time",
                "-of", "json", path])
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
    out = _ct_run([ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals",
                "%.3f%%+4" % max(0.0, t), "-show_entries", "packet=pts_time", "-of", "csv=p=0", path])
    ts = sorted(_ct_num(x) for x in out.decode("ascii", "replace").split() if x and x != "N/A")
    if len(ts) < 20 or ts[-1] - ts[0] <= 0.5:
        return 0.0
    return (len(ts) - 1) / (ts[-1] - ts[0])


def _ct_has_frame(path, t, ffmpeg):
    out = _ct_run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-ss", "%.3f" % t,
                "-i", path, "-an", "-sn", "-dn", "-frames:v", "1", "-vf", "scale=16:16",
                "-f", "rawvideo", "-pix_fmt", "gray", "-"], timeout=60)
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
    out = _ct_run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
                "-i", path, "-an", "-sn", "-dn", "-filter_complex", graph,
                "-frames:v", str(n * _ct_BURST_FRAMES - 1), "-fps_mode", "passthrough",
                "-f", "rawvideo", "-pix_fmt", "gray", "-"], timeout=300)
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
    out = _ct_run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", str(threads),
                "-noaccurate_seek", "-ss", "%.3f" % max(0.0, t), "-i", path,
                "-an", "-sn", "-dn", "-filter_complex", graph, "-frames:v", str(_ct_BURST_FRAMES),
                "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"])
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


def _ct_median(v):
    s = sorted(v)
    n = len(s)
    if not n:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


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
    med = [_ct_median(s[p:n:5]) for p in range(5)]
    p = min(range(5), key=lambda q: med[q])
    rest = _ct_median([x for q in range(5) if q != p for x in s[q:n:5]])
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
             noise=_ct_median(noise) if noise else -1.0, static=len(noise))
    for k in ("mid", "line", "low", "flat", "strong", "peak"):
        f[k] = _ct_median([x[k] for x in sp]) if sp else -1.0
        f[k + "4"] = _ct_median([x[k + "4"] for x in sp]) if sp else -1.0
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
_src_all__ = ["detect_source"]

_src_LOSSLESS = {"huffyuv", "ffvhuff", "ffv1", "utvideo", "lagarith", "rawvideo", "v210", "v410",
            "yuv4", "r210", "magicyuv", "y41p", "ayuv", "zlib", "mszh", "cllc", "vble"}
_src_CAPTURE_EXT = (".avi", ".mpg", ".mpeg", ".dv")
_src_NPOS = 10                 # sample positions
_src_NFRAMES = 2               # consecutive frames per position (each gives 2 fields)
_src_TIME_BUDGET = 11.0        # stop sampling after this many seconds (hard limit for callers: 15 s)


# ---------------------------------------------------------------------------------- helpers ---
def _src_run(cmd, timeout):
    """Run a command, return stdout bytes ('' on any failure). Never raises."""
    try:
        kw = {}
        if os.name == "nt":                       # no console window flashing when run from a GUI
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        p = subprocess.run([str(c) for c in cmd], stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           timeout=timeout, **kw)
        return p.stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return b""


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


def _src_median(v):
    v = sorted(v)
    n = len(v)
    if not n:
        return None
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def _src_probe(path, ffprobe, info):
    """Stream facts. Uses dvd_upscale.py's probe() dict (w, h, sar, fps, duration) when given."""
    meta = {}
    out = _src_run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=codec_name,width,height,sample_aspect_ratio,avg_frame_rate,r_frame_rate,"
                "field_order,duration:stream_tags:format=duration,format_name",
                "-of", "json", path], 30)
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
    out = _src_run([ffprobe, "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#36",
                "-show_entries", "frame=repeat_pict", "-of", "csv=p=0", path], 20)
    vals = [ln.strip().strip(",") for ln in out.decode("ascii", "replace").splitlines()]
    vals = [v for v in vals if v.lstrip("-").isdigit()]
    if len(vals) < 8:
        return None
    return sum(1 for v in vals if int(v) > 0) / len(vals)


def _src_grab(path, ffmpeg, t, W, H, n, timeout=20.0):
    """n consecutive frames at t s, luma only, fields deinterleaved (top field rows on top)."""
    raw = _src_run([ffmpeg, "-v", "error", "-nostdin", "-threads", "2", "-ss", "%.3f" % t,
                "-i", path, "-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", str(n),
                "-vf", "scale=%d:%d:flags=neighbor,il=l=d:c=d,format=gray" % (W, H),
                "-f", "rawvideo", "-"], timeout)
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
            rag.append(_src_median(d))
    return _src_median(noise), (_src_median(rag) if len(rag) >= 2 else None)


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
        med = [_src_median(c) for c in acc.colvals[side]]
        edge = _src_median(med[1:4])
        inner = _src_median(med[24:40])
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
              hs_sign=None if hs_sign is None else round(hs_sign, 2), hs_median=_src_median(acc.hs),
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


def check_values(a):
    """Option values that would only fail later (after detection, or on every movie of a batch)."""
    if a.ai_blend is not None and not 0 <= a.ai_blend <= 1:
        sys.exit("--ai-blend must be between 0 and 1")
    if a.smooth is not None and a.smooth < 0 or a.sharpen is not None and not 0 <= a.sharpen <= 2:
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
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--dar", default=None, help="force aspect, e.g. 16:9 or 4:3")
    p.add_argument("--fps", default=None, help="override output fps, e.g. 24000/1001")
    p.add_argument("--chunk-frames", type=int, default=None)
    p.add_argument("--test", type=int, default=0, help="only process first N seconds")
    p.add_argument("--esrgan", default="realesrgan-ncnn-vulkan")
    p.add_argument("--gpu", default=None,
                   help="Vulkan GPU index for the upscaler (-g). Several, e.g. 0,1 (a laptop's "
                        "NVIDIA plus the processor's built-in graphics): the first works through "
                        "the movie, the others upscale whole chunks alongside it")
    p.add_argument("--tile", default=None, help="tile size if GPU runs out of memory (-t)")
    p.add_argument("--gpu-threads", type=int, default=None,
                   help="frames the GPU upscales at once with the anime and VHS models "
                        "(default 4; the bigger live-action/CGI model always does 2): more can "
                        "keep a GPU busier. The picture is the same either way")
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

STOP / RESUME
  Ctrl+C                       stop (finished chunks are kept)
  run the same command again   continue where it stopped (with the same type as before)
  shutdown /a                  cancel a --shutdown countdown

USEFUL EXTRAS (add to any command above)
  --hevc             smaller files, but needs a newer TV/player (H.264 is the default)
  --height 720       720p instead of 1080p
  --dar 16:9         fix a squeezed/stretched picture (or --dar 4:3)
  --ai-blend 0.5     gentler AI (less "painted" look; default 0.75 live, 1 anime)
  --fast             no AI: much quicker, ordinary resize
  --cpu              encode without an NVIDIA GPU (slow)
  --tile 128         if the GPU runs out of memory
  --gpu-threads 6    frames the GPU works on at once (default 4): try 6 or 8 for speed; same picture

MORE
  python dvd_upscale.py --help          every option, briefly
  python dvd_upscale.py --all --help    folder-mode options
  queue_log.txt                         what happened in --all / --queue runs
"""


def main():
    a = build_parser().parse_args()
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
    resolve_type(a, find)
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
    if not 0 <= a.ai_blend <= 1:
        sys.exit("--ai-blend must be between 0 and 1")
    if a.smooth < 0 or not 0 <= a.sharpen <= 2:
        sys.exit("--smooth must be 0 or more, --sharpen between 0 and 2")
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
        a.dar = a.dar.replace(":", "/")
        if frac(a.dar) <= 0:
            sys.exit(f"Invalid --dar '{a.dar}', use e.g. 16:9 or 4:3")
    if a.fps:
        if frac(a.fps) <= 0:
            sys.exit(f"Invalid --fps '{a.fps}', use e.g. 24000/1001 or 25")
    if a.height < 2 or a.height % 2:
        sys.exit("--height must be an even number")
    if a.chunk_frames < 1:
        sys.exit("--chunk-frames must be at least 1")
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
        try:
            Path(a.output).resolve().parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            sys.exit(f"Can't create the output folder '{Path(a.output).parent}': {e}")
        print(f"Saving to: {a.output}")

    tools = ["ffmpeg", "ffprobe"] + ([] if a.fast or a.analyze else [a.esrgan])
    for t in tools:
        if not find(t):
            sys.exit(f"Missing tool: {t}")
    if not a.fast and not a.analyze:
        a.esrgan_path = find(a.esrgan)
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

    if not a.fast:
        kind = (a.type if a.type != "vhs" else
                f"vhs {'movie tape' if a.mode == 'telecine' else 'camcorder tape'}")
        print(f"Upscaler: {a.model} x{a.scale} ({kind} preset)")
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
    if a.analyze:
        report = os.environ.get("DVD_UPSCALE_REPORT")
        if report:
            Path(report).write_text(json.dumps(dict(type=a.type, mode=a.mode,
                                                    combed=bool(getattr(a, "combed", False)))))
        return

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

    work = Path(a.work + ("_test" if a.test else ""))
    work.mkdir(parents=True, exist_ok=True)
    a.lock = lock_work(work)
    keep_awake()
    st = Path(a.input).stat()
    # encoder args are included so a resumed run never mixes chunks from different encoder
    # settings (their stream headers differ, and joining them breaks playback)
    fp = dict(input=str(Path(a.input).resolve()), size=st.st_size, mtime=int(st.st_mtime),
              mode=a.mode, fps=a.fps, model=a.model, scale=a.scale, height=a.height,
              chunk=a.chunk_frames, fast=a.fast, dar=a.dar, test=a.test, enc=" ".join(a.enc), w=a.out_w,
              denoise=a.denoise, ai_blend=a.ai_blend, smooth=a.smooth, sharpen=a.sharpen,
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
        changed = [k for k in fp if old.get(k) != fp[k]]

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
        sys.exit(f"Settings or input changed since the last run in '{work}': "
                 f"{', '.join(changed)}{hint}. "
                 + (f"Delete the folder '{work.resolve()}' to start this movie fresh." if queue
                    else "Delete that folder (or use --work NEWNAME) to start fresh."))
    sf.write_text(json.dumps(fp))
    # what it was detected as: a resumed run keeps it, and the --all/--queue log shows it
    if a.detected or not (work / "detected.json").exists():
        (work / "detected.json").write_text(json.dumps(dict(type=a.type, **a.detected)))

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
    # MPEG program/transport streams (capture cards, DVD recorders, .vob): they have no index, so
    # -ss can land frames off (measured on 100 s MPEG-2 captures: up to 15 frames late with
    # ffmpeg 6.1, up to 12 with a 2026 build), repeating frames at chunk starts and moving the
    # picture against the sound. Chunks are cut from a stream copy of the video in .mkv instead,
    # which seeks exactly; its timeline starts at the source's first video frame.
    a.chunk_input, cut = a.input, vo
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
            os.replace(tmp, idx)
        a.chunk_input, cut = str(idx), 0.0
    try:            # set by --all / --queue: which movie this is, and how much video comes after
        queue = json.loads(os.environ.get("DVD_UPSCALE_QUEUE") or "null")
        later = float(queue["later_secs"]) if queue else 0.0
    except (ValueError, TypeError, KeyError):
        queue, later = None, 0.0
    done_before = sum((work / f"chunk_{i:05d}.mkv").exists() for i, *_ in plan)
    if done_before:
        print(f"Resuming: {done_before} of {len(plan)} chunks already done")
    chunks, notes = {}, []
    # three steps per chunk, overlapped: while chunk i is upscaled on the GPU, chunk i+1's frames
    # are read and chunk i-1 is encoded in the background, so the GPU doesn't wait for either.
    # With --gpu 0,1 the other GPUs each take whole chunks alongside (see helper below). A
    # failure on the main GPU stops the movie as before
    todo = [entry for entry in plan if not (work / f"chunk_{entry[0]:05d}.mkv").exists()]
    for entry in plan:
        if entry not in todo:
            chunks[entry[0]] = work / f"chunk_{entry[0]:05d}.mkv"     # done in an earlier run
    waiting = list(todo)                    # the chunks no GPU has taken yet
    lock, make_lock = threading.Lock(), threading.Lock()
    devices = gpu_list(a)
    a.stop_lanes = threading.Event()
    a.cancel_lanes = set()                  # helpers whose chunk the main GPU took back
    main_secs, skipped, shown = [], set(), [0]  # main GPU's upscale times; chunks with no video
    progress = {"chunks": 0, "video": 0.0, "helped": 0}
    t_start = time.time()
    encoding = reading = current = None     # (Background, Chunk) / Chunk

    def claim(keep=0):
        with lock:
            return waiting.pop(0) if len(waiting) > keep else None

    def give_back(entry):
        with lock:
            waiting.insert(0, entry)

    def make(entry, gpu=devices[0], lane=None):
        i, t, length, expected = entry
        # (an AVI with B-frames: after the first chunk, which is read without -ss, seek by
        # ffmpeg's late picture times, else each later chunk started 1-2 frames early)
        seek = max(0.0, cut + float(t) + (info["seek_delay"] if t else 0.0) - eps)
        warm = i > 0 and (a.type == "vhs" and t * a.vhs_fin >= a.vhs_warm[0] + 1
                          or a.type != "vhs" and a.mode == "telecine" and not a.pal and t >= 1)
        with make_lock:             # (it sets this chunk's warm-up trim on a, then reads it)
            job = Chunk(a, i, f"{seek:.6f}", f"{float(length):.6f}", expected, work,
                        f"chunk {i + 1}/{len(plan)}", last=(i == plan[-1][0]), warm=warm)
        job.gpu, job.lane = gpu, lane
        job.make_room = None if lane else stop_helpers
        return job

    def counted(job, helped=False):
        with lock:
            progress["chunks"] += 1
            progress["video"] += job.length
            progress["helped"] += helped

    def helper(dev):
        """Another GPU: whole chunks, from reading the frames to the chunk file, alongside the
        main one. The last few chunks are left to the main GPU (faster: nothing waits for a
        slow GPU at the end). Whatever goes wrong, the chunk goes back to the main GPU (its
        frames deleted first) and this GPU stops helping."""
        lane = f"GPU {dev}"
        while not a.stop_lanes.is_set():
            entry = claim(keep=3)
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
                                 "the main GPU does the rest")
                return
            finally:
                LANE_STATUS.pop(lane, None)
            with lock:
                chunks[job.idx] = path
            counted(job, helped=True)
            for msg in (warning, warning2):     # (only now: a chunk handed back warns again)
                if msg:
                    notes.append(msg)

    def stop_helpers():
        """The main GPU's chunk doesn't fit on the drive next to the other GPUs' frames: they
        stop (their chunks come back to the main GPU, their frames are deleted) rather than the
        movie failing. True if any were still running."""
        busy = [h for h in helpers if h.thread.is_alive()]
        if not busy:
            return False
        a.stop_lanes.set()
        for h in busy:
            try:
                h.wait()
            except Exception:
                pass
        notes.append("  NOTE: the drive with the work folder is too full for two chunks at "
                     "once: the other GPU stopped helping, the main GPU does the rest")
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
                notes.append(f"  NOTE: the main GPU takes {lane}'s chunk back ("
                             + ("it stopped making progress)" if stuck else
                                "it will be done with it sooner)"))

    def done_encoding():
        nonlocal encoding
        if encoding:
            bg, job = encoding
            encoding = None
            chunks[job.idx], warning = bg.wait()
            if warning:
                status_line()
                print(warning, flush=True)

    def show_notes():
        while notes:
            status_line()
            print(notes.pop(0), flush=True)

    others = [] if a.fast else [d for d in devices[1:] if d]
    old_ctrl_c = signal.getsignal(signal.SIGINT)
    if others and old_ctrl_c is signal.default_int_handler \
            and threading.current_thread() is threading.main_thread():
        def ctrl_c(sig, frame):
            a.stop_lanes.set()      # the other GPUs stop now, before they start anything new
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT, ctrl_c)
    helpers = [Background(lambda d=d: helper(d)) for d in others]
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
            done_encoding()                     # the previous chunk's encode
            encoding, current = (Background(job.finish), job), None
            counted(job)
            show_notes()
            elapsed = time.time() - t_start
            per_chunk = elapsed / progress["chunks"]
            remaining = len(todo) - progress["chunks"]
            left = remaining * per_chunk
            movie = f" - movie {queue['n']} of {queue['of']} - this movie:" if queue else " -"
            helped = (f" ({progress['helped']} by the other GPU)"
                      if helpers and progress["helped"] else "")
            shown[0] = len(plan) - remaining
            say(f"[{len(plan) - remaining}/{len(plan)}] {per_chunk:.0f}s/chunk{helped}"
                + (f"{movie} {eta_text(left)}" if remaining else
                   ", all chunks done, finishing the file..."))
            if later > 0 and remaining:
                # the movies still to come, at this movie's speed (seconds of work per second
                # of video): rough, a live-action movie takes longer than an anime one
                rest = left + later * elapsed / progress["video"]
                say(f"        all {queue['of']} movies: {eta_text(rest)} (rough)")
        if encoding:
            status_line(f"  {encoding[1].label}: encoding")
        done_encoding()
        status_line()
        show_notes()
        if helpers and progress["chunks"] and shown[0] != len(plan):
            # (another GPU finished the last chunk to be counted)
            say(f"[{len(plan)}/{len(plan)}] ({progress['helped']} by the other GPU), all "
                "chunks done, finishing the file...")
    except BaseException:
        # stopped (Ctrl+C) or failed: let the steps already running in the background end
        # first, so nothing is left writing on its own (a finished encode keeps its chunk)
        a.stop_lanes.set()
        for bg in [x[0] for x in (encoding, reading) if x] + helpers:
            try:
                bg.wait()
            except BaseException:
                pass
        for job in (current, reading and reading[1]):
            if job:
                job.clear_frames()              # made again next time
        raise
    finally:
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
         "-of", "json", str(a.input)]))["streams"]
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
        os.replace(part, out)
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
         "-of", "csv=p=0", str(joined)], encoding="utf-8", errors="replace").strip())
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


# ---------------------------------------------------------------------------------------------
# Queue mode: python dvd_upscale.py --queue [queue.txt] [--shutdown]
# Each movie runs as its own run of this script, so one movie crashing can't take the queue
# down, and Ctrl+C, the work-folder lock and resuming behave exactly as for a single movie.

VALUE_OPTS = ("--type", "--mode", "--model", "--scale", "--height", "--dar", "--fps",
              "--chunk-frames", "--test", "--esrgan", "--gpu", "--tile", "--gpu-threads",
              "--ai-blend", "--smooth",
              "--sharpen", "--work", "--chroma-delay", "--mask")
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


def media_duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, encoding="utf-8", errors="replace")
    try:
        return float(r.stdout.strip().split(",")[0])
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
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json",
                        str(path)], capture_output=True, encoding="utf-8", errors="replace")
    try:
        tags = json.loads(r.stdout).get("format", {}).get("tags", {})
    except ValueError:
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
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags", "-of", "json",
                        str(path)], capture_output=True, encoding="utf-8", errors="replace")
    try:
        tags = json.loads(r.stdout).get("format", {}).get("tags", {})
    except ValueError:
        return False
    return any(OUTPUT_TAG in str(v) for v in tags.values())
TYPE_DIRS = {"live": "live", "live action": "live", "live-action": "live", "anime": "anime",
             "cgi": "cgi", "3d": "cgi", "vhs": "vhs"}


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
            tape = (kind or passed.type) == "vhs"
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
            if kind:
                args += ["--type", kind]
            args += ["--work", str(rel.with_name(stem + "_work"))]
            jobs.append((len(jobs) + 1, str(rel), args, None, False,
                         str(rel.with_name(f"{stem} {height}p.mkv"))))
    return jobs, waiting


def video_height(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=height", "-of", "csv=p=0", str(path)],
                       capture_output=True, encoding="utf-8", errors="replace")
    m = re.match(r"\s*(\d+)", r.stdout)
    return int(m.group(1)) if m else None


def video_length(path):
    """Length of the video stream itself (not the whole file, which counts the audio too)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=duration:stream_tags", "-of", "json", str(path)],
                       capture_output=True, encoding="utf-8", errors="replace")
    try:
        st = json.loads(r.stdout)["streams"][0]
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
             "-of", "json", str(path)]))
        return [s.get("codec_type") for s in j["streams"]], j["streams"]
    try:
        (t_src, s_src), (t_out, _) = tracks(src), tracks(out)
    except (subprocess.CalledProcessError, ValueError, KeyError):
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
    except (subprocess.CalledProcessError, ValueError, KeyError, ZeroDivisionError):
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
                        r"video_(joined|audio)\.mkv|video_index(\.part)?\.mkv|settings\.json|"
                        r"detected\.json|\.lock")


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
        # the folder given anywhere ("--all --shutdown D:\Movies"), not only right after --all
        dirs = [w for w in extra if not w.startswith("-") and Path(w).is_dir()]
        if a.all == "." and len(dirs) == 1 and \
                not any(extra[k - 1] in VALUE_OPTS for k, w in enumerate(extra) if w == dirs[0]
                        and k > 0):
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
    keep_awake()

    def note(msg):
        print(msg, flush=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")

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
            note(f"Original deleted (to the Recycle Bin, if its drive has one): {args[0]}"
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
        try:
            env = dict(os.environ, DVD_UPSCALE_QUEUE=json.dumps(
                dict(n=pos, of=len(valid), later_secs=later_secs, folder=folder_mode)))
            rc = subprocess.run([sys.executable, str(script), *args], cwd=base,
                                env=env).returncode
        except KeyboardInterrupt:
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
        note("Shutting down in 60 s (cancel with: shutdown /a)")
        cmd = (["shutdown", "/s", "/t", "60"] if sys.platform == "win32"
               else ["shutdown", "-h", "+1"])
        subprocess.run(cmd)


def hms_text(secs):
    m, s = divmod(int(secs), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}"


def has_dvd_pcm(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                        "stream=codec_name", "-of", "csv=p=0", str(path)],
                       capture_output=True, stdin=subprocess.DEVNULL)
    return any(c.strip() in ("pcm_dvd", "pcm_bluray")
               for c in r.stdout.decode("utf-8", "replace").splitlines())


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
    out = src.with_name(f"{src.stem} clip {tag}.mkv")
    dur = media_duration(src)
    if dur and start >= dur:
        sys.exit(f"The video is only {hms_text(dur)} long: pick an earlier start.")
    rc = subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{start:.3f}",
                         "-i", str(src), "-t", f"{length:.3f}", "-map", "0:v:0",
                         *([] if a.video_only else ["-map", "0:a?"]),
                         # DVD/recorder PCM can't go in .mkv as it is: lossless FLAC instead
                         "-c", "copy", *([] if a.video_only else ["-c:a", "flac"]
                                         if has_dvd_pcm(src) else []),
                         "-metadata:s", "DURATION-eng=", str(out)],
                        stdin=subprocess.DEVNULL).returncode
    if rc or not out.exists() or not (media_duration(out) or 0) > 0:
        sys.exit("Couldn't cut the clip (see the ffmpeg error above).")
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


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):      # a name the console/log can't show: '?', not a crash
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    # ffmpeg/ffprobe/the upscaler next to this script work even when the movies are elsewhere
    # (every ffmpeg call uses the bare name, and --all/--queue run each movie in its folder)
    _here = str(Path(__file__).resolve().parent)
    if not shutil.which("ffmpeg") and shutil.which("ffmpeg", path=_here):
        os.environ["PATH"] = _here + os.pathsep + os.environ.get("PATH", "")
    if sys.argv[1:2] == ["--clip"]:
        try:
            clip_main(sys.argv[2:])
        except KeyboardInterrupt:
            sys.exit(130)
        sys.exit(0)
    if len(sys.argv) == 1 or sys.argv[1:] in (["--commands"], ["commands"]):
        print(COMMANDS)             # plain "python dvd_upscale.py" shows the cheat sheet too
        sys.exit(0)
    queue_mode = any(w in ("--queue", "--all") or w.startswith(("--queue=", "--all="))
                     for w in sys.argv[1:])
    try:
        if queue_mode:
            queue_main(sys.argv[1:])
        else:
            main()
    except KeyboardInterrupt:
        if not queue_mode:
            print("\nStopped. Re-run the same command to resume.", file=sys.stderr)
        sys.exit(130)       # distinct code: the queue stops instead of moving on
    except (subprocess.CalledProcessError, RuntimeError) as e:
        sys.exit(f"\nFailed: {e}\nRe-run the same command to retry from the last finished chunk.")
    except OSError as e:                         # disk full, file in use, missing folder...
        sys.exit(f"\nFailed: {e}\nFix that and re-run the same command: finished chunks are kept.")
