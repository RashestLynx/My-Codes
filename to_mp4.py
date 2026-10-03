#!/usr/bin/env python3
"""Safely convert video files (MKV, AVI, MOV, WebM, WMV, ...) to MP4 using ffmpeg.

Safety guarantees:
  * The input file is never modified or deleted.
  * Existing output files are not overwritten unless --overwrite is given.
  * Output is written to a temporary file first and only moved into place
    after ffmpeg succeeds and the result passes a sanity check, so a crash
    or Ctrl+C never leaves a half-written .mp4 behind.
  * Streams already compatible with MP4 are copied losslessly (fast, no
    quality loss); only incompatible streams are re-encoded.
  * ffmpeg is invoked without a shell, so odd filenames are handled safely.

Usage:
  python3 to_mp4.py movie.mkv
  python3 to_mp4.py clip.avi -o out.mp4
  python3 to_mp4.py videos/                  # converts every video file inside
  python3 to_mp4.py movie.webm --reencode    # H.264/AAC for maximum compatibility

A single file can be anything ffmpeg can read. Folder mode picks up the
extensions in VIDEO_EXTENSIONS and skips files that are already .mp4.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

# File types picked up when converting a whole folder.
VIDEO_EXTENSIONS = {
    ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv", ".m4v", ".mpg", ".mpeg",
    ".ts", ".m2ts", ".mts", ".vob", ".3gp", ".3g2", ".ogv", ".asf", ".divx",
    ".f4v", ".rm", ".rmvb",
}

# Codecs MP4 can hold as-is (stream copy).
MP4_VIDEO_CODECS = {"h264", "hevc", "av1", "mpeg4", "vp9"}
MP4_AUDIO_CODECS = {"aac", "mp3", "ac3", "eac3", "opus", "flac", "alac"}
# Text subtitles can be converted to MP4's mov_text; image subtitles (PGS,
# VobSub) cannot and are skipped.
TEXT_SUBTITLE_CODECS = {"subrip", "ass", "ssa", "webvtt", "mov_text", "text"}

# How much shorter (seconds, or 1% of the length if larger) the output may be
# than the input before it is treated as truncated.
DURATION_TOLERANCE = 2.0


class ConversionError(Exception):
    pass


def require_tools():
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise ConversionError(
                f"'{tool}' was not found on PATH. Install ffmpeg first "
                "(e.g. 'sudo apt install ffmpeg', 'brew install ffmpeg', "
                "or 'winget install ffmpeg')."
            )


def probe(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", ffmpeg_path(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise ConversionError(f"ffprobe could not read '{path}':\n{result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ConversionError(f"ffprobe returned unreadable output for '{path}'")


def ffmpeg_path(path):
    # The "file:" prefix stops ffmpeg from treating names containing ':' or '|'
    # as protocols (e.g. "concat:...").
    return "file:" + str(path)


@lru_cache(maxsize=None)
def has_encoder(name):
    result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    return any(line.split()[1:2] == [name] for line in result.stdout.splitlines())


def duration_of(info):
    try:
        return float(info["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        return None


def build_stream_args(info, force_reencode, crf):
    """Return ffmpeg -map/-c arguments for each stream, plus warnings."""
    args, warnings = [], []
    out_index = {"v": 0, "a": 0, "s": 0}
    needs_strict = False

    for stream in info.get("streams", []):
        idx = stream["index"]
        kind = stream.get("codec_type")
        codec = stream.get("codec_name", "unknown")

        if kind == "video":
            # Skip embedded cover art / thumbnails.
            if stream.get("disposition", {}).get("attached_pic"):
                continue
            n = out_index["v"]
            args += ["-map", f"0:{idx}"]
            if codec in MP4_VIDEO_CODECS and not force_reencode:
                args += [f"-c:v:{n}", "copy"]
                if codec == "hevc":
                    # Makes HEVC playable on Apple devices / QuickTime.
                    args += [f"-tag:v:{n}", "hvc1"]
            else:
                if not has_encoder("libx264"):
                    raise ConversionError(
                        f"video stream {idx} ({codec}) must be re-encoded, but this ffmpeg "
                        "build has no libx264 encoder"
                    )
                # yuv420p needs even width/height; round odd sizes down by one pixel.
                args += [f"-c:v:{n}", "libx264", f"-crf:v:{n}", str(crf),
                         f"-preset:v:{n}", "medium", f"-pix_fmt:v:{n}", "yuv420p",
                         f"-filter:v:{n}", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
                if not force_reencode:
                    warnings.append(f"video stream {idx} ({codec}) will be re-encoded to H.264")
            out_index["v"] += 1

        elif kind == "audio":
            n = out_index["a"]
            args += ["-map", f"0:{idx}"]
            if codec in MP4_AUDIO_CODECS and not force_reencode:
                args += [f"-c:a:{n}", "copy"]
                if codec in ("flac", "opus"):
                    # Marked experimental in MP4 on ffmpeg versions before 6.0.
                    needs_strict = True
            else:
                channels = stream.get("channels") or 2
                bitrate = min(max(channels, 2) * 96, 768)
                args += [f"-c:a:{n}", "aac", f"-b:a:{n}", f"{bitrate}k"]
                if not force_reencode:
                    warnings.append(f"audio stream {idx} ({codec}) will be re-encoded to AAC")
            out_index["a"] += 1

        elif kind == "subtitle":
            if codec in TEXT_SUBTITLE_CODECS:
                n = out_index["s"]
                args += ["-map", f"0:{idx}", f"-c:s:{n}", "mov_text"]
                out_index["s"] += 1
            else:
                warnings.append(
                    f"subtitle stream {idx} ({codec}) cannot be stored in MP4 "
                    "(image-based or unsupported); skipped"
                )

        else:
            # Attachments (fonts), data streams, etc. are not supported by MP4.
            if kind not in (None, "attachment"):
                warnings.append(f"{kind} stream {idx} ({codec}) is not supported in MP4; skipped")

    if out_index["v"] == 0 and out_index["a"] == 0:
        raise ConversionError("no video or audio streams found to convert")
    if needs_strict:
        args += ["-strict", "experimental"]
    return args, warnings


def verify_output(src_info, out_path):
    out_info = probe(out_path)
    if not any(s.get("codec_type") in ("video", "audio") for s in out_info.get("streams", [])):
        raise ConversionError("output contains no audio or video streams")
    src_dur, out_dur = duration_of(src_info), duration_of(out_info)
    tolerance = max(DURATION_TOLERANCE, (src_dur or 0) * 0.01)
    if src_dur and out_dur and src_dur - out_dur > tolerance:
        raise ConversionError(
            f"output ({out_dur:.1f}s) is shorter than the input ({src_dur:.1f}s); "
            "the conversion may be incomplete"
        )


def convert(src, dst, overwrite=False, force_reencode=False, crf=18):
    src, dst = Path(src).resolve(), Path(dst).resolve()

    if not src.is_file():
        raise ConversionError(f"input file not found: {src}")
    # samefile also catches case-insensitive filesystems, symlinks and hard links,
    # where --overwrite would otherwise replace the source.
    if src == dst or (dst.exists() and os.path.samefile(src, dst)):
        raise ConversionError("output path must be different from the input path")
    if dst.is_dir():
        raise ConversionError(f"output path is a directory: {dst}")
    if dst.exists() and not overwrite:
        raise ConversionError(f"output already exists: {dst} (use --overwrite to replace it)")

    info = probe(src)
    stream_args, warnings = build_stream_args(info, force_reencode, crf)
    for w in warnings:
        print(f"  warning: {w}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = None

    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-stats", "-y",
        "-i", ffmpeg_path(src),
        *stream_args,
        "-map_metadata", "0",
        "-map_chapters", "0",
        "-movflags", "+faststart",
        "-f", "mp4",
    ]

    try:
        # Temp file in the same directory so the final rename is atomic.
        # The stem is shortened to stay under filesystem name-length limits.
        fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.stem[:100]}.", suffix=".part.mp4",
                                        dir=dst.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        result = subprocess.run(cmd + [ffmpeg_path(tmp)])
        if result.returncode != 0:
            raise ConversionError(f"ffmpeg failed with exit code {result.returncode}")
        verify_output(info, tmp)
        # mkstemp creates files as 0600; give the result normal permissions.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, dst)
    except BaseException:
        # Covers errors and Ctrl+C: never leave a partial file behind.
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError as e:
                print(f"  warning: could not remove temporary file {tmp}: {e}", file=sys.stderr)
        raise

    return dst


def collect_inputs(path):
    p = Path(path)
    if p.is_dir():
        # Hidden files are skipped, which also covers this script's own temp files.
        return sorted(
            f for f in p.iterdir()
            if f.is_file() and not f.name.startswith(".")
            and f.suffix.lower() in VIDEO_EXTENSIONS
        )
    return [p]


def main():
    parser = argparse.ArgumentParser(description="Safely convert video files to MP4.")
    parser.add_argument("input", help="a video file (any format ffmpeg reads), or a folder of video files")
    parser.add_argument("-o", "--output",
                        help="output file, or output folder (required to be a folder for folder input); "
                             "defaults to the same location with a .mp4 extension")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    parser.add_argument("--reencode", action="store_true",
                        help="re-encode everything to H.264/AAC for maximum compatibility")
    parser.add_argument("--crf", type=int, default=18,
                        help="H.264 quality when re-encoding (0-51, lower is better; default 18)")
    args = parser.parse_args()

    if not 0 <= args.crf <= 51:
        parser.error("--crf must be between 0 and 51")

    try:
        require_tools()
    except ConversionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    inputs = collect_inputs(args.input)
    if not inputs:
        print(f"error: no video files found in {args.input}", file=sys.stderr)
        return 1

    is_batch = Path(args.input).is_dir()
    if is_batch and args.output and Path(args.output).exists() and not Path(args.output).is_dir():
        print(f"error: --output must be a folder when converting a folder: {args.output}",
              file=sys.stderr)
        return 1
    # Files sharing a name ("clip.mkv", "clip.avi") would all become "clip.mp4",
    # so those get the original extension kept: "clip.mkv.mp4", "clip.avi.mp4".
    stem_counts = {}
    for src in inputs:
        stem_counts[src.stem.lower()] = stem_counts.get(src.stem.lower(), 0) + 1

    failures = 0
    planned = set()
    for src in inputs:
        name = (src.name if stem_counts[src.stem.lower()] > 1 else src.stem) + ".mp4"
        if args.output:
            out = Path(args.output)
            dst = out / name if is_batch or out.is_dir() else out
        else:
            dst = src.with_name(name)

        print(f"Converting: {src} -> {dst}")
        # Backstop against two inputs mapping to one output (e.g. "a.mkv.mp4"
        # also existing as an input stem): never let the second replace the first.
        key = os.path.normcase(str(dst.resolve()))
        if key in planned:
            print(f"  error: {dst.name} was already created from another file in this run; skipped",
                  file=sys.stderr)
            failures += 1
            continue
        planned.add(key)
        try:
            convert(src, dst, overwrite=args.overwrite,
                    force_reencode=args.reencode, crf=args.crf)
            print("  done")
        except KeyboardInterrupt:
            print("\nInterrupted; partial output removed.", file=sys.stderr)
            return 130
        except (ConversionError, OSError) as e:
            print(f"  error: {e}", file=sys.stderr)
            failures += 1

    if len(inputs) > 1:
        print(f"\n{len(inputs) - failures}/{len(inputs)} converted successfully")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
