#!/usr/bin/env python3
"""Safely convert MKV files to MP4 using ffmpeg.

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
  python3 mkv_to_mp4.py movie.mkv
  python3 mkv_to_mp4.py movie.mkv -o out.mp4
  python3 mkv_to_mp4.py folder_with_mkvs/          # converts every .mkv inside
  python3 mkv_to_mp4.py movie.mkv --reencode       # force full re-encode
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Codecs MP4 can hold as-is (stream copy).
MP4_VIDEO_CODECS = {"h264", "hevc", "av1", "mpeg4", "vp9"}
MP4_AUDIO_CODECS = {"aac", "mp3", "ac3", "eac3", "opus", "flac", "alac"}
# Text subtitles can be converted to MP4's mov_text; image subtitles (PGS,
# VobSub) cannot and are skipped.
TEXT_SUBTITLE_CODECS = {"subrip", "ass", "ssa", "webvtt", "mov_text", "text"}

# Allowed difference (seconds) between input and output duration.
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
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ConversionError(f"ffprobe could not read '{path}':\n{result.stderr.strip()}")
    return json.loads(result.stdout)


def duration_of(info):
    try:
        return float(info["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        return None


def build_stream_args(info, force_reencode, crf):
    """Return ffmpeg -map/-c arguments for each stream, plus warnings."""
    args, warnings = [], []
    out_index = {"v": 0, "a": 0, "s": 0}

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
                args += [f"-c:v:{n}", "libx264", f"-crf:v:{n}", str(crf),
                         f"-preset:v:{n}", "medium", f"-pix_fmt:v:{n}", "yuv420p"]
                if not force_reencode:
                    warnings.append(f"video stream {idx} ({codec}) will be re-encoded to H.264")
            out_index["v"] += 1

        elif kind == "audio":
            n = out_index["a"]
            args += ["-map", f"0:{idx}"]
            if codec in MP4_AUDIO_CODECS and not force_reencode:
                args += [f"-c:a:{n}", "copy"]
            else:
                args += [f"-c:a:{n}", "aac", f"-b:a:{n}", "192k"]
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
                    f"subtitle stream {idx} ({codec}) is image-based and cannot be stored in MP4; skipped"
                )

        else:
            # Attachments (fonts), data streams, etc. are not supported by MP4.
            if kind not in (None, "attachment"):
                warnings.append(f"{kind} stream {idx} ({codec}) is not supported in MP4; skipped")

    if out_index["v"] == 0 and out_index["a"] == 0:
        raise ConversionError("no video or audio streams found to convert")
    return args, warnings


def verify_output(src_info, out_path):
    out_info = probe(out_path)
    if not any(s.get("codec_type") in ("video", "audio") for s in out_info.get("streams", [])):
        raise ConversionError("output contains no audio or video streams")
    src_dur, out_dur = duration_of(src_info), duration_of(out_info)
    if src_dur and out_dur and abs(src_dur - out_dur) > DURATION_TOLERANCE:
        raise ConversionError(
            f"output duration ({out_dur:.1f}s) differs from input ({src_dur:.1f}s); "
            "the conversion may be incomplete"
        )


def convert(src, dst, overwrite=False, force_reencode=False, crf=18):
    src, dst = Path(src).resolve(), Path(dst).resolve()

    if not src.is_file():
        raise ConversionError(f"input file not found: {src}")
    if src == dst:
        raise ConversionError("output path must be different from the input path")
    if dst.exists() and not overwrite:
        raise ConversionError(f"output already exists: {dst} (use --overwrite to replace it)")
    dst.parent.mkdir(parents=True, exist_ok=True)

    info = probe(src)
    stream_args, warnings = build_stream_args(info, force_reencode, crf)
    for w in warnings:
        print(f"  warning: {w}")

    # Temp file in the same directory so the final rename is atomic.
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.stem}.", suffix=".part.mp4", dir=dst.parent)
    os.close(fd)
    tmp = Path(tmp_name)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-i", str(src),
        *stream_args,
        "-map_metadata", "0",
        "-map_chapters", "0",
        "-movflags", "+faststart",
        "-f", "mp4",
        str(tmp),
    ]

    try:
        result = subprocess.run(cmd)
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
        tmp.unlink(missing_ok=True)
        raise

    return dst


def collect_inputs(path):
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".mkv")
    return [p]


def main():
    parser = argparse.ArgumentParser(description="Safely convert MKV files to MP4.")
    parser.add_argument("input", help="an .mkv file, or a folder containing .mkv files")
    parser.add_argument("-o", "--output",
                        help="output file (single input) or output folder (folder input); "
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
        print(f"error: no .mkv files found in {args.input}", file=sys.stderr)
        return 1

    is_batch = Path(args.input).is_dir()
    failures = 0
    for src in inputs:
        if args.output:
            out = Path(args.output)
            dst = out / (src.stem + ".mp4") if is_batch else out
        else:
            dst = src.with_suffix(".mp4")

        print(f"Converting: {src} -> {dst}")
        try:
            convert(src, dst, overwrite=args.overwrite,
                    force_reencode=args.reencode, crf=args.crf)
            print("  done")
        except KeyboardInterrupt:
            print("\nInterrupted; partial output removed.", file=sys.stderr)
            return 130
        except ConversionError as e:
            print(f"  error: {e}", file=sys.stderr)
            failures += 1

    if len(inputs) > 1:
        print(f"\n{len(inputs) - failures}/{len(inputs)} converted successfully")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
