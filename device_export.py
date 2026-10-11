#!/usr/bin/env python3
"""device_export.py - copy photos/data off an UNLOCKED, trusted phone.

This uses the official, first-party device protocols. It can only talk to a
phone that is:
  * powered on and UNLOCKED, and
  * has approved this computer ('Trust This Computer' on iPhone, or the USB
    'Allow' / file-transfer prompt on Android).
That approval is the on-device consent step - a locked phone cannot give it,
so this tool cannot and does not bypass any lock. It is for getting YOUR data
off once you're back in.

Backends (install the one you need):
  iPhone  -> pymobiledevice3   (pip install pymobiledevice3)
  Android -> adb / platform-tools (Android SDK platform-tools)

Examples:
    python device_export.py --check
    python device_export.py --ios --photos --out ./iphone_photos
    python device_export.py --ios --backup --out ./iphone_backup
    python device_export.py --android --photos --out ./android_photos
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

IOS_TOOL = "pymobiledevice3"
ANDROID_TOOL = "adb"


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """Run a command, capturing output; never raise on non-zero exit."""
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def tool_available(name: str) -> bool:
    return shutil.which(name) is not None


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------
def check_environment() -> None:
    print("Backend tools")
    print("-" * 13)
    for label, tool, install in [
        ("iPhone  (pymobiledevice3)", IOS_TOOL, "pip install pymobiledevice3"),
        ("Android (adb)", ANDROID_TOOL, "install Android SDK platform-tools"),
    ]:
        have = tool_available(tool)
        mark = "found" if have else "MISSING"
        print(f"  {label:<28}: {mark}")
        if not have:
            print(f"      -> install with: {install}")

    print("\nVisible devices")
    print("-" * 15)
    if tool_available(IOS_TOOL):
        _show_ios_devices()
    if tool_available(ANDROID_TOOL):
        _show_android_devices()
    if not tool_available(IOS_TOOL) and not tool_available(ANDROID_TOOL):
        print("  (no backend tools installed yet)")


def _show_ios_devices() -> None:
    res = _run([IOS_TOOL, "usbmux", "list"], timeout=30)
    if res.returncode == 0 and res.stdout.strip():
        print("  iPhone: device(s) detected (see pymobiledevice3 output).")
    else:
        print("  iPhone: none detected (plug in, unlock, tap 'Trust').")


def _show_android_devices() -> None:
    res = _run([ANDROID_TOOL, "devices"], timeout=30)
    lines = [l for l in res.stdout.splitlines()[1:] if l.strip()]
    if not lines:
        print("  Android: none detected (plug in, unlock, allow file transfer).")
        return
    for line in lines:
        serial, _, state = line.partition("\t")
        state = state.strip()
        if state == "unauthorized":
            print(f"  Android {serial}: UNAUTHORIZED -> unlock phone and tap 'Allow'.")
        elif state == "device":
            print(f"  Android {serial}: ready.")
        else:
            print(f"  Android {serial}: {state}")


# ---------------------------------------------------------------------------
# iPhone export
# ---------------------------------------------------------------------------
def ios_photos(out: Path) -> int:
    """Pull the camera roll (DCIM) over AFC. Requires a trusted, unlocked phone."""
    if not tool_available(IOS_TOOL):
        print(f"ERROR: {IOS_TOOL} not installed (pip install pymobiledevice3).")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    print(f"Pulling iPhone DCIM -> {out}")
    print("  (Phone must be unlocked and 'trusted' for this to work.)")
    res = _run([IOS_TOOL, "afc", "pull", "DCIM", str(out)], timeout=3600)
    sys.stdout.write(res.stdout)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        print("\n  If this failed: unlock the phone, re-tap 'Trust', and retry.")
    return res.returncode


def ios_backup(out: Path) -> int:
    """Full encrypted-capable backup via the official backup service."""
    if not tool_available(IOS_TOOL):
        print(f"ERROR: {IOS_TOOL} not installed (pip install pymobiledevice3).")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    print(f"Creating full iPhone backup -> {out}")
    print("  (Phone must be unlocked and 'trusted'.)")
    res = _run([IOS_TOOL, "backup2", "backup", "--full", str(out)], timeout=7200)
    sys.stdout.write(res.stdout)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
    return res.returncode


# ---------------------------------------------------------------------------
# Android export
# ---------------------------------------------------------------------------
def android_photos(out: Path) -> int:
    """Pull /sdcard/DCIM over adb. Requires an unlocked, authorized phone."""
    if not tool_available(ANDROID_TOOL):
        print(f"ERROR: {ANDROID_TOOL} not installed (Android platform-tools).")
        return 2

    state = _run([ANDROID_TOOL, "devices"], timeout=30)
    rows = [l for l in state.stdout.splitlines()[1:] if l.strip()]
    if not rows:
        print("No Android device detected. Plug in, unlock, allow file transfer.")
        return 1
    if all("unauthorized" in r for r in rows):
        print("Device is UNAUTHORIZED. Unlock the phone and tap 'Allow' on the")
        print("USB debugging prompt, then run this again.")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    print(f"Pulling Android /sdcard/DCIM -> {out}")
    res = _run([ANDROID_TOOL, "pull", "/sdcard/DCIM", str(out)], timeout=3600)
    sys.stdout.write(res.stdout)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
    return res.returncode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true",
                   help="report installed backends and visible devices")
    p.add_argument("--ios", action="store_true", help="target an iPhone")
    p.add_argument("--android", action="store_true", help="target an Android phone")
    p.add_argument("--photos", action="store_true", help="pull the camera roll (DCIM)")
    p.add_argument("--backup", action="store_true", help="full iPhone backup (iOS only)")
    p.add_argument("--out", type=Path, default=Path("./export"),
                   help="destination folder (default: ./export)")
    args = p.parse_args(argv)

    if args.check or not (args.ios or args.android):
        check_environment()
        if not (args.ios or args.android):
            print("\nNothing to export (pass --ios or --android with --photos/--backup).")
        return 0

    if args.ios and args.backup:
        return ios_backup(args.out)
    if args.ios and args.photos:
        return ios_photos(args.out)
    if args.android and args.photos:
        return android_photos(args.out)
    if args.android and args.backup:
        print("Full-image backup over adb varies by device; use --photos, or")
        print("use Samsung Smart Switch on the desktop for a complete backup.")
        return 2

    p.error("choose what to export: --photos (or --backup for iOS)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
