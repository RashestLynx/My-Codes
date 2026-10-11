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
import contextlib
import datetime as _dt
import io
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

IOS_TOOL = "pymobiledevice3"
ANDROID_TOOL = "adb"


def _run(cmd: list[str], timeout: int = 600, cwd: str | None = None
         ) -> subprocess.CompletedProcess:
    """Run a command, capturing output; never raise on non-zero exit."""
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False, cwd=cwd
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
# Device identifiers (serial / IMEI) - needs an unlocked, trusted device
# ---------------------------------------------------------------------------
def ios_info() -> int:
    """Read SerialNumber / IMEI from a trusted, unlocked iPhone via lockdown."""
    if not tool_available(IOS_TOOL):
        print(f"ERROR: {IOS_TOOL} not installed (pip install pymobiledevice3).")
        return 2
    res = _run([IOS_TOOL, "lockdown", "info"], timeout=60)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        print("Could not read device. Unlock the iPhone and tap 'Trust', then retry.")
        return res.returncode
    wanted = {
        "SerialNumber": "Serial number",
        "InternationalMobileEquipmentIdentity": "IMEI",
        "InternationalMobileEquipmentIdentity2": "IMEI 2",
        "DeviceName": "Device name",
        "ProductType": "Model (product type)",
        "ProductVersion": "iOS version",
    }
    print("iPhone identifiers")
    print("-" * 18)
    text = res.stdout
    for key, label in wanted.items():
        value = _grep_value(text, key)
        if value:
            print(f"  {label:<22}: {value}")
    print("  (Also printed on the SIM tray, the box, and the receipt.)")
    return 0


def android_info() -> int:
    """Read serial / IMEI from an unlocked, authorized Android phone via adb."""
    if not tool_available(ANDROID_TOOL):
        print(f"ERROR: {ANDROID_TOOL} not installed (Android platform-tools).")
        return 2
    print("Android identifiers")
    print("-" * 19)
    serial = _run([ANDROID_TOOL, "shell", "getprop", "ro.serialno"], timeout=30)
    if serial.returncode == 0 and serial.stdout.strip():
        print(f"  Serial number         : {serial.stdout.strip()}")
    else:
        print("  Serial number         : unavailable (unlock + authorize first)")
    # IMEI retrieval is restricted on newer Android; try the legacy service call.
    imei = _run(
        [ANDROID_TOOL, "shell", "service", "call", "iphonesubinfo", "1"], timeout=30
    )
    digits = _parse_service_call_imei(imei.stdout) if imei.returncode == 0 else ""
    # Only trust a result that is a valid 15-digit IMEI; never show partial junk.
    if len(digits) == 15 and _luhn_is_valid(digits):
        print(f"  IMEI                  : {digits}")
    else:
        print("  IMEI                  : unavailable via adb on this Android version")
        print("    -> dial *#06#, or see the SIM tray / box / receipt.")
    return 0


def _grep_value(text: str, key: str) -> str:
    """Pull a value for `key` from pymobiledevice3's key/value or plist-ish output."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(key):
            # handles 'Key: value' and 'Key = value' style lines
            for sep in (":", "="):
                if sep in stripped:
                    return stripped.split(sep, 1)[1].strip().strip("\"',")
    return ""


def _luhn_is_valid(number: str) -> bool:
    """Luhn checksum used to validate a candidate IMEI before trusting it."""
    if not number.isdigit():
        return False
    total = 0
    for i, ch in enumerate(number[::-1]):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _parse_service_call_imei(raw: str) -> str:
    """Decode digits from the UTF-16 hex words 'service call iphonesubinfo' prints.

    Each 32-bit word printed as 8 hex digits holds two UTF-16 code units; the
    low 16 bits are the earlier character, so read (low half, high half) and
    interpret each half directly as the code point.
    """
    digits = ""
    for token in raw.split():
        if len(token) == 8 and all(c in "0123456789abcdefABCDEF" for c in token):
            for half in (token[4:8], token[0:4]):  # low half first
                try:
                    ch = chr(int(half, 16))
                except ValueError:
                    continue
                if ch.isdigit():
                    digits += ch
    return digits


# ---------------------------------------------------------------------------
# Hardware / chipset identification
# ---------------------------------------------------------------------------
# Apple product type -> (marketing name, SoC). A partial but useful map.
APPLE_MODELS = {
    "iPhone9,1": ("iPhone 7", "Apple A10 Fusion"),
    "iPhone9,3": ("iPhone 7", "Apple A10 Fusion"),
    "iPhone10,1": ("iPhone 8", "Apple A11 Bionic"),
    "iPhone10,4": ("iPhone 8", "Apple A11 Bionic"),
    "iPhone10,3": ("iPhone X", "Apple A11 Bionic"),
    "iPhone10,6": ("iPhone X", "Apple A11 Bionic"),
    "iPhone11,2": ("iPhone XS", "Apple A12 Bionic"),
    "iPhone11,8": ("iPhone XR", "Apple A12 Bionic"),
    "iPhone12,1": ("iPhone 11", "Apple A13 Bionic"),
    "iPhone12,3": ("iPhone 11 Pro", "Apple A13 Bionic"),
    "iPhone12,8": ("iPhone SE (2nd gen)", "Apple A13 Bionic"),
    "iPhone13,1": ("iPhone 12 mini", "Apple A14 Bionic"),
    "iPhone13,2": ("iPhone 12", "Apple A14 Bionic"),
    "iPhone13,3": ("iPhone 12 Pro", "Apple A14 Bionic"),
    "iPhone14,5": ("iPhone 13", "Apple A15 Bionic"),
    "iPhone14,2": ("iPhone 13 Pro", "Apple A15 Bionic"),
    "iPhone14,6": ("iPhone SE (3rd gen)", "Apple A15 Bionic"),
    "iPhone14,7": ("iPhone 14", "Apple A15 Bionic"),
    "iPhone15,2": ("iPhone 14 Pro", "Apple A16 Bionic"),
    "iPhone15,4": ("iPhone 15", "Apple A16 Bionic"),
    "iPhone16,1": ("iPhone 15 Pro", "Apple A17 Pro"),
}

# Qualcomm platform codename (ro.board.platform) -> Snapdragon marketing name.
QCOM_PLATFORMS = {
    "msm8998": "Snapdragon 835",
    "sdm845": "Snapdragon 845",
    "sdm660": "Snapdragon 660",
    "msmnile": "Snapdragon 855",
    "kona": "Snapdragon 865",
    "lahaina": "Snapdragon 888",
    "taro": "Snapdragon 8 Gen 1",
    "kalama": "Snapdragon 8 Gen 2",
    "pineapple": "Snapdragon 8 Gen 3",
}


def apple_model_lookup(product_type: str) -> tuple[str, str] | None:
    return APPLE_MODELS.get(product_type.strip())


def ios_hardware() -> int:
    """Report iPhone model + Apple SoC from lockdown info."""
    if not tool_available(IOS_TOOL):
        print(f"ERROR: {IOS_TOOL} not installed (pip install pymobiledevice3).")
        return 2
    res = _run([IOS_TOOL, "lockdown", "info"], timeout=60)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        print("Could not read device. Unlock the iPhone and tap 'Trust', then retry.")
        return res.returncode
    product_type = _grep_value(res.stdout, "ProductType")
    arch = _grep_value(res.stdout, "CPUArchitecture")
    hw_model = _grep_value(res.stdout, "HardwareModel")
    print("iPhone hardware")
    print("-" * 15)
    print(f"  Product type   : {product_type or 'unknown'}")
    named = apple_model_lookup(product_type)
    if named:
        print(f"  Model          : {named[0]}")
        print(f"  Chip (SoC)     : {named[1]}")
    else:
        print("  Model/chip     : not in local map (see ProductType above)")
    if arch:
        print(f"  CPU arch       : {arch}")
    if hw_model:
        print(f"  Hardware model : {hw_model}")
    return 0


def android_hardware() -> int:
    """Report Android manufacturer, model, and SoC via adb getprop."""
    if not tool_available(ANDROID_TOOL):
        print(f"ERROR: {ANDROID_TOOL} not installed (Android platform-tools).")
        return 2

    def prop(name: str) -> str:
        r = _run([ANDROID_TOOL, "shell", "getprop", name], timeout=30)
        return r.stdout.strip() if r.returncode == 0 else ""

    manufacturer = prop("ro.product.manufacturer")
    model = prop("ro.product.model")
    platform = prop("ro.board.platform")
    hardware = prop("ro.hardware")
    abi = prop("ro.product.cpu.abi")
    soc_mfr = prop("ro.soc.manufacturer")   # Android 12+
    soc_model = prop("ro.soc.model")        # Android 12+

    print("Android hardware")
    print("-" * 16)
    if not any([manufacturer, model, platform, soc_model]):
        print("  (no data - unlock the phone and authorize this computer)")
        return 1
    if manufacturer or model:
        print(f"  Device         : {manufacturer} {model}".strip())
    # Prefer the explicit SoC fields; fall back to platform codename mapping.
    soc = f"{soc_mfr} {soc_model}".strip()
    if not soc:
        soc = QCOM_PLATFORMS.get(platform, "")
    if soc:
        print(f"  Chip (SoC)     : {soc}")
    if platform:
        label = QCOM_PLATFORMS.get(platform)
        print(f"  Platform code  : {platform}" + (f" ({label})" if label else ""))
    if hardware:
        print(f"  ro.hardware    : {hardware}")
    if abi:
        print(f"  CPU ABI        : {abi}")
    return 0


# ---------------------------------------------------------------------------
# Manufacture-date estimate (legacy Apple serials only)
# ---------------------------------------------------------------------------
# Apple's 2010-2020 serials encoded the build date. Serials since ~2021 are
# randomized, so the date is NOT derivable from them - use checkcoverage.
_APPLE_YEAR_ALPHABET = "CDFGHJKLMNPQRSTVWXYZ"          # 2 chars per year from 2010
_APPLE_WEEK_ALPHABET = "123456789CDFGHJKLMNPQRTVWXY"   # 1..27 within a half-year


def apple_serial_date(serial: str) -> str:
    """Estimate manufacture date from a legacy (11/12-char) Apple serial.

    Returns a human string, or a note that the date can't be derived. This is
    an ESTIMATE and only works for pre-2021 serials; checkcoverage.apple.com
    is authoritative.
    """
    s = serial.strip().upper()
    if len(s) not in (11, 12):
        return "not derivable from this serial (new-format/randomized) - " \
               "use checkcoverage.apple.com"
    year_c = s[3]
    week_c = s[4]
    if year_c not in _APPLE_YEAR_ALPHABET or week_c not in _APPLE_WEEK_ALPHABET:
        return "not a legacy-format serial - use checkcoverage.apple.com"
    pos = _APPLE_YEAR_ALPHABET.index(year_c)
    year = 2010 + pos // 2
    half = pos % 2  # 0 = first half of year, 1 = second half
    week = _APPLE_WEEK_ALPHABET.index(week_c) + 1 + (26 if half else 0)
    week = min(week, 53)
    return (
        f"IF this is a pre-2021 device: ~{year}, around week {week}. "
        "NOTE: serials from ~2021 on are randomized and look identical, so this "
        "number is meaningless for newer phones - confirm at checkcoverage.apple.com."
    )


# ---------------------------------------------------------------------------
# Authenticity / counterfeit heuristics (NOT definitive)
# ---------------------------------------------------------------------------
_DISCLAIMER = (
    "These are heuristics, not proof. Confirm at the authoritative source: "
    "Apple -> checkcoverage.apple.com; Samsung/Android -> the carrier/maker "
    "IMEI check and the Settings 'About phone' screen."
)


def ios_authenticity() -> int:
    """Cross-protocol check: a genuine iPhone speaks Apple's lockdown protocol;
    a fake 'iPhone' that is really Android does not (but answers adb)."""
    print("iPhone authenticity signals")
    print("-" * 27)
    ios_ok = tool_available(IOS_TOOL)
    speaks_lockdown = False
    text = ""
    if ios_ok:
        res = _run([IOS_TOOL, "lockdown", "info"], timeout=60)
        speaks_lockdown = res.returncode == 0 and bool(res.stdout.strip())
        text = res.stdout

    # Does it answer Android's adb instead? Strong red flag for a "fake iPhone".
    answers_adb = False
    if tool_available(ANDROID_TOOL):
        adb = _run([ANDROID_TOOL, "devices"], timeout=30)
        answers_adb = any(
            l.strip() and not l.startswith("List")
            for l in adb.stdout.splitlines()
        )

    if speaks_lockdown:
        print("  [ok ] Responds to Apple's lockdown protocol (genuine iOS trait).")
        device_class = _grep_value(text, "DeviceClass")
        product = _grep_value(text, "ProductName") or _grep_value(text, "ProductType")
        serial = _grep_value(text, "SerialNumber")
        if device_class.lower() == "iphone":
            print(f"  [ok ] Device class reports as iPhone ({product}).")
        if serial:
            print(f"  [ i ] Serial {serial} - verify it at checkcoverage.apple.com.")
            print(f"        Manufacture date estimate: {apple_serial_date(serial)}")
    elif answers_adb:
        print("  [!! ] Does NOT speak Apple's protocol but DOES answer Android adb.")
        print("        A real iPhone never does this -> very likely a FAKE iPhone")
        print("        (an Android phone dressed up to look like iOS).")
    else:
        print("  [ ? ] No response. Ensure it's unlocked + trusted, or pymobiledevice3")
        print("        is installed, then retry. Can't assess authenticity yet.")
    print(f"\n  {_DISCLAIMER}")
    return 0


def android_authenticity() -> int:
    """Consistency checks that commonly expose counterfeit Android/Samsung phones."""
    if not tool_available(ANDROID_TOOL):
        print(f"ERROR: {ANDROID_TOOL} not installed (Android platform-tools).")
        return 2

    def prop(name: str) -> str:
        r = _run([ANDROID_TOOL, "shell", "getprop", name], timeout=30)
        return r.stdout.strip() if r.returncode == 0 else ""

    brand = prop("ro.product.brand")
    manufacturer = prop("ro.product.manufacturer")
    model = prop("ro.product.model")
    fingerprint = prop("ro.build.fingerprint")
    platform = prop("ro.board.platform")
    hardware = prop("ro.hardware")
    abi = prop("ro.product.cpu.abi")
    gms = prop("ro.com.google.gmsversion")

    if not (brand or model):
        print("No data - unlock the phone and authorize this computer, then retry.")
        return 1

    print("Android authenticity signals")
    print("-" * 28)
    print(f"  Claims to be   : {manufacturer} {model} (brand '{brand}')".strip())

    # 1) Emulator / spoof environment.
    if hardware.lower() in {"goldfish", "ranchu"} or prop("ro.kernel.qemu") == "1":
        print("  [!! ] Looks like an EMULATOR, not a physical phone.")

    # 2) Build fingerprint should mention the same brand/model.
    fp = fingerprint.lower()
    if fingerprint:
        if brand and brand.lower() in fp:
            print("  [ok ] Build fingerprint brand matches the claimed brand.")
        else:
            print("  [warn] Build fingerprint brand does NOT match claimed brand:")
            print(f"         {fingerprint}")
    else:
        print("  [warn] No build fingerprint reported (unusual for a genuine ROM).")

    # 3) Samsung/flagship claim but a MediaTek SoC is a classic clone tell.
    is_mediatek = platform.lower().startswith("mt") or "mt" in hardware.lower()[:2]
    claims_premium = any(
        k in model.lower() for k in ("galaxy s", "galaxy note", "galaxy z")
    )
    if claims_premium and is_mediatek:
        print("  [!! ] Claims a Samsung flagship but runs a MediaTek SoC "
              f"(platform '{platform}') - a very common counterfeit pattern.")
    elif platform:
        label = QCOM_PLATFORMS.get(platform)
        print(f"  [ i ] SoC platform: {platform}" + (f" ({label})" if label else ""))

    # 4) Google Play certification hint.
    if gms:
        print(f"  [ok ] Google Mobile Services present (gmsversion {gms}).")
    else:
        print("  [warn] No Google Mobile Services version - check Play Protect")
        print("         certification in the Play Store (Settings -> About).")

    # 5) 64-bit ABI is expected on any genuine modern phone.
    if abi and "arm64" not in abi and "x86_64" not in abi:
        print(f"  [warn] CPU ABI is '{abi}', not 64-bit - unusual for a modern phone.")

    print(f"\n  {_DISCLAIMER}")
    return 0


# ---------------------------------------------------------------------------
# This computer (laptop/desktop) - reads the machine the script runs on.
# No phone protocol needed; the OS reports its own specs and health.
# ---------------------------------------------------------------------------
def _linux_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _total_ram_bytes() -> int | None:
    # Works on Linux and macOS via sysconf; None elsewhere.
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, AttributeError, OSError):
        return None


def computer_specs() -> int:
    uname = platform.uname()
    print("Computer specifications")
    print("-" * 23)
    print(f"  Hostname       : {uname.node}")
    print(f"  OS             : {uname.system} {uname.release}")
    print(f"  OS version     : {uname.version}")
    print(f"  Architecture   : {uname.machine}")
    print(f"  Processor      : {_linux_cpu_model()}")
    print(f"  CPU cores      : {os.cpu_count()}")
    ram = _total_ram_bytes()
    if ram:
        print(f"  RAM            : {ram / 1024**3:.1f} GB")
    try:
        du = shutil.disk_usage("/")
        print(f"  Disk (/)       : {du.total / 1024**3:.0f} GB total, "
              f"{du.free / 1024**3:.0f} GB free")
    except OSError:
        pass
    print(f"  Python         : {platform.python_version()}")
    return 0


def computer_diagnose() -> int:
    print("Computer diagnostics")
    print("-" * 20)
    found = False

    # Disk space.
    try:
        du = shutil.disk_usage("/")
        used_pct = du.used / du.total * 100
        print(f"  Disk used      : {used_pct:.0f}% of / "
              f"({du.free / 1024**3:.0f} GB free)")
        if used_pct >= 90:
            found = True
            _emit_fix("storage_full")
    except OSError:
        pass

    # CPU load (Unix).
    try:
        load1, _, _ = os.getloadavg()
        cores = os.cpu_count() or 1
        print(f"  Load (1 min)   : {load1:.2f} over {cores} cores")
        if load1 > cores * 2:
            found = True
            print("      High sustained CPU load.")
            print("      Fix: find heavy processes (Task Manager / Activity")
            print("           Monitor / top), close or update them; reboot.")
    except (OSError, AttributeError):
        pass

    # Battery (Linux sysfs; other OSes need vendor tools).
    bat = Path("/sys/class/power_supply/BAT0")
    if bat.exists():
        try:
            cap = (bat / "capacity").read_text().strip()
            status = (bat / "status").read_text().strip()
            print(f"  Battery        : {cap}% ({status})")
        except OSError:
            pass
    else:
        print("  Battery        : not readable via stdlib on this OS")
        print("      (Windows: 'powercfg /batteryreport'; macOS: System")
        print("       Information -> Power; for health use the vendor tool.)")

    if not found:
        print("\n  No disk/load faults detected in readable data.")
    print("\n  Physical/vendor diagnostics:")
    print("      Windows: 'dxdiag', vendor support assistant (Dell/HP/Lenovo).")
    print("      macOS  : Apple Diagnostics (hold D / power on startup).")
    return 0


# ---------------------------------------------------------------------------
# Diagnostics: readable faults (battery / storage / thermal / crashes) + fixes
# ---------------------------------------------------------------------------
# Suggested fixes keyed by detected condition. These are general guidance.
FIXES = {
    "battery_worn": [
        "Battery is worn (high cycle count / reduced health).",
        "Fix: schedule a battery replacement (Apple/Samsung service or an",
        "     authorized shop). Meanwhile, enable battery-saver and avoid heat.",
    ],
    "battery_unhealthy": [
        "Battery health reports a fault (not 'good').",
        "Fix: stop charging if it reads OVERHEAT/OVER_VOLTAGE; have the battery",
        "     checked/replaced by a technician before further use.",
    ],
    "overheating": [
        "Device temperature is high.",
        "Fix: remove the case, stop charging, close heavy apps, let it cool.",
        "     Persistent heat with light use suggests a battery/board fault.",
    ],
    "storage_full": [
        "Storage is nearly full - this causes crashes, slowness, update fails.",
        "Fix: delete large videos/apps, clear caches, offload photos to cloud,",
        "     then reboot. Aim to keep >10% free.",
    ],
    "low_memory": [
        "Very little free RAM - apps may be getting killed.",
        "Fix: reboot, close background apps, remove memory-heavy apps.",
    ],
    "app_crashes": [
        "Recent app/system crashes were found in the crash log.",
        "Fix: update or reinstall the crashing app(s); if system processes",
        "     crash, install pending OS updates or back up and factory reset.",
    ],
    "hardware_note": [
        "Software cannot see physical faults (screen, cameras, speakers, mic,",
        "buttons, water damage). For those, run the maker's built-in hardware",
        "test and/or visit authorized service:",
        "  Samsung: dial *#0*# for the hardware test menu (screen, sensors...).",
        "  iPhone: Apple Support app / Genius Bar diagnostics.",
    ],
}


def _emit_fix(key: str) -> None:
    for line in FIXES[key]:
        print(f"      {line}")


def android_diagnose() -> int:
    """Read battery, thermal, storage, memory and recent crashes; suggest fixes."""
    if not tool_available(ANDROID_TOOL):
        print(f"ERROR: {ANDROID_TOOL} not installed (Android platform-tools).")
        return 2

    def shell(args: list[str]) -> str:
        r = _run([ANDROID_TOOL, "shell", *args], timeout=30)
        return r.stdout if r.returncode == 0 else ""

    print("Android diagnostics")
    print("-" * 19)
    found = False

    # Battery health + temperature.
    batt = shell(["dumpsys", "battery"])
    health_map = {
        "1": "unknown", "2": "good", "3": "overheat", "4": "dead",
        "5": "over-voltage", "6": "unspecified failure", "7": "cold",
    }
    health_val = temp_c = level = None
    for line in batt.splitlines():
        s = line.strip()
        if s.startswith("health:"):
            health_val = s.split(":", 1)[1].strip()
        elif s.startswith("temperature:"):
            try:
                temp_c = int(s.split(":", 1)[1]) / 10.0
            except ValueError:
                pass
        elif s.startswith("level:"):
            level = s.split(":", 1)[1].strip()
    if health_val:
        name = health_map.get(health_val, health_val)
        print(f"  Battery health : {name}  (level {level}%, "
              f"{temp_c if temp_c is not None else '?'} C)")
        if name not in ("good", "unknown"):
            found = True
            _emit_fix("battery_unhealthy")
        if temp_c is not None and temp_c >= 43:
            found = True
            _emit_fix("overheating")

    # Storage.
    df = shell(["df", "-h", "/data"])
    for line in df.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 5 and parts[4].endswith("%"):
            try:
                used = int(parts[4].rstrip("%"))
            except ValueError:
                continue
            print(f"  Storage used   : {used}% of /data")
            if used >= 90:
                found = True
                _emit_fix("storage_full")
            break

    # Memory.
    mem = shell(["cat", "/proc/meminfo"])
    total = avail = None
    for line in mem.splitlines():
        if line.startswith("MemTotal"):
            total = int(line.split()[1])
        elif line.startswith("MemAvailable"):
            avail = int(line.split()[1])
    if total and avail:
        pct = avail / total * 100
        print(f"  Free RAM       : {avail // 1024} MB ({pct:.0f}%)")
        if pct < 8:
            found = True
            _emit_fix("low_memory")

    # Recent crashes.
    crash = _run([ANDROID_TOOL, "logcat", "-b", "crash", "-d", "-t", "400"], timeout=30)
    fatals = [l for l in crash.stdout.splitlines() if "FATAL EXCEPTION" in l]
    if fatals:
        print(f"  Crash log      : {len(fatals)} recent fatal crash entr"
              f"{'y' if len(fatals) == 1 else 'ies'}")
        found = True
        _emit_fix("app_crashes")
    else:
        print("  Crash log      : no recent fatal crashes")

    if not found:
        print("\n  No software/battery faults detected in readable data.")
    print("\n  Physical hardware check:")
    _emit_fix("hardware_note")
    return 0


def ios_diagnose() -> int:
    """Read iPhone battery (cycle count) and recent crash reports; suggest fixes."""
    if not tool_available(IOS_TOOL):
        print(f"ERROR: {IOS_TOOL} not installed (pip install pymobiledevice3).")
        return 2

    print("iPhone diagnostics")
    print("-" * 18)
    found = False

    batt = _run([IOS_TOOL, "diagnostics", "battery"], timeout=60)
    if batt.returncode == 0 and batt.stdout.strip():
        cycles = _grep_value(batt.stdout, "CycleCount")
        if cycles:
            print(f"  Battery cycles : {cycles}")
            try:
                if int(cycles) >= 800:
                    found = True
                    _emit_fix("battery_worn")
            except ValueError:
                pass
        design = _grep_value(batt.stdout, "DesignCapacity")
        actual = _grep_value(batt.stdout, "AppleRawMaxCapacity") or \
            _grep_value(batt.stdout, "NominalChargeCapacity")
        try:
            if design and actual and int(actual) < 0.8 * int(design):
                found = True
                print("  Battery health : below ~80% of design capacity")
                _emit_fix("battery_worn")
        except ValueError:
            pass
    else:
        print("  Battery        : unreadable (unlock + trust, then retry)")

    crashes = _run([IOS_TOOL, "crash", "ls"], timeout=60)
    if crashes.returncode == 0:
        reports = [l for l in crashes.stdout.splitlines() if l.strip()
                   and not l.strip().endswith("/")]
        if reports:
            print(f"  Crash reports  : {len(reports)} on device")
            found = True
            _emit_fix("app_crashes")
        else:
            print("  Crash reports  : none found")

    if not found:
        print("\n  No software/battery faults detected in readable data.")
    print("\n  Physical hardware check:")
    _emit_fix("hardware_note")
    return 0


# ---------------------------------------------------------------------------
# Full specs
# ---------------------------------------------------------------------------
def ios_specs() -> int:
    """Dump a broad set of iPhone specs from lockdown info."""
    if not tool_available(IOS_TOOL):
        print(f"ERROR: {IOS_TOOL} not installed (pip install pymobiledevice3).")
        return 2
    res = _run([IOS_TOOL, "lockdown", "info"], timeout=60)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        print("Could not read device. Unlock the iPhone and tap 'Trust', then retry.")
        return res.returncode
    text = res.stdout
    product_type = _grep_value(text, "ProductType")
    named = apple_model_lookup(product_type)

    print("iPhone specifications")
    print("-" * 21)
    if named:
        print(f"  Model              : {named[0]}")
        print(f"  Chip (SoC)         : {named[1]}")
    fields = [
        ("ProductType", "Product type"),
        ("ModelNumber", "Model number"),
        ("RegionInfo", "Region"),
        ("ProductVersion", "iOS version"),
        ("BuildVersion", "Build"),
        ("DeviceClass", "Device class"),
        ("CPUArchitecture", "CPU architecture"),
        ("HardwareModel", "Hardware model"),
        ("DeviceColor", "Color"),
        ("TotalDiskCapacity", "Total storage (bytes)"),
        ("SerialNumber", "Serial number"),
        ("InternationalMobileEquipmentIdentity", "IMEI"),
        ("WiFiAddress", "Wi-Fi MAC"),
        ("BluetoothAddress", "Bluetooth MAC"),
    ]
    for key, label in fields:
        value = _grep_value(text, key)
        if value:
            print(f"  {label:<18} : {value}")
    return 0


def android_specs() -> int:
    """Dump a broad set of Android specs via adb getprop and shell utilities."""
    if not tool_available(ANDROID_TOOL):
        print(f"ERROR: {ANDROID_TOOL} not installed (Android platform-tools).")
        return 2

    def prop(name: str) -> str:
        r = _run([ANDROID_TOOL, "shell", "getprop", name], timeout=30)
        return r.stdout.strip() if r.returncode == 0 else ""

    def shell(args: list[str]) -> str:
        r = _run([ANDROID_TOOL, "shell", *args], timeout=30)
        return r.stdout.strip() if r.returncode == 0 else ""

    manufacturer = prop("ro.product.manufacturer")
    model = prop("ro.product.model")
    if not (manufacturer or model):
        print("No data - unlock the phone and authorize this computer, then retry.")
        return 1

    soc = f"{prop('ro.soc.manufacturer')} {prop('ro.soc.model')}".strip()
    platform = prop("ro.board.platform")
    if not soc:
        soc = QCOM_PLATFORMS.get(platform, platform)

    # Memory and storage.
    mem_total = ""
    meminfo = shell(["cat", "/proc/meminfo"])
    for line in meminfo.splitlines():
        if line.startswith("MemTotal"):
            try:
                kb = int(line.split()[1])
                mem_total = f"{kb / 1024 / 1024:.1f} GB"
            except (IndexError, ValueError):
                pass
            break
    storage = ""
    df = shell(["df", "-h", "/data"])
    for line in df.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            storage = f"{parts[1]} total (/data partition)"
            break

    screen = shell(["wm", "size"]).replace("Physical size:", "").strip()
    density = shell(["wm", "density"]).replace("Physical density:", "").strip()
    cores = ""
    cpuinfo = shell(["cat", "/proc/cpuinfo"])
    if cpuinfo:
        cores = str(cpuinfo.count("processor"))

    specs = [
        ("Device", f"{manufacturer} {model}".strip()),
        ("Chip (SoC)", soc),
        ("Platform code", platform),
        ("Android version", prop("ro.build.version.release")),
        ("API level (SDK)", prop("ro.build.version.sdk")),
        ("Build ID", prop("ro.build.display.id")),
        ("Build date", prop("ro.build.date")),
        ("Security patch", prop("ro.build.version.security_patch")),
        ("CPU ABI", prop("ro.product.cpu.abi")),
        ("CPU cores", cores),
        ("RAM", mem_total),
        ("Storage", storage),
        ("Screen", screen),
        ("Density (dpi)", density),
        ("Serial", prop("ro.serialno")),
    ]
    print("Android specifications")
    print("-" * 22)
    for label, value in specs:
        if value:
            print(f"  {label:<16} : {value}")
    return 0


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
# Cameras (PTP/MTP via gphoto2) - a different protocol from phones.
# ---------------------------------------------------------------------------
CAMERA_TOOL = "gphoto2"
_CAMERA_INSTALL = "install gphoto2 (Linux: apt install gphoto2; macOS: brew install gphoto2)"


def camera_info() -> int:
    """Detect a connected camera and print its model/summary via gphoto2."""
    if not tool_available(CAMERA_TOOL):
        print(f"ERROR: {CAMERA_TOOL} not installed - {_CAMERA_INSTALL}.")
        return 2
    detect = _run([CAMERA_TOOL, "--auto-detect"], timeout=60)
    sys.stdout.write(detect.stdout)
    lines = [l for l in detect.stdout.splitlines()[2:] if l.strip()]
    if not lines:
        print("No camera detected. Connect it, power on, and set USB mode to")
        print("PTP / 'PC connection' (not 'charge only' or 'mass storage').")
        return 1
    summary = _run([CAMERA_TOOL, "--summary"], timeout=60)
    print("\nCamera summary")
    print("-" * 14)
    sys.stdout.write(summary.stdout)
    if summary.returncode != 0:
        sys.stderr.write(summary.stderr)
    return 0


def camera_photos(out: Path) -> int:
    """Download all photos from the camera into `out` via gphoto2."""
    if not tool_available(CAMERA_TOOL):
        print(f"ERROR: {CAMERA_TOOL} not installed - {_CAMERA_INSTALL}.")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading all camera files -> {out}")
    # Run inside the destination so gphoto2 writes there.
    res = _run([CAMERA_TOOL, "--get-all-files", "--skip-existing"],
               timeout=7200, cwd=str(out))
    sys.stdout.write(res.stdout)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        print("\n  If it failed: set the camera's USB mode to PTP and ensure no")
        print("  other program (Photos, gvfs) has grabbed the device.")
    return res.returncode


# ---------------------------------------------------------------------------
# Cars (OBD-II via an ELM327 adapter + python-OBD) - reads trouble codes.
# ---------------------------------------------------------------------------
# A small map of common generic trouble codes to plain-language fixes.
DTC_FIXES = {
    "P0300": ("Random/multiple cylinder misfire",
              "Check spark plugs, coils, and fuel delivery; clear and recheck."),
    "P0171": ("System too lean (bank 1)",
              "Look for vacuum/intake leaks, a dirty MAF sensor, or weak fuel pump."),
    "P0420": ("Catalyst efficiency below threshold (bank 1)",
              "Often a failing catalytic converter or an O2 sensor; verify sensor first."),
    "P0455": ("Large EVAP system leak",
              "Check the gas cap first (reseat/replace), then EVAP hoses."),
    "P0128": ("Coolant thermostat below regulating temperature",
              "Usually a stuck-open thermostat; replace it."),
    "P0442": ("Small EVAP system leak",
              "Reseat/replace the gas cap; inspect EVAP lines for small cracks."),
}


def _decode_dtc_prefix(code: str) -> str:
    systems = {"P": "Powertrain", "C": "Chassis", "B": "Body", "U": "Network"}
    if not code:
        return ""
    return systems.get(code[0].upper(), "Unknown system")


def car_diagnose(port: str | None) -> int:
    """Read stored diagnostic trouble codes over OBD-II and suggest fixes."""
    try:
        import obd  # python-OBD; talks to an ELM327 adapter
    except ImportError:
        print("ERROR: python-OBD not installed (pip install obd).")
        print("You also need an ELM327 OBD-II adapter plugged into the car's")
        print("OBD-II port (usually under the dashboard) and the ignition on.")
        return 2

    print("Car OBD-II diagnostics")
    print("-" * 22)
    connection = obd.OBD(port) if port else obd.OBD()
    if not connection.is_connected():
        print("Could not connect to the vehicle. Check that:")
        print("  - the ELM327 adapter is seated in the OBD-II port,")
        print("  - the ignition is ON (engine running for live data),")
        print("  - the right serial port is given via --port (e.g. /dev/ttyUSB0,")
        print("    COM3, or a Bluetooth rfcomm device).")
        return 1

    vin = connection.query(obd.commands.VIN)
    if vin.value:
        print(f"  VIN            : {vin.value}")

    resp = connection.query(obd.commands.GET_DTC)
    codes = resp.value or []
    if not codes:
        print("  Trouble codes  : none stored. No stored faults reported.")
    else:
        print(f"  Trouble codes  : {len(codes)} stored")
        for code, desc in codes:
            system = _decode_dtc_prefix(code)
            print(f"\n  {code}  [{system}]")
            if desc:
                print(f"      Meaning: {desc}")
            if code in DTC_FIXES:
                meaning, fix = DTC_FIXES[code]
                print(f"      ({meaning})")
                print(f"      Fix: {fix}")
            else:
                print("      Fix: look up this exact code for your make/model; a")
                print("           scan-tool live-data check pinpoints the part.")
    connection.close()
    print("\n  Note: codes point to a symptom, not always the root cause. A shop")
    print("  scan tool with live data is authoritative before replacing parts.")
    return 0


# ---------------------------------------------------------------------------
# Auto: detect connected devices and run every read-only check on each
# ---------------------------------------------------------------------------
def _ios_device_present() -> bool:
    if not tool_available(IOS_TOOL):
        return False
    res = _run([IOS_TOOL, "usbmux", "list"], timeout=30)
    out = res.stdout.strip()
    return res.returncode == 0 and out not in ("", "[]")


def _android_targets() -> list[str]:
    if not tool_available(ANDROID_TOOL):
        return []
    res = _run([ANDROID_TOOL, "devices"], timeout=30)
    targets = []
    for line in res.stdout.splitlines()[1:]:
        line = line.strip()
        if line.endswith("\tdevice") or line.endswith(" device"):
            targets.append(line.split()[0])
    return targets


def auto_report(save: str | None = None) -> int:
    """Detect every reachable target and run all read-only checks on each."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("=" * 60)
        print("AUTOMATIC DEVICE REPORT")
        print(f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}")
        print("Read-only checks. Phones must be unlocked + trusted to read.")
        print("=" * 60)

        # This computer is always available.
        print("\n########## THIS COMPUTER ##########")
        computer_specs()
        print()
        computer_diagnose()

        # iPhone.
        print("\n########## iPHONE (Apple) ##########")
        if not tool_available(IOS_TOOL):
            print("  pymobiledevice3 not installed - skipping iPhone checks.")
        elif not _ios_device_present():
            print("  No iPhone detected (plug in, unlock, tap 'Trust').")
        else:
            for fn in (ios_hardware, ios_specs, ios_info, ios_authenticity,
                       ios_diagnose):
                print()
                fn()

        # Android (phones, TVs, boxes).
        print("\n########## ANDROID ##########")
        if not tool_available(ANDROID_TOOL):
            print("  adb not installed - skipping Android checks.")
        else:
            targets = _android_targets()
            if not targets:
                print("  No authorized Android device (unlock + allow USB/ADB).")
            else:
                print(f"  Detected: {', '.join(targets)}")
                for fn in (android_hardware, android_specs, android_info,
                           android_authenticity, android_diagnose):
                    print()
                    fn()

        print("\n" + "=" * 60)
        print("END OF REPORT")
        print("=" * 60)

    text = buf.getvalue()
    sys.stdout.write(text)
    if save:
        try:
            with open(save, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"\n(Report saved to {save})")
        except OSError as exc:
            print(f"\nCould not save report: {exc}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--auto", action="store_true",
                   help="detect all connected devices and run every check")
    p.add_argument("--save", metavar="FILE",
                   help="also write the --auto report to this file")
    p.add_argument("--check", action="store_true",
                   help="report installed backends and visible devices")
    p.add_argument("--ios", action="store_true", help="target an iPhone")
    p.add_argument("--android", action="store_true",
                   help="target an Android device (phone, TV, box, head-unit)")
    p.add_argument("--computer", action="store_true",
                   help="target THIS laptop/desktop (reads its own OS)")
    p.add_argument("--camera", action="store_true",
                   help="target a camera over PTP/MTP (needs gphoto2)")
    p.add_argument("--car", action="store_true",
                   help="target a car over OBD-II (needs python-OBD + ELM327)")
    p.add_argument("--port", metavar="DEV",
                   help="serial port for --car (e.g. /dev/ttyUSB0, COM3)")
    p.add_argument("--net", metavar="HOST[:PORT]",
                   help="connect adb over the network first (Android TV / Fire TV "
                        "/ boxes); e.g. --net 192.168.1.50:5555")
    p.add_argument("--info", action="store_true",
                   help="read serial / IMEI from the unlocked, trusted device")
    p.add_argument("--hardware", action="store_true",
                   help="report model + chipset (SoC) from the connected device")
    p.add_argument("--specs", action="store_true",
                   help="dump full specs (OS, RAM, storage, screen, SoC, ...)")
    p.add_argument("--apple-model", dest="apple_model", metavar="PRODUCTTYPE",
                   help="offline: map an Apple product type (e.g. iPhone12,1) to "
                        "its model name and chip - no device needed")
    p.add_argument("--apple-serial-date", dest="apple_serial_date", metavar="SERIAL",
                   help="offline: estimate manufacture date from a legacy Apple "
                        "serial - no device needed")
    p.add_argument("--authenticity", action="store_true",
                   help="run counterfeit/authenticity heuristics on the device")
    p.add_argument("--diagnose", action="store_true",
                   help="check battery/storage/thermal/crashes and suggest fixes")
    p.add_argument("--photos", action="store_true", help="pull the camera roll (DCIM)")
    p.add_argument("--backup", action="store_true", help="full iPhone backup (iOS only)")
    p.add_argument("--out", type=Path, default=Path("./export"),
                   help="destination folder (default: ./export)")
    args = p.parse_args(argv)

    # Auto: detect everything and run all checks.
    if args.auto:
        return auto_report(save=args.save)

    # Offline lookup needs no device and no backend tools.
    if args.apple_model:
        named = apple_model_lookup(args.apple_model)
        print(f"Apple product type: {args.apple_model}")
        if named:
            print(f"  Model : {named[0]}")
            print(f"  Chip  : {named[1]}")
        else:
            print("  Not in the local map. Cross-check at checkcoverage.apple.com.")
        return 0

    if args.apple_serial_date:
        print(f"Apple serial: {args.apple_serial_date}")
        print(f"  Manufacture date: {apple_serial_date(args.apple_serial_date)}")
        return 0

    # This computer: no device, no protocol - reads the local OS.
    if args.computer:
        if args.diagnose:
            return computer_diagnose()
        return computer_specs()  # default to specs

    # Camera over PTP/MTP (gphoto2).
    if args.camera:
        if args.photos:
            return camera_photos(args.out)
        return camera_info()  # default: detect + summary

    # Car over OBD-II (python-OBD + ELM327 adapter).
    if args.car:
        return car_diagnose(args.port)  # diagnostics is the only action

    # Network adb for Android TVs / boxes / head-units.
    if args.net:
        if not tool_available(ANDROID_TOOL):
            print(f"ERROR: {ANDROID_TOOL} not installed (needed for --net).")
            return 2
        target = args.net if ":" in args.net else f"{args.net}:5555"
        conn = _run([ANDROID_TOOL, "connect", target], timeout=30)
        sys.stdout.write(conn.stdout)
        if "connected" not in conn.stdout.lower():
            print("Could not connect. On the device, enable Developer options ->")
            print("'ADB debugging' / 'Network debugging' and confirm the IP:port.")
            return 1
        args.android = True  # treat it as an Android target from here on

    any_target = args.ios or args.android or args.computer or args.camera or args.car
    if args.check or not any_target:
        check_environment()
        if not any_target:
            print("\nNothing to do. Pick a target (--ios / --android / --computer"
                  " / --camera / --car) and an action (--info/--hardware/--specs/"
                  "--authenticity/--diagnose/--photos), or --auto / --apple-model.")
        return 0

    if args.ios and args.info:
        return ios_info()
    if args.android and args.info:
        return android_info()
    if args.ios and args.hardware:
        return ios_hardware()
    if args.android and args.hardware:
        return android_hardware()
    if args.ios and args.specs:
        return ios_specs()
    if args.android and args.specs:
        return android_specs()
    if args.ios and args.authenticity:
        return ios_authenticity()
    if args.android and args.authenticity:
        return android_authenticity()
    if args.ios and args.diagnose:
        return ios_diagnose()
    if args.android and args.diagnose:
        return android_diagnose()
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

    p.error("choose an action: --info, --hardware, --specs, --authenticity, "
            "--diagnose, --photos (or --backup for iOS)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
