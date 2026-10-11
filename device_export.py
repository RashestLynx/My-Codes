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
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true",
                   help="report installed backends and visible devices")
    p.add_argument("--ios", action="store_true", help="target an iPhone")
    p.add_argument("--android", action="store_true", help="target an Android phone")
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
    p.add_argument("--photos", action="store_true", help="pull the camera roll (DCIM)")
    p.add_argument("--backup", action="store_true", help="full iPhone backup (iOS only)")
    p.add_argument("--out", type=Path, default=Path("./export"),
                   help="destination folder (default: ./export)")
    args = p.parse_args(argv)

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

    if args.check or not (args.ios or args.android):
        check_environment()
        if not (args.ios or args.android):
            print("\nNothing to do (pass --ios or --android with "
                  "--info/--hardware/--photos/--backup, or use --apple-model).")
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
            "--photos (or --backup for iOS)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
