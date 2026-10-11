#!/usr/bin/env python3
"""phone_info.py - look up public information about a phone from its IMEI or serial.

This tool does NOT bypass locks or access private data. It only:
  * validates an IMEI (Luhn checksum) and splits out its parts (TAC / serial),
  * does a light sanity check on an Apple-style serial number,
  * points you at the OFFICIAL services that can recover the account/data
    tied to the device.

Usage:
    python phone_info.py --imei 490154203237518
    python phone_info.py --serial C39XXXXXXXXX
    python phone_info.py --imei 490154203237518 --platform apple
"""

from __future__ import annotations

import argparse
import sys

# Official, first-party services. These are where real recovery happens:
# you recover the ACCOUNT, then the account gives you the device + backups.
RECOVERY_SERVICES = {
    "apple": [
        ("Apple Account recovery", "https://iforgot.apple.com"),
        ("Check coverage / verify a device", "https://checkcoverage.apple.com"),
        ("iCloud (backups, photos, Find My)", "https://www.icloud.com"),
        ("Manage your Apple Account", "https://appleid.apple.com"),
    ],
    "samsung": [
        ("Samsung account recovery", "https://account.samsung.com"),
        ("Find My Mobile (remote unlock if enabled)", "https://findmymobile.samsung.com"),
        ("Google account recovery", "https://accounts.google.com/signin/recovery"),
        ("Google Find Hub (device + backups)", "https://www.google.com/android/find"),
    ],
}


def luhn_is_valid(number: str) -> bool:
    """Return True if the digit string passes the Luhn checksum (used by IMEI)."""
    if not number.isdigit():
        return False
    total = 0
    reverse = number[::-1]
    for i, ch in enumerate(reverse):
        d = int(ch)
        if i % 2 == 1:  # every second digit from the right is doubled
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def analyze_imei(imei: str) -> dict:
    """Break an IMEI into its parts and report validity.

    IMEI layout (15 digits):
        TAC (8)  - Type Allocation Code: identifies the make/model
        SNR (6)  - device serial within that TAC
        CD  (1)  - Luhn check digit
    """
    cleaned = "".join(c for c in imei if c.isdigit())
    result = {
        "input": imei,
        "digits": cleaned,
        "length_ok": len(cleaned) == 15,
        "valid_checksum": False,
        "tac": None,
        "serial": None,
        "check_digit": None,
    }
    if len(cleaned) == 15:
        result["valid_checksum"] = luhn_is_valid(cleaned)
        result["tac"] = cleaned[:8]
        result["serial"] = cleaned[8:14]
        result["check_digit"] = cleaned[14]
    return result


def analyze_apple_serial(serial: str) -> dict:
    """Light structural check for an Apple serial number.

    Modern Apple serials are typically 10-12 alphanumeric characters.
    This does not decode manufacturing details; use checkcoverage.apple.com
    for authoritative device info.
    """
    cleaned = serial.strip().upper()
    return {
        "input": serial,
        "normalized": cleaned,
        "length": len(cleaned),
        "plausible": 10 <= len(cleaned) <= 12 and cleaned.isalnum(),
    }


def print_services(platform: str | None) -> None:
    keys = [platform] if platform in RECOVERY_SERVICES else list(RECOVERY_SERVICES)
    print("\nOfficial recovery / lookup services")
    print("-" * 38)
    print("Recover the ACCOUNT first; the account unlocks the device and backups.")
    for key in keys:
        print(f"\n{key.capitalize()}:")
        for name, url in RECOVERY_SERVICES[key]:
            print(f"  - {name}: {url}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--imei", help="15-digit IMEI to validate and break down")
    parser.add_argument("--serial", help="device serial number to sanity-check")
    parser.add_argument(
        "--platform",
        choices=sorted(RECOVERY_SERVICES),
        help="limit the service list to apple or samsung",
    )
    args = parser.parse_args(argv)

    if not args.imei and not args.serial:
        parser.error("give at least one of --imei or --serial")

    if args.imei:
        info = analyze_imei(args.imei)
        print("IMEI analysis")
        print("-" * 13)
        print(f"  cleaned digits : {info['digits']}")
        print(f"  length is 15   : {info['length_ok']}")
        print(f"  checksum valid : {info['valid_checksum']}")
        if info["tac"]:
            print(f"  TAC (make/model prefix) : {info['tac']}")
            print(f"  device serial portion   : {info['serial']}")
            print(f"  Luhn check digit        : {info['check_digit']}")
        if info["length_ok"] and not info["valid_checksum"]:
            print("  NOTE: 15 digits but checksum fails - likely mistyped.")

    if args.serial:
        s = analyze_apple_serial(args.serial)
        print("\nSerial analysis")
        print("-" * 15)
        print(f"  normalized : {s['normalized']}")
        print(f"  length     : {s['length']}")
        print(f"  plausible  : {s['plausible']}")
        print("  (Use checkcoverage.apple.com for authoritative Apple device info.)")

    print_services(args.platform)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
