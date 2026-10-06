#!/usr/bin/env python3
r"""
PS3 disc dump tool -- for discs you own.

Reads a PS3 disc (or image file) sector by sector, decrypts the encrypted
regions with a disc key YOU supply, and writes a decrypted ISO for RPCS3.

Keys come from you, not from the drive: a Redump key file or an IRD file for
your exact disc version (see psdevwiki.com/ps3/Bluray_disc). Redump's "Disc
Key" is the final 16-byte key (--key-type final); an IRD's data1 is d1.

D1 KEYS
  --key-type d1 derives the disc key from d1 (from --key-file, or the IRD's
  data1 if no key file is given). The derivation's constant AES key and IV
  are NOT included; put them in a file (16-byte key then 16-byte IV, raw or
  hex) and pass --d1-secrets FILE or set PS3_D1_SECRETS. With a Redump disc
  key you don't need any of this: use --key-type final.

FORMAT NOTES
  - Region table (sector 0): u32 BE count of *unencrypted* regions, u32
    reserved, then (start, end) u32 BE pairs with inclusive ends, one per
    unencrypted region. The gaps between them are the encrypted regions.
    All-zero sectors are left alone. (Matches ps3dec / redump tooling.)
  - Per-sector CBC IV = sector number, 16-byte big-endian.
  - IRD layout (read_ird) and IRD region/file hashes are still UNVERIFIED --
    an IRD mismatch may mean the parser is wrong rather than the dump.

COMMANDS
  selftest                          offline checks, no disc needed
  drives                            list optical drives and whether a disc is in
  probe  [--device D] [--wait S]    identify the disc; no key needed
  ird-info game.ird                 inspect an IRD
  check-key [--device D] --key-file K --key-type final|d1 [--ird game.ird]
                                    spot-check a key on the disc in seconds
  dump   [--device D] [--key-file K] --key-type final|d1 --out game.iso
         [--d1-secrets FILE] [--ird game.ird] [--redump-dat redump.dat]
         [--redump-sha1 H] [--verify] [--limit-sectors N] [--sectors N]
         [--force] [--resume] [--wait S] [--no-spot-check] [--no-hash]
  verify game.iso [--ird game.ird] [--redump-dat redump.dat] [--hashes F]
                                    full integrity check of a finished ISO

VERIFICATION
  - Key spot check (before dumping): decrypts samples of the encrypted
    regions and checks IRD file MD5s, file magics (SCE\0, \0PSF, ...) and
    entropy. A wrong key aborts here instead of after an hour.
  - Hashes: CRC32, MD5 and SHA-1 of the raw disc (encrypted, exactly what
    Redump lists) and of the decrypted ISO, saved to game.iso.hashes.txt.
  - Redump: raw hashes compared to a Redump .dat or --redump-sha1/md5/crc32.
  - IRD: per-region MD5s and per-file MD5s of the decrypted ISO.
  - --verify / verify: re-read the finished ISO from storage and recheck it.

  --device defaults to "auto": the drive holding a PS3 disc is picked for
  you. Explicit examples: /dev/sr0 (Linux, may need sudo), \\.\E: (Windows,
  run as administrator), /dev/rdisk2 (macOS).
  Tip: run dump with --limit-sectors 4096 first as a quick trial.

Requires: pip install cryptography
"""
import argparse
import bisect
import collections
import glob
import gzip
import hashlib
import math
import os
import queue
import random
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
import zlib

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

SECTOR = 2048
CHUNK_SECTORS = 512          # 1 MiB per read
ISO_MAGIC = b"\x01CD001"     # ISO 9660 primary volume descriptor, sector 16
PS3_VOLUME_ID = "PS3VOLUME"
RETRIES = 3
ZERO_SECTOR = bytes(SECTOR)
PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


class Ps3Error(Exception):
    pass


# ---------------------------------------------------------------- crypto ---
def sector_iv(sector_no: int) -> bytes:
    return sector_no.to_bytes(16, "big")


def decrypt_sector(key: bytes, sector_no: int, data: bytes) -> bytes:
    d = Cipher(algorithms.AES(key), modes.CBC(sector_iv(sector_no))).decryptor()
    return d.update(data) + d.finalize()


def encrypt_sector(key: bytes, sector_no: int, data: bytes) -> bytes:
    e = Cipher(algorithms.AES(key), modes.CBC(sector_iv(sector_no))).encryptor()
    return e.update(data) + e.finalize()


def load_d1_secrets(path):
    """Read the d1 -> disc key constants: 16-byte key then 16-byte IV.

    Accepts 32 raw bytes or 64 hex characters (whitespace ignored). The
    constants are not shipped with this tool; supply them yourself.
    """
    if not path:
        raise Ps3Error("--key-type d1 needs --d1-secrets FILE (or the "
                       "PS3_D1_SECRETS environment variable) holding the "
                       "16-byte derivation key followed by the 16-byte IV. "
                       "Or use a Redump disc key with --key-type final.")
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise Ps3Error(f"Cannot read d1 secrets file: {e}")
    if len(raw) != 32:
        txt = "".join(raw.decode("ascii", "ignore").split())
        try:
            raw = bytes.fromhex(txt) if len(txt) == 64 else b""
        except ValueError:
            raw = b""
    if len(raw) != 32:
        raise Ps3Error(f"{path}: expected 32 raw bytes or 64 hex chars "
                       f"(16-byte key, then 16-byte IV)")
    return raw[:16], raw[16:]


def derive_disc_key(d1: bytes, secret_key: bytes, secret_iv: bytes) -> bytes:
    """d1 -> final disc key: one AES-128-CBC encryption block."""
    if len(d1) != 16:
        raise Ps3Error(f"d1 must be 16 bytes, got {len(d1)}")
    e = Cipher(algorithms.AES(secret_key), modes.CBC(secret_iv)).encryptor()
    return e.update(d1) + e.finalize()


# ------------------------------------------------------------------ keys ---
def load_key_file(path: str) -> bytes:
    """Raw 16-byte file, or text containing 32 hex characters."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise Ps3Error(f"Cannot read key file: {e}")
    if len(raw) == 16:
        return raw
    txt = "".join(raw.decode("ascii", "ignore").split())
    if len(txt) == 32:
        try:
            return bytes.fromhex(txt)
        except ValueError:
            pass
    raise Ps3Error(f"{path}: expected 16 raw bytes or 32 hex chars "
                   f"(file is {len(raw)} bytes)")


# --------------------------------------------------------------- regions ---
def parse_regions(sector0: bytes):
    """Return [(start, end, encrypted)] in sector numbers, end inclusive.

    Sector 0 lists only the unencrypted regions; encrypted regions are the
    gaps between consecutive unencrypted ones.
    """
    if len(sector0) < SECTOR:
        raise Ps3Error("Sector 0 is short; the read failed")
    count = struct.unpack(">I", sector0[0:4])[0]
    if not 1 <= count <= (SECTOR - 8) // 8:
        raise Ps3Error(
            f"Implausible region count {count}. First bytes of sector 0: "
            f"{sector0[:32].hex(' ')}. Not a PS3 disc, or the read failed.")
    pairs = struct.unpack(f">{count * 2}I", sector0[8:8 + count * 8])
    regions, prev_end = [], -1
    for i in range(count):
        start, end = pairs[2 * i], pairs[2 * i + 1]
        if start > end or start <= prev_end:
            raise Ps3Error(f"Region {i} ({start}-{end}) is reversed or "
                           f"overlaps the previous one; bad region table.")
        if i and start > prev_end + 1:
            regions.append((prev_end + 1, start - 1, True))
        regions.append((start, end, False))
        prev_end = end
    return regions


class EncryptedIndex:
    """Fast 'which sectors of this chunk are encrypted' lookups."""

    def __init__(self, regions):
        enc = [(s, e + 1) for s, e, is_enc in regions if is_enc]
        self.starts = [s for s, _ in enc]
        self.ends = [e for _, e in enc]          # exclusive

    def ranges(self, first: int, last: int):
        """Yield (lo, hi) encrypted sub-ranges of [first, last)."""
        i = bisect.bisect_right(self.ends, first)
        while i < len(self.starts) and self.starts[i] < last:
            yield max(self.starts[i], first), min(self.ends[i], last)
            i += 1


def is_encrypted(sector_no: int, regions) -> bool:
    return any(enc and s <= sector_no <= e for s, e, enc in regions)


def decrypt_chunk(aes, view: memoryview, first: int, index: EncryptedIndex):
    """Decrypt, in place, the encrypted sectors of a chunk starting at `first`."""
    last = first + len(view) // SECTOR
    for lo, hi in index.ranges(first, last):
        for s in range(lo, hi):
            off = (s - first) * SECTOR
            blk = view[off:off + SECTOR]
            if blk == ZERO_SECTOR:               # unwritten / zero-filled
                continue
            blk[:] = Cipher(aes, modes.CBC(sector_iv(s))).decryptor().update(blk)


class RegionHasher:
    """MD5 per region, fed chunk by chunk (for IRD comparison)."""

    def __init__(self, regions):
        self.regions = regions
        self.md5 = [hashlib.md5() for _ in regions]
        self.starts = [r[0] for r in regions]

    def feed(self, first: int, view: memoryview):
        last = first + len(view) // SECTOR
        i = max(bisect.bisect_right(self.starts, first) - 1, 0)
        while i < len(self.regions) and self.regions[i][0] < last:
            s, e, _ = self.regions[i]
            lo, hi = max(s, first), min(e + 1, last)
            if lo < hi:
                self.md5[i].update(view[(lo - first) * SECTOR:(hi - first) * SECTOR])
            i += 1

    def digests(self):
        return [m.digest() for m in self.md5]


def compare_ird_hashes(digests, ird_hashes, regions):
    """Return a (status, detail) check result."""
    if len(digests) != len(ird_hashes):
        return WARN, (f"skipped: IRD has {len(ird_hashes)} region hashes, "
                      f"disc has {len(digests)} regions")
    bad = [i for i, (a, b) in enumerate(zip(digests, ird_hashes)) if a != b]
    if not bad:
        return PASS, f"all {len(digests)} region MD5s match"
    enc_only = all(regions[i][2] for i in bad)
    return FAIL, (f"{len(bad)}/{len(digests)} regions differ (indexes {bad[:8]})"
                  + ("; only encrypted regions differ (wrong key, or damage there)"
                     if enc_only else ""))


# ------------------------------------------------------------------- IRD ---
def read_ird(path: str) -> dict:
    """Best-effort IRD reader. UNTESTED on real files (layout from memory)."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
        if data[:4] != b"3IRD":
            raise Ps3Error("Not an IRD file (bad magic)")
        version = data[4]
        pos = 5
        game_id = data[pos:pos + 9].decode("ascii", "replace"); pos += 9
        nlen = data[pos]; pos += 1
        name = data[pos:pos + nlen].decode("utf-8", "replace"); pos += nlen
        pos += 4 + 5 + 5                      # update / game / app version
        if version == 7:
            pos += 4
        hlen = struct.unpack("<I", data[pos:pos + 4])[0]; pos += 4
        header = gzip.decompress(data[pos:pos + hlen]); pos += hlen
        flen = struct.unpack("<I", data[pos:pos + 4])[0]; pos += 4 + flen
        rcount = data[pos]; pos += 1
        hashes = [data[pos + 16 * i:pos + 16 * (i + 1)] for i in range(rcount)]
        pos += 16 * rcount
        fcount = struct.unpack("<I", data[pos:pos + 4])[0]; pos += 4
        files = {}
        for _ in range(fcount):                 # u64 start sector, MD5
            files[struct.unpack("<Q", data[pos:pos + 8])[0]] = data[pos + 8:pos + 24]
            pos += 24
        pos += 4                              # extra config + attachments
        if version >= 9:
            pos += 115                        # PIC
        data1 = data[pos:pos + 16]
        if len(data1) != 16:
            raise Ps3Error("IRD truncated before data1")
    except (OSError, struct.error, EOFError, gzip.BadGzipFile, zlib.error,
            IndexError) as e:
        raise Ps3Error(f"Could not parse IRD ({e}); layout assumption may be wrong")
    return {"version": version, "game_id": game_id, "name": name,
            "header": header, "region_hashes": hashes, "file_hashes": files,
            "data1": data1}


def normalize_title_id(tid) -> str:
    return re.sub(r"[^A-Z0-9]", "", (tid or "").upper())


# --------------------------------------------------------------- device ----
def open_device(path: str):
    try:
        return open(path, "rb", buffering=0)
    except OSError as e:
        raise Ps3Error(f"Cannot open {path}: {e} (admin/sudo needed? disc inserted?)")


def read_into(dev, sector: int, view: memoryview) -> int:
    """Fill `view` from `sector` onward; returns bytes read (may be short)."""
    dev.seek(sector * SECTOR)
    got = 0
    while got < len(view):
        n = dev.readinto(view[got:])
        if not n:
            break
        got += n
    return got


def read_at(dev, sector: int, count: int) -> bytes:
    buf = bytearray(count * SECTOR)
    try:
        n = read_into(dev, sector, memoryview(buf))
    except OSError:
        return b""
    return bytes(buf[:n])


def read_chunk(dev, sector: int, view: memoryview):
    """Read a whole chunk; fall back to per-sector retries. Returns bad sectors."""
    try:
        if read_into(dev, sector, view) == len(view):
            return []
    except OSError:
        pass
    bad = []
    for i in range(len(view) // SECTOR):
        part = view[i * SECTOR:(i + 1) * SECTOR]
        for attempt in range(RETRIES):
            try:
                if read_into(dev, sector + i, part) == SECTOR:
                    break
            except OSError:
                pass
            time.sleep(0.2 * (attempt + 1))      # let the drive recover
        else:
            bad.append(sector + i)
            part[:] = ZERO_SECTOR
    return bad


def _ioctl_size(fd: int) -> int:
    try:
        if sys.platform.startswith("linux"):
            import fcntl
            BLKGETSIZE64 = 0x80081272
            return struct.unpack("Q", fcntl.ioctl(fd, BLKGETSIZE64, b"\0" * 8))[0]
        if sys.platform == "darwin":
            import fcntl
            bs = struct.unpack("I", fcntl.ioctl(fd, 0x40046418, b"\0" * 4))[0]
            bc = struct.unpack("Q", fcntl.ioctl(fd, 0x40086419, b"\0" * 8))[0]
            return bs * bc
        if os.name == "nt":
            import ctypes
            import msvcrt
            from ctypes import wintypes
            IOCTL_DISK_GET_LENGTH_INFO = 0x0007405C
            length = ctypes.c_longlong(0)
            ret = wintypes.DWORD(0)
            ok = ctypes.windll.kernel32.DeviceIoControl(
                wintypes.HANDLE(msvcrt.get_osfhandle(fd)),
                IOCTL_DISK_GET_LENGTH_INFO, None, 0, ctypes.byref(length),
                ctypes.sizeof(length), ctypes.byref(ret), None)
            return length.value if ok else 0
    except (OSError, ImportError, AttributeError):
        pass
    return 0


def _windows_volume_size(path: str) -> int:
    m = re.match(r"^\\\\[.?]\\([A-Za-z]):$", path)
    if os.name != "nt" or not m:
        return 0
    import ctypes
    total = ctypes.c_ulonglong(0)
    ok = ctypes.windll.kernel32.GetDiskFreeSpaceExW(
        f"{m.group(1)}:\\", None, ctypes.byref(total), None)
    return total.value if ok else 0


def device_sectors(dev, override=None) -> int:
    if override:
        return override
    fd = dev.fileno()
    if stat.S_ISREG(os.fstat(fd).st_mode):
        size = os.fstat(fd).st_size
    else:
        size = _ioctl_size(fd) or _windows_volume_size(dev.name)
        if not size:
            try:
                size = os.lseek(fd, 0, os.SEEK_END)
            except OSError:
                size = 0
    if size <= 0:
        raise Ps3Error("Could not detect disc size; pass --sectors N.")
    return size // SECTOR


# ------------------------------------------------------- drive detection ---
def _read_text(path: str) -> str:
    try:
        with open(path, encoding="ascii", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return ""


def _linux_drives():
    drives = []
    for sysdir in sorted(glob.glob("/sys/block/sr*")):
        name = os.path.basename(sysdir)
        udev = {}
        majmin = _read_text(f"{sysdir}/dev")
        for line in _read_text(f"/run/udev/data/b{majmin}").splitlines():
            if line.startswith("E:") and "=" in line:
                k, v = line[2:].split("=", 1)
                udev[k] = v
        size512 = int(_read_text(f"{sysdir}/size") or 0)
        drives.append({
            "device": f"/dev/{name}",
            "name": " ".join(filter(None, (_read_text(f"{sysdir}/device/vendor"),
                                           _read_text(f"{sysdir}/device/model")))),
            "media": size512 > 0,
            "sectors": size512 * 512 // SECTOR,
            "label": udev.get("ID_FS_LABEL"),
            "bd": (udev.get("ID_CDROM_BD") == "1") if udev else None,
        })
    return drives


def _windows_drives():
    import ctypes
    k32 = ctypes.windll.kernel32
    drives, mask = [], k32.GetLogicalDrives()
    for i in range(26):
        if not mask >> i & 1:
            continue
        root = f"{chr(65 + i)}:\\"
        if k32.GetDriveTypeW(root) != 5:          # DRIVE_CDROM
            continue
        label = ctypes.create_unicode_buffer(261)
        media = bool(k32.GetVolumeInformationW(root, label, 261, None, None,
                                               None, None, 0))
        dev = rf"\\.\{chr(65 + i)}:"
        drives.append({"device": dev, "name": root, "media": media,
                       "sectors": _windows_volume_size(dev) // SECTOR if media else 0,
                       "label": label.value if media else None, "bd": None})
    return drives


def _mac_drives():
    try:
        out = subprocess.run(["drutil", "status"], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    vendor = re.search(r"Vendor\s+Product.*\n\s*(.+)", out)
    return [{"device": d.replace("/dev/disk", "/dev/rdisk"),
             "name": vendor.group(1).strip() if vendor else "optical drive",
             "media": True, "sectors": 0, "label": None,
             "bd": "BD" in out or None}
            for d in sorted(set(re.findall(r"/dev/disk\d+", out)))]


def list_drives():
    if sys.platform.startswith("linux"):
        return _linux_drives()
    if os.name == "nt":
        return _windows_drives()
    if sys.platform == "darwin":
        return _mac_drives()
    return []


def describe_drive(d) -> str:
    bits = [d["device"], d["name"] or "?"]
    if d["bd"] is False:
        bits.append("[not a Blu-ray drive]")
    if not d["media"]:
        bits.append("-- no disc")
    else:
        bits.append(f"-- disc in, label {d['label'] or '?'}")
        if d["sectors"]:
            bits.append(f"{d['sectors'] * SECTOR / 1e9:.2f} GB")
    return "  ".join(bits)


def choose_device(device, wait=0.0) -> str:
    """Return `device`, or auto-detect the drive holding a PS3 disc."""
    if device and device != "auto":
        return device
    deadline = time.time() + (wait or 0)
    announced = False
    while True:
        drives = list_drives()
        loaded = [d for d in drives if d["media"] and d["bd"] is not False]
        ps3 = [d for d in loaded if (d["label"] or "").upper() == PS3_VOLUME_ID]
        pick = ps3 if ps3 else loaded
        if len(pick) == 1:
            print(f"Using {describe_drive(pick[0])}")
            return pick[0]["device"]
        if len(pick) > 1:
            raise Ps3Error("Several drives have a disc; pick one with --device:\n  "
                           + "\n  ".join(describe_drive(d) for d in pick))
        if time.time() >= deadline:
            if not drives:
                raise Ps3Error("No optical drive found. Pass --device "
                               "explicitly (or an image file path).")
            raise Ps3Error("No disc found in: \n  " + "\n  ".join(
                describe_drive(d) for d in drives) + "\nInsert the disc or "
                "use --wait SECONDS.")
        if not announced:
            print("Waiting for a disc...", flush=True)
            announced = True
        time.sleep(2)


# ------------------------------------------------- disc identification ---
def _iso_dir(read, lba: int, size: int):
    """Yield (NAME, lba, size, flags) for an ISO 9660 directory."""
    data = read(lba, (min(size, 1 << 20) + SECTOR - 1) // SECTOR)
    pos = 0
    while pos < len(data):
        ln = data[pos]
        if ln == 0:                               # records never span sectors
            pos = (pos // SECTOR + 1) * SECTOR
            continue
        rec = data[pos:pos + ln]
        if len(rec) < 34:
            break
        nl = rec[32]
        name = rec[33:33 + nl]
        if name not in (b"\0", b"\1"):
            yield (name.decode("ascii", "replace").split(";")[0].upper(),
                   int.from_bytes(rec[2:6], "little"),
                   int.from_bytes(rec[10:14], "little"), rec[25])
        pos += ln


def _iso_find(read, pvd: bytes, path: str):
    """Return the bytes of `path` (e.g. 'PS3_GAME/PARAM.SFO'), or None."""
    lba = int.from_bytes(pvd[158:162], "little")
    size = int.from_bytes(pvd[166:170], "little")
    parts = path.upper().split("/")
    for i, part in enumerate(parts):
        for name, elba, esize, flags in _iso_dir(read, lba, size):
            if name == part and bool(flags & 2) == (i < len(parts) - 1):
                lba, size = elba, esize
                break
        else:
            return None
    return read(lba, (min(size, 1 << 20) + SECTOR - 1) // SECTOR)[:size]


def walk_iso(read, pvd: bytes):
    """Return [(path, lba, size)] for every file on the ISO 9660 tree.

    Multi-extent files (> 4 GiB) are merged; their extents are assumed to
    be contiguous, which is how PS3 discs are mastered.
    """
    files, seen = [], set()
    stack = [("", int.from_bytes(pvd[158:162], "little"),
              int.from_bytes(pvd[166:170], "little"))]
    while stack:
        prefix, lba, size = stack.pop()
        if lba in seen:
            continue
        seen.add(lba)
        pending = None
        for name, elba, esize, flags in _iso_dir(read, lba, size):
            if flags & 2:
                stack.append((prefix + name + "/", elba, esize))
                continue
            if pending and pending[0] == prefix + name:
                pending[2] += esize
            else:
                if pending:
                    files.append(tuple(pending))
                pending = [prefix + name, elba, esize]
            if not flags & 0x80:                 # last extent of this file
                files.append(tuple(pending))
                pending = None
        if pending:
            files.append(tuple(pending))
    return sorted(files, key=lambda f: f[1])


def parse_sfb_title_id(data):
    if not data or data[:4] != b".SFB":
        return None
    for off in range(0x20, min(len(data), 0x200), 0x20):
        key = data[off:off + 16].rstrip(b"\0")
        if not key:
            break
        if key == b"TITLE_ID":
            o, n = struct.unpack(">II", data[off + 16:off + 24])
            return data[o:o + n].rstrip(b"\0").decode("ascii", "replace")
    m = re.search(rb"[A-Z]{4}-?\d{5}", data)
    return m.group().decode() if m else None


def parse_sfo_title(data):
    if not data or data[:4] != b"\0PSF":
        return None
    kstart, dstart, count = struct.unpack("<III", data[8:20])
    for i in range(min(count, 256)):
        koff, _fmt, ln, _mx, doff = struct.unpack("<HHIII", data[20 + 16 * i:36 + 16 * i])
        key = data[kstart + koff:data.index(b"\0", kstart + koff)]
        if key == b"TITLE":
            return data[dstart + doff:dstart + doff + ln].rstrip(b"\0").decode("utf-8", "replace")
    return None


def identify_disc(read) -> dict:
    """Best-effort: ISO volume id, title id (PS3_DISC.SFB) and title (PARAM.SFO).

    `read(sector, count)` must return decrypted data when it can; without a
    key PARAM.SFO may sit in an encrypted region and come back as None.
    """
    info = {"iso": False, "volume_id": None, "title_id": None, "title": None}
    pvd = read(16, 1)
    if len(pvd) < SECTOR or pvd[:6] != ISO_MAGIC:
        return info
    info["iso"] = True
    info["volume_id"] = pvd[40:72].decode("ascii", "replace").strip()
    try:
        info["title_id"] = parse_sfb_title_id(_iso_find(read, pvd, "PS3_DISC.SFB"))
        info["title"] = parse_sfo_title(_iso_find(read, pvd, "PS3_GAME/PARAM.SFO"))
    except (struct.error, ValueError, IndexError):
        pass
    return info


def make_reader(dev, regions=None, key=None):
    """read(sector, count) that decrypts on the fly when a key is known."""
    aes = algorithms.AES(key) if key and regions else None
    index = EncryptedIndex(regions) if aes else None

    def read(sector, count):
        buf = bytearray(read_at(dev, sector, count))
        if aes and len(buf) % SECTOR == 0:
            decrypt_chunk(aes, memoryview(buf), sector, index)
        return bytes(buf)
    return read


def print_disc_info(info):
    if not info["iso"]:
        print("ISO 9660 descriptor at sector 16: NOT found "
              "(drive may not expose this disc's data, or it isn't readable)")
        return
    print(f"Volume id: {info['volume_id']}"
          + ("" if info["volume_id"] == PS3_VOLUME_ID else "  (not PS3VOLUME -- "
             "is this a PS3 game disc?)"))
    print(f"Title id:  {info['title_id'] or 'not found'}")
    print(f"Title:     {info['title'] or 'unknown (PARAM.SFO not readable without key)'}")


# ---------------------------------------------------------------- probe ----
def probe(device: str):
    """Check the drive reads the disc and identify it. No key needed."""
    with open_device(device) as dev:
        try:
            total = device_sectors(dev)
            print(f"Readable size: {total} sectors ({total * SECTOR / 1e9:.2f} GB)")
        except Ps3Error as e:
            print("Readable size:", e)
        s0 = read_at(dev, 0, 1)
        try:
            regions = parse_regions(s0)
            enc = [r for r in regions if r[2]]
            print(f"Region table: {len(regions)} regions, {len(enc)} encrypted:",
                  regions)
        except Ps3Error as e:
            print("Region table: could not parse --", e)
        print_disc_info(identify_disc(make_reader(dev)))




# --------------------------------------------------------------- hashing ---
HASH_KEYS = ("size", "crc32", "md5", "sha1")


class MultiHash:
    """CRC32 + MD5 + SHA-1 of one byte stream (the set Redump publishes)."""

    def __init__(self):
        self.size, self.crc = 0, 0
        self.md5, self.sha1 = hashlib.md5(), hashlib.sha1()

    def update(self, data):
        self.size += len(data)
        self.crc = zlib.crc32(data, self.crc)
        self.md5.update(data)
        self.sha1.update(data)

    def result(self) -> dict:
        return {"size": str(self.size), "crc32": f"{self.crc:08x}",
                "md5": self.md5.hexdigest(), "sha1": self.sha1.hexdigest()}


class FileHasher:
    """MD5 per file, fed chunk by chunk. `files` = [(path, lba, size)] by lba."""

    def __init__(self, files):
        self.files = files
        self.starts = [f[1] * SECTOR for f in files]
        self.md5 = [hashlib.md5() for _ in files]

    def feed(self, first: int, view: memoryview):
        a = first * SECTOR
        b = a + len(view)
        i = max(bisect.bisect_right(self.starts, a) - 1, 0)
        while i < len(self.files) and self.starts[i] < b:
            fs = self.starts[i]
            lo, hi = max(fs, a), min(fs + self.files[i][2], b)
            if lo < hi:
                self.md5[i].update(view[lo - a:hi - a])
            i += 1

    def digests(self):
        return [m.digest() for m in self.md5]


def write_hashes(path, hashes: dict):
    with open(path, "w") as f:
        f.write("# ps3_dump hashes. raw = disc exactly as read (encrypted; what "
                "Redump lists), iso = decrypted output\n")
        for prefix, h in hashes.items():
            for k in HASH_KEYS:
                f.write(f"{prefix}.{k} {h[k]}\n")


def read_hashes(path) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and not parts[0].startswith("#"):
                prefix, _, field = parts[0].partition(".")
                out.setdefault(prefix, {})[field] = parts[1].lower()
    return out


def print_hashes(label, h):
    pad = " " * len(label)
    print(f"{label}  size  {h['size']}\n{pad}  crc32 {h['crc32']}\n"
          f"{pad}  md5   {h['md5']}\n{pad}  sha1  {h['sha1']}")


# ---------------------------------------------------------------- redump ---
def load_redump(dat=None, sha1=None, md5=None, crc32=None) -> list:
    """Expected Redump entries from a .dat (XML) and/or command-line hashes."""
    entries = []
    if dat:
        try:
            root = ET.parse(dat).getroot()
        except (OSError, ET.ParseError) as e:
            raise Ps3Error(f"Cannot read Redump dat {dat}: {e}")
        for game in root.iter("game"):
            for rom in game.iter("rom"):
                if not (rom.get("name") or "").lower().endswith(".iso"):
                    continue
                e = {"name": game.get("name") or rom.get("name")}
                for attr, key in (("size", "size"), ("crc", "crc32"),
                                  ("md5", "md5"), ("sha1", "sha1")):
                    if rom.get(attr):
                        e[key] = rom.get(attr).lower()
                entries.append(e)
        if not entries:
            raise Ps3Error(f"{dat}: no .iso entries found")
    given = {k: v.lower().removeprefix("0x") for k, v in
             (("sha1", sha1), ("md5", md5), ("crc32", crc32)) if v}
    if given:
        entries.append({"name": "command line", **given})
    return entries


def compare_redump(raw: dict, entries):
    for e in entries:
        fields = [k for k in HASH_KEYS if k in e]
        if fields and all(raw[k] == e[k] for k in fields):
            return PASS, f"matches '{e['name']}' ({', '.join(fields)})"
    same_size = [e for e in entries if e.get("size") == raw["size"]]
    hint = (f"; '{same_size[0]['name']}' has the same size but other hashes "
            f"(bad read, or a different revision)" if same_size else "")
    return FAIL, f"raw sha1 {raw['sha1']} matches no Redump entry{hint}"


def redump_args(a) -> list:
    return load_redump(a.redump_dat, a.redump_sha1, a.redump_md5, a.redump_crc32)


# ------------------------------------------------------------ verification -
def compare_ird_files(files, digests, ird_files):
    bad = [f[0] for f, d in zip(files, digests) if ird_files.get(f[1]) != d]
    missing = len(set(ird_files) - {f[1] for f in files})
    if not files:
        return WARN, "no IRD file entries matched files on the disc"
    note = f"; {missing} IRD entries not checked" if missing else ""
    if bad:
        return FAIL, f"{len(bad)}/{len(files)} files differ, e.g. {bad[:3]}{note}"
    return PASS, f"all {len(files)} file MD5s match{note}"


def print_summary(checks):
    if not checks:
        return
    print("\nSummary")
    w = max(len(c[0]) for c in checks)
    for name, status, detail in checks:
        print(f"  {status:4}  {name:<{w}}  {detail}")


class Progress:
    def __init__(self, label, total, start=0):
        self.label, self.total, self.start = label, max(total, 1), start
        self.t0, self.last = time.time(), 0.0

    def update(self, done, extra=""):
        now = time.time()
        if now - self.last < 0.5 and done < self.total:
            return
        self.last = now
        rate = (done - self.start) * SECTOR / max(now - self.t0, 1e-6)
        eta = (self.total - done) * SECTOR / max(rate, 1)
        print(f"\r{self.label}: {done * 100 // self.total:3d}%  {rate / 1e6:6.1f} MB/s  "
              f"ETA {int(eta // 60)}m{int(eta % 60):02d}s  {extra}   ",
              end="", flush=True)

    def done(self):
        print()


def iter_file_chunks(f, start, end):
    """Yield (sector, view) over sectors [start, end) of an open image file."""
    buf = bytearray(CHUNK_SECTORS * SECTOR)
    sector = start
    f.seek(start * SECTOR)
    while sector < end:
        n = min(CHUNK_SECTORS, end - sector)
        got = read_into(f, sector, memoryview(buf)[:n * SECTOR])
        got -= got % SECTOR
        if not got:
            return
        yield sector, memoryview(buf)[:got]
        sector += got // SECTOR


def scan_iso(iso_path, ird=None):
    """One pass over a decrypted ISO: hashes plus IRD region / file MD5s."""
    with open(iso_path, "rb", buffering=0) as f:
        regions = parse_regions(read_at(f, 0, 1))
        total = os.fstat(f.fileno()).st_size // SECTOR
        files = []
        if ird and ird["file_hashes"]:
            pvd = read_at(f, 16, 1)
            if pvd[:6] == ISO_MAGIC:
                files = [x for x in walk_iso(lambda s, n: read_at(f, s, n), pvd)
                         if x[1] in ird["file_hashes"]]
        mh, rh, fh = MultiHash(), RegionHasher(regions), FileHasher(files)
        prog = Progress("Verifying", total)
        for sector, view in iter_file_chunks(f, 0, total):
            mh.update(view)
            rh.feed(sector, view)
            fh.feed(sector, view)
            prog.update(sector + len(view) // SECTOR)
        prog.done()
    return {"regions": regions, "iso": mh.result(), "region_md5": rh.digests(),
            "files": files, "file_md5": fh.digests()}


def ird_checks(ird, regions, region_md5, files, file_md5):
    out = []
    if region_md5 is not None:
        out.append(("IRD region MD5s", *compare_ird_hashes(region_md5, ird["region_hashes"], regions)))
    if ird["file_hashes"]:
        out.append(("IRD file MD5s", *compare_ird_files(files, file_md5, ird["file_hashes"])))
    return out


def verify_cmd(iso_path, ird=None, redump=None, hashes_path=None):
    """Full check of a finished ISO against everything we have."""
    res = scan_iso(iso_path, ird)
    print_hashes("Decrypted ISO", res["iso"])
    checks = []
    hashes_path = hashes_path or iso_path + ".hashes.txt"
    saved = read_hashes(hashes_path) if os.path.exists(hashes_path) else {}
    if "iso" in saved:
        same = saved["iso"] == res["iso"]
        checks.append(("Dump-time hashes", PASS if same else FAIL,
                       "ISO unchanged since it was dumped" if same else
                       f"ISO differs from {hashes_path} -- file corrupted or modified"))
    if ird:
        checks += ird_checks(ird, res["regions"], res["region_md5"],
                             res["files"], res["file_md5"])
    if redump:
        if "raw" in saved:
            checks.append(("Redump", *compare_redump(saved["raw"], redump)))
        else:
            checks.append(("Redump", SKIP, "Redump hashes cover the encrypted disc; "
                           "they're recorded by `dump` in the .hashes.txt file"))
    if not checks:
        checks.append(("Verification", WARN, "nothing to compare against: pass "
                       "--ird, --redump-dat or keep the .hashes.txt from the dump"))
    print_summary(checks)
    if any(c[1] == FAIL for c in checks):
        raise Ps3Error("Verification failed")


# ---------------------------------------------------------- spot checks ----
MAGIC_BY_EXT = {".SELF": b"SCE\0", ".SPRX": b"SCE\0", ".EDAT": b"NPD\0",
                ".SDAT": b"NPD\0", ".SFO": b"\0PSF", ".PNG": b"\x89PNG",
                ".PAM": b"PAMF", ".PMF": b"PAMF", ".AT3": b"RIFF"}


def _magic_for(path):
    base = path.rsplit("/", 1)[-1]
    if base == "EBOOT.BIN":
        return b"SCE\0"
    return MAGIC_BY_EXT.get(os.path.splitext(base)[1])


def sector_entropy(data) -> float:
    n = len(data)
    return -sum(c / n * math.log2(c / n) for c in collections.Counter(data).values())


def spot_check(dev, regions, key, ird=None, samples=48):
    """Quick key check on the encrypted regions, before the long dump.

    1. IRD MD5s of small files in encrypted regions (definitive).
    2. Known magics (SCE\\0, NPD\\0, \\0PSF, PNG, ...) at encrypted file starts.
    3. Entropy of sampled decrypted sectors: a wrong key turns everything into
       noise (~7.9 bits/byte); real data usually has some structure.
    """
    enc = [r for r in regions if r[2]]
    if not enc:
        return SKIP, "disc has no encrypted regions"
    read = make_reader(dev, regions, key)
    files = []
    pvd = read(16, 1)
    if pvd[:6] == ISO_MAGIC:
        try:
            files = walk_iso(read, pvd)
        except (struct.error, ValueError, IndexError):
            pass
    enc_files = [f for f in files if f[2] and is_encrypted(f[1], regions)]

    ird_files = (ird or {}).get("file_hashes") or {}
    cands = sorted((f for f in enc_files if f[1] in ird_files and f[2] <= 4 << 20),
                   key=lambda f: f[2])[:3]
    ird_note = None
    if cands:
        for path, lba, size in cands:
            if hashlib.md5(read(lba, -(-size // SECTOR))[:size]).digest() == ird_files[lba]:
                return PASS, f"IRD MD5 matches for {path} (definitive)"
        ird_note = f"IRD MD5 mismatch on {len(cands)} encrypted files"

    hits = checked = 0
    for path, lba, size in enc_files:
        magic = _magic_for(path)
        if not magic or size < len(magic):
            continue
        checked += 1
        hits += read(lba, 1)[:len(magic)] == magic
        if checked >= 8:
            break

    rng = random.Random(0)
    total_enc = sum(e - s + 1 for s, e, _ in enc)
    low = tested = 0
    for _ in range(samples):
        k = rng.randrange(total_enc)
        for s, e, _ in enc:
            if k <= e - s:
                sec = s + k
                break
            k -= e - s + 1
        raw = read_at(dev, sec, 1)
        if len(raw) != SECTOR or raw == ZERO_SECTOR:
            continue
        tested += 1
        low += sector_entropy(decrypt_sector(key, sec, raw)) < 7.5

    detail = (f"file magics {hits}/{checked}, structured sectors {low}/{tested}"
              + (f"; {ird_note}" if ird_note else ""))
    if hits or low:
        return (WARN if ird_note else PASS), detail
    if checked >= 2 or ird_note:
        return FAIL, detail + " -- decrypted data is noise, key looks wrong"
    return WARN, detail + " -- not enough evidence either way"


def check_key(device, key, ird=None):
    with open_device(device) as dev:
        regions = parse_regions(read_at(dev, 0, 1))
        status, detail = spot_check(dev, regions, key, ird)
    print(f"Key spot check: {status} -- {detail}")
    if status == FAIL:
        raise Ps3Error("Key spot check failed")


# ----------------------------------------------------------- read-ahead ----
def iter_chunks(dev, start, total, depth=4):
    """Yield (sector, view, bad_sectors) while a thread reads ahead.

    Disc reads block in the kernel, so overlapping them with decryption and
    hashing keeps the drive streaming instead of idling between chunks.
    """
    free, full = queue.Queue(), queue.Queue()
    for _ in range(depth):
        free.put(bytearray(CHUNK_SECTORS * SECTOR))
    stop = threading.Event()

    def worker():
        try:
            s = start
            while s < total:
                buf = free.get()
                if stop.is_set():
                    break
                n = min(CHUNK_SECTORS, total - s)
                full.put((s, buf, n, read_chunk(dev, s, memoryview(buf)[:n * SECTOR])))
                s += n
            full.put(None)
        except BaseException as e:               # hand errors to the consumer
            full.put(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        while True:
            item = full.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            s, buf, n, bad = item
            yield s, memoryview(buf)[:n * SECTOR], bad
            free.put(buf)
    finally:
        stop.set()
        free.put(bytearray(0))                   # wake a worker waiting on a buffer
        t.join()


# ----------------------------------------------------------------- dump ----
def dump(device, key, out_path, ird=None, limit=None, sectors=None,
         force=False, resume=False, redump=None, spot=True,
         full_verify=False, hashing=True):
    part = out_path + ".part"
    if os.path.exists(out_path) and not force:
        raise Ps3Error(f"{out_path} exists (use --force to overwrite)")
    start = 0
    if resume and os.path.exists(part):
        start = os.path.getsize(part) // SECTOR
    elif os.path.exists(part) and not force:
        raise Ps3Error(f"{part} exists from an earlier run; use --resume to "
                       f"continue it or --force to start over")
    checks, bad, raw_bad = [], [], []
    with open_device(device) as dev:
        disc_total = device_sectors(dev, sectors)
        total = min(disc_total, limit) if limit else disc_total
        complete = total == disc_total
        start = min(start, total)
        free = shutil.disk_usage(os.path.dirname(os.path.abspath(out_path))).free
        if free < (total - start) * SECTOR:
            raise Ps3Error(f"Need {(total - start) * SECTOR / 1e9:.1f} GB free, "
                           f"have {free / 1e9:.1f} GB")

        s0 = read_at(dev, 0, 1)
        try:
            regions = parse_regions(s0)
            if ird and ird["header"][:SECTOR] != s0:
                print("WARNING: disc sector 0 differs from the IRD header -- "
                      "the IRD may be for another version of this game")
        except Ps3Error:
            if not ird:
                raise
            print("Disc sector 0 unreadable/invalid; using the IRD's region table")
            regions = parse_regions(ird["header"][:SECTOR])
        n_enc = sum(1 for r in regions if r[2])
        print(f"{total} sectors, {len(regions)} regions, {n_enc} encrypted")

        reader = make_reader(dev, regions, key)
        disc = identify_disc(reader)
        print_disc_info(disc)
        if ird and disc["title_id"] and \
                normalize_title_id(disc["title_id"]) != normalize_title_id(ird["game_id"]):
            raise Ps3Error(f"Disc is {disc['title_id']} but the IRD is for "
                           f"{ird['game_id']}; wrong IRD/key for this disc")

        if spot:
            status, detail = spot_check(dev, regions, key, ird)
            print(f"Key spot check: {status} -- {detail}")
            checks.append(("Key spot check", status, detail))
            if status == FAIL:
                raise Ps3Error("Key spot check failed: the encrypted regions "
                               "decrypt to noise. Wrong key, or wrong "
                               "--key-type? (--no-spot-check to dump anyway)")

        files = []
        if hashing and ird and ird["file_hashes"]:
            pvd = reader(16, 1)
            if pvd[:6] == ISO_MAGIC:
                files = [f for f in walk_iso(reader, pvd) if f[1] in ird["file_hashes"]
                         and f[1] * SECTOR + f[2] <= total * SECTOR]
        raw_h, iso_h = MultiHash(), MultiHash()
        region_h, file_h = RegionHasher(regions), FileHasher(files)

        def feed_output(sector, view):
            iso_h.update(view)
            region_h.feed(sector, view)
            file_h.feed(sector, view)

        if start and hashing:                    # rebuild hash state for resume
            with open(part, "rb", buffering=0) as pf:
                prog = Progress("Re-hashing partial dump", start)
                for s, view in iter_file_chunks(pf, 0, start):
                    feed_output(s, view)
                    prog.update(s + len(view) // SECTOR)
                prog.done()
            prog = Progress("Re-reading disc for raw hashes", start)
            for s, view, b in iter_chunks(dev, 0, start):
                raw_bad += b
                raw_h.update(view)
                prog.update(s + len(view) // SECTOR)
            prog.done()

        aes, index = algorithms.AES(key), EncryptedIndex(regions)
        if start:
            print(f"Resuming at sector {start} ({start * 100 // max(total, 1)}%)")
        with open(part, "r+b" if start else "wb") as out:
            out.truncate(start * SECTOR)
            out.seek(start * SECTOR)
            prog = Progress("Dumping", total, start)
            for s, view, b in iter_chunks(dev, start, total):
                bad += b
                if hashing:
                    raw_h.update(view)           # before decryption = Redump hash
                decrypt_chunk(aes, view, s, index)
                if hashing:
                    feed_output(s, view)
                out.write(view)
                prog.update(s + len(view) // SECTOR, f"bad {len(bad)}")
            prog.done()
    os.replace(part, out_path)

    if bad:
        bad_list = out_path + ".bad-sectors.txt"
        with open(bad_list, "w") as f:
            f.write("\n".join(map(str, bad)) + "\n")
        checks.append(("Readable sectors", FAIL, f"{len(bad)} unreadable sectors "
                       f"zero-filled (first {bad[:5]}, list in {bad_list}); "
                       f"clean the disc and retry"))
    if total > 16:
        with open(out_path, "rb") as f:
            f.seek(16 * SECTOR)
            ok = f.read(6) == ISO_MAGIC
        checks.append(("ISO 9660 header", PASS if ok else FAIL,
                       "present" if ok else "missing -- bad read or wrong regions"))

    hashes = {}
    if hashing:
        hashes["iso"] = iso_h.result()
        print_hashes("Decrypted ISO", hashes["iso"])
        if complete and not bad and not raw_bad:
            hashes["raw"] = raw_h.result()
            print_hashes("Raw disc (Redump)", hashes["raw"])
        write_hashes(out_path + ".hashes.txt", hashes)
        print(f"Hashes saved to {out_path}.hashes.txt")

    if redump:
        if "raw" in hashes:
            checks.append(("Redump", *compare_redump(hashes["raw"], redump)))
        else:
            checks.append(("Redump", SKIP, "needs a complete dump with no bad "
                           "sectors and hashing on"))
    if ird and hashing:
        checks += ird_checks(ird, regions, region_h.digests() if complete else None,
                             files, file_h.digests())

    if full_verify:
        res = scan_iso(out_path, ird)
        if hashing:
            same = res["iso"] == hashes["iso"]
            checks.append(("Re-read of output", PASS if same else FAIL,
                           "ISO on disk matches what was written" if same else
                           "ISO on disk differs from what was written -- storage problem?"))
        elif ird:
            checks += ird_checks(ird, res["regions"], res["region_md5"] if complete
                                 else None, res["files"], res["file_md5"])

    if not redump and not ird:
        checks.append(("Key proof", SKIP, "pass --ird or --redump-dat for a "
                       "hash check, or load the ISO in RPCS3"))
    print_summary(checks)
    if any(c[1] == FAIL for c in checks):
        raise Ps3Error("Dump finished with failed checks (see summary)")
    print("Done:", out_path)


# ------------------------------------------------------------- selftest ----
def _make_test_disc():
    """Synthetic 64-sector disc: plain 0-29, encrypted 30-49, plain 50-63."""
    hdr = (struct.pack(">II", 2, 0) + struct.pack(">IIII", 0, 29, 50, 63)).ljust(SECTOR, b"\0")
    original = [hdr] + [os.urandom(SECTOR) for _ in range(63)]
    original[40] = ZERO_SECTOR                   # zero sectors stay unencrypted

    def dirrec(name, lba, size, is_dir=False):
        rec = bytes([0, 0]) + lba.to_bytes(4, "little") + lba.to_bytes(4, "big") \
            + size.to_bytes(4, "little") + size.to_bytes(4, "big") + bytes(7) \
            + bytes([2 if is_dir else 0, 0, 0]) + (1).to_bytes(2, "little") \
            + (1).to_bytes(2, "big") + bytes([len(name)]) + name
        rec += b"\0" * (len(rec) % 2)
        return bytes([len(rec)]) + rec[1:]

    root = dirrec(b"\0", 20, SECTOR, True) + dirrec(b"\1", 20, SECTOR, True) \
        + dirrec(b"PS3_DISC.SFB;1", 22, 0x300) + dirrec(b"PS3_GAME", 21, SECTOR, True)
    game = dirrec(b"\0", 21, SECTOR, True) + dirrec(b"\1", 20, SECTOR, True) \
        + dirrec(b"PARAM.SFO;1", 35, 64) + dirrec(b"EBOOT.BIN;1", 36, 3 * SECTOR - 100)
    sfb = bytearray(0x300)
    sfb[0:4] = b".SFB"
    sfb[0x20:0x28] = b"TITLE_ID"
    sfb[0x30:0x38] = struct.pack(">II", 0x220, 10)
    sfb[0x220:0x22a] = b"BLUS-00000"
    sfo = bytearray(64)
    sfo[0:4] = b"\0PSF"
    sfo[8:20] = struct.pack("<III", 36, 44, 1)
    sfo[20:36] = struct.pack("<HHIII", 0, 0x0204, 9, 16, 0)
    sfo[36:42] = b"TITLE\0"
    sfo[44:53] = b"Test Game"

    pvd = bytearray(os.urandom(SECTOR))
    pvd[0:7] = ISO_MAGIC + b"\x01"
    pvd[40:72] = PS3_VOLUME_ID.encode().ljust(32)
    pvd[156:190] = dirrec(b"\0", 20, SECTOR, True)
    original[16] = bytes(pvd)
    original[20] = root.ljust(SECTOR, b"\0")
    original[21] = game.ljust(SECTOR, b"\0")
    original[22] = bytes(sfb).ljust(SECTOR, b"\0")
    original[35] = bytes(sfo).ljust(SECTOR, b"\0")    # inside the encrypted region
    original[36] = b"SCE\0" + original[36][4:]        # EBOOT.BIN, encrypted too
    return hdr, original


def _make_test_ird(hdr, region_md5, file_md5, key):
    hdr_c = gzip.compress(hdr)
    body = b"3IRD" + bytes([9]) + b"BLUS00000" + bytes([4]) + b"Test"
    body += b"0100" + b"01.00" + b"01.00"
    body += struct.pack("<I", len(hdr_c)) + hdr_c + struct.pack("<I", 0)
    body += bytes([len(region_md5)]) + b"".join(region_md5)
    body += struct.pack("<I", len(file_md5))
    body += b"".join(struct.pack("<Q", lba) + md5 for lba, md5 in file_md5.items())
    body += b"\0\0\0\0" + b"\0" * 115 + key + b"\0" * 16
    return gzip.compress(body)


def _expect_fail(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except Ps3Error:
        return
    raise AssertionError(f"{fn.__name__} should have failed")


def selftest():
    key = bytes(range(16))
    plain = os.urandom(SECTOR)
    assert decrypt_sector(key, 7, encrypt_sector(key, 7, plain)) == plain
    assert encrypt_sector(key, 7, plain) != encrypt_sector(key, 8, plain)

    # d1 derivation = one CBC block: AES-ECB(k, d1 XOR iv). Dummy constants.
    sk, siv, d1 = bytes(range(16, 32)), bytes(range(32, 48)), os.urandom(16)
    ecb = Cipher(algorithms.AES(sk), modes.ECB()).encryptor()
    assert derive_disc_key(d1, sk, siv) == ecb.update(
        bytes(a ^ b for a, b in zip(d1, siv))) + ecb.finalize()

    m = MultiHash()
    m.update(b"abc")
    assert m.result() == {"size": "3", "crc32": "352441c2",
                          "md5": "900150983cd24fb0d6963f7d28e17f72",
                          "sha1": "a9993e364706816aba3e25717850c26c9cd0d89d"}
    assert sector_entropy(bytes(SECTOR)) == 0 and sector_entropy(os.urandom(SECTOR)) > 7.8

    hdr, original = _make_test_disc()
    regions = parse_regions(hdr)
    assert regions == [(0, 29, False), (30, 49, True), (50, 63, False)], regions
    assert not is_encrypted(29, regions) and is_encrypted(30, regions)
    assert not is_encrypted(50, regions)
    assert list(EncryptedIndex(regions).ranges(0, 64)) == [(30, 50)]
    assert list(EncryptedIndex(regions).ranges(32, 40)) == [(32, 40)]
    assert list(EncryptedIndex(regions).ranges(50, 64)) == []
    image = b"".join(encrypt_sector(key, s, original[s])
                     if is_encrypted(s, regions) and original[s] != ZERO_SECTOR
                     else original[s] for s in range(64))
    expected = b"".join(original)
    raw_sha1 = hashlib.sha1(image).hexdigest()
    region_md5 = [hashlib.md5(expected[s * SECTOR:(e + 1) * SECTOR]).digest()
                  for s, e, _ in regions]
    eboot_size = 3 * SECTOR - 100
    file_md5 = {35: hashlib.md5(original[35][:64]).digest(),
                36: hashlib.md5(expected[36 * SECTOR:36 * SECTOR + eboot_size]).digest()}

    with tempfile.TemporaryDirectory() as td:
        img, outp = os.path.join(td, "disc.img"), os.path.join(td, "out.iso")
        with open(img, "wb") as f:
            f.write(image)

        kp = os.path.join(td, "k.key")
        with open(kp, "wb") as f:
            f.write(key)
        assert load_key_file(kp) == key
        with open(kp, "w") as f:
            f.write(key.hex() + "\n")
        assert load_key_file(kp) == key
        with open(kp, "w") as f:
            f.write((sk + siv).hex(" ") + "\n")
        assert load_d1_secrets(kp) == (sk, siv)
        _expect_fail(load_d1_secrets, None)

        ip = os.path.join(td, "t.ird")
        with open(ip, "wb") as f:
            f.write(_make_test_ird(hdr, region_md5, file_md5, key))
        ird = read_ird(ip)
        assert ird["data1"] == key and ird["game_id"] == "BLUS00000", ird
        assert ird["file_hashes"] == file_md5
        assert parse_regions(ird["header"][:SECTOR]) == regions

        with open_device(img) as dev:            # identification + spot checks
            assert device_sectors(dev) == 64
            info = identify_disc(make_reader(dev))
            assert info["volume_id"] == PS3_VOLUME_ID, info
            assert info["title_id"] == "BLUS-00000" and info["title"] is None, info
            reader = make_reader(dev, regions, key)
            assert identify_disc(reader)["title"] == "Test Game"
            assert [f[0] for f in walk_iso(reader, reader(16, 1))] == \
                ["PS3_DISC.SFB", "PS3_GAME/PARAM.SFO", "PS3_GAME/EBOOT.BIN"]
            assert spot_check(dev, regions, key)[0] == PASS
            assert "definitive" in spot_check(dev, regions, key, ird)[1]
            wrong = bytes(16)
            assert spot_check(dev, regions, wrong)[0] == FAIL
            assert spot_check(dev, regions, wrong, ird)[0] == FAIL
            got = b"".join(bytes(v) for _, v, _ in iter_chunks(dev, 0, 64))
            assert got == image

        dat = os.path.join(td, "redump.dat")
        with open(dat, "w") as f:
            f.write(f'<?xml version="1.0"?><datafile><game name="Test Game (USA)">'
                    f'<rom name="Test Game (USA).iso" size="{len(image)}" '
                    f'crc="{zlib.crc32(image):08x}" md5="{hashlib.md5(image).hexdigest()}" '
                    f'sha1="{raw_sha1}"/></game></datafile>')
        redump = load_redump(dat)
        assert redump[0]["name"] == "Test Game (USA)" and redump[0]["sha1"] == raw_sha1

        dump(img, key, outp, redump=redump, full_verify=True)   # end-to-end
        with open(outp, "rb") as f:
            assert f.read() == expected, "dump mismatch"
        saved = read_hashes(outp + ".hashes.txt")
        assert saved["raw"]["sha1"] == raw_sha1
        assert saved["iso"]["sha1"] == hashlib.sha1(expected).hexdigest()
        _expect_fail(dump, img, key, outp)                      # no overwrite
        _expect_fail(dump, img, key, outp, force=True,          # Redump mismatch
                     redump=load_redump(sha1="00" * 20))
        _expect_fail(dump, img, bytes(16), outp, force=True)    # wrong key

        os.remove(outp)                         # interrupted dump, then resume
        dump(img, key, outp, limit=37)
        os.replace(outp, outp + ".part")
        dump(img, key, outp, resume=True, ird=ird, redump=redump)
        with open(outp, "rb") as f:
            assert f.read() == expected, "resume mismatch"
        assert read_hashes(outp + ".hashes.txt") == saved, "resume hashes differ"

        dump(img, key, outp, ird=ird, force=True, full_verify=True)
        verify_cmd(outp, ird, redump)
        res = scan_iso(outp, ird)
        assert res["region_md5"] == region_md5 and len(res["files"]) == 2
        with open(outp, "r+b") as f:                            # corrupt EBOOT
            f.seek(37 * SECTOR)
            f.write(b"X")
        _expect_fail(verify_cmd, outp, ird)
    print("selftest OK")


# ------------------------------------------------------------------ main ---
def add_redump_args(sp):
    sp.add_argument("--redump-dat", help="Redump .dat (XML) to compare against")
    sp.add_argument("--redump-sha1")
    sp.add_argument("--redump-md5")
    sp.add_argument("--redump-crc32")


def resolve_key(a, ird):
    if a.key_file:
        key = load_key_file(a.key_file)
    elif a.key_type == "d1" and ird:
        key = ird["data1"]
        print("Using d1 from the IRD")
    else:
        raise Ps3Error("--key-file is required (or --ird with --key-type d1)")
    if a.key_type == "d1":
        key = derive_disc_key(key, *load_d1_secrets(a.d1_secrets))
    return key


def main():
    ap = argparse.ArgumentParser(description="PS3 disc dump tool")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("drives")
    p = sub.add_parser("probe")
    i = sub.add_parser("ird-info"); i.add_argument("ird")
    d = sub.add_parser("dump")
    c = sub.add_parser("check-key", help="spot-check a key on the disc, no dump")
    for sp in (p, d, c):
        sp.add_argument("--device", default="auto",
                        help="drive or image path (default: auto-detect)")
        sp.add_argument("--wait", type=float, default=0,
                        help="seconds to wait for a disc when auto-detecting")
    for sp in (d, c):
        sp.add_argument("--key-file",
                        help="disc key or d1 (optional for d1 when --ird is given)")
        sp.add_argument("--key-type", choices=["final", "d1"], required=True)
        sp.add_argument("--d1-secrets", default=os.environ.get("PS3_D1_SECRETS"),
                        help="file with the d1 derivation key + IV (32 bytes / 64 hex)")
        sp.add_argument("--ird")
    d.add_argument("--out", required=True)
    d.add_argument("--limit-sectors", type=int)
    d.add_argument("--sectors", type=int, help="override detected disc size")
    d.add_argument("--force", action="store_true")
    d.add_argument("--resume", action="store_true",
                   help="continue an interrupted dump from OUT.part")
    d.add_argument("--verify", action="store_true",
                   help="re-read the finished ISO and check it end to end")
    d.add_argument("--no-spot-check", action="store_true",
                   help="skip the key check on encrypted regions before dumping")
    d.add_argument("--no-hash", action="store_true",
                   help="skip CRC32/MD5/SHA-1 hashing")
    add_redump_args(d)
    v = sub.add_parser("verify", help="full integrity check of a finished ISO")
    v.add_argument("iso")
    v.add_argument("--ird")
    v.add_argument("--hashes", help="hash file from dump (default ISO.hashes.txt)")
    add_redump_args(v)
    a = ap.parse_args()

    try:
        if a.cmd == "selftest":
            selftest()
        elif a.cmd == "drives":
            drives = list_drives()
            print("\n".join(describe_drive(x) for x in drives)
                  or "No optical drives found.")
        elif a.cmd == "probe":
            probe(choose_device(a.device, a.wait))
        elif a.cmd == "ird-info":
            info = read_ird(a.ird)
            print(info["game_id"], info["name"], f"(IRD v{info['version']})")
            print("regions:", parse_regions(info["header"][:SECTOR]))
            print(f"{len(info['region_hashes'])} region hashes, "
                  f"{len(info['file_hashes'])} file hashes")
        elif a.cmd == "verify":
            verify_cmd(a.iso, read_ird(a.ird) if a.ird else None,
                       redump_args(a), a.hashes)
        elif a.cmd == "check-key":
            ird = read_ird(a.ird) if a.ird else None
            check_key(choose_device(a.device, a.wait), resolve_key(a, ird), ird)
        else:
            ird = read_ird(a.ird) if a.ird else None
            key = resolve_key(a, ird)
            dump(choose_device(a.device, a.wait), key, a.out, ird,
                 a.limit_sectors, a.sectors, a.force, a.resume,
                 redump=redump_args(a), spot=not a.no_spot_check,
                 full_verify=a.verify, hashing=not a.no_hash)
    except Ps3Error as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        sys.exit("\ninterrupted (re-run with --resume to continue)")


if __name__ == "__main__":
    main()
