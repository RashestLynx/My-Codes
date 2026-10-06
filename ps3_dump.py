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
  - IRD layout (read_ird) and IRD region hashes are still UNVERIFIED -- a
    hash mismatch may mean the parser is wrong rather than the dump.

COMMANDS
  selftest                          offline checks, no disc needed
  drives                            list optical drives and whether a disc is in
  probe  [--device D] [--wait S]    identify the disc; no key needed
  ird-info game.ird                 inspect an IRD
  dump   [--device D] [--key-file K] --key-type final|d1 --out game.iso
         [--d1-secrets FILE]
         [--ird game.ird] [--limit-sectors N] [--sectors N] [--force]
         [--resume] [--wait S]
  verify --ird game.ird game.iso    check a finished ISO against IRD hashes

  --device defaults to "auto": the drive holding a PS3 disc is picked for
  you. Explicit examples: /dev/sr0 (Linux, may need sudo), \\.\E: (Windows,
  run as administrator), /dev/rdisk2 (macOS).
  Tip: run dump with --limit-sectors 4096 first as a quick trial.

Requires: pip install cryptography
"""
import argparse
import bisect
import glob
import gzip
import hashlib
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
import zlib

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

SECTOR = 2048
CHUNK_SECTORS = 512          # 1 MiB per read
ISO_MAGIC = b"\x01CD001"     # ISO 9660 primary volume descriptor, sector 16
PS3_VOLUME_ID = "PS3VOLUME"
RETRIES = 3
ZERO_SECTOR = bytes(SECTOR)


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


def compare_ird_hashes(digests, ird_hashes) -> bool:
    if len(digests) != len(ird_hashes):
        print(f"IRD hash check: skipped (IRD has {len(ird_hashes)} region "
              f"hashes, disc has {len(digests)} regions)")
        return False
    bad = [i for i, (a, b) in enumerate(zip(digests, ird_hashes)) if a != b]
    if not bad:
        print(f"IRD hash check: all {len(digests)} regions match -- dump and key are good")
        return True
    print(f"IRD hash check: {len(bad)}/{len(digests)} regions differ (indexes "
          f"{bad[:8]}). If only encrypted regions differ the key is likely "
          f"wrong; the IRD hash layout itself is also unverified.")
    return False


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
        fcount = struct.unpack("<I", data[pos:pos + 4])[0]
        pos += 4 + fcount * (8 + 16)
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
            "header": header, "region_hashes": hashes, "data1": data1}


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
    """Yield (NAME, lba, size, is_dir) for an ISO 9660 directory."""
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
                   int.from_bytes(rec[10:14], "little"), bool(rec[25] & 2))
        pos += ln


def _iso_find(read, pvd: bytes, path: str):
    """Return the bytes of `path` (e.g. 'PS3_GAME/PARAM.SFO'), or None."""
    lba = int.from_bytes(pvd[158:162], "little")
    size = int.from_bytes(pvd[166:170], "little")
    parts = path.upper().split("/")
    for i, part in enumerate(parts):
        for name, elba, esize, is_dir in _iso_dir(read, lba, size):
            if name == part and is_dir == (i < len(parts) - 1):
                lba, size = elba, esize
                break
        else:
            return None
    return read(lba, (min(size, 1 << 20) + SECTOR - 1) // SECTOR)[:size]


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


# ----------------------------------------------------------------- dump ----
def dump(device, key, out_path, ird=None, limit=None, sectors=None,
         force=False, resume=False):
    part = out_path + ".part"
    if os.path.exists(out_path) and not force:
        raise Ps3Error(f"{out_path} exists (use --force to overwrite)")
    start = 0
    if resume and os.path.exists(part):
        start = os.path.getsize(part) // SECTOR
    elif os.path.exists(part) and not force:
        raise Ps3Error(f"{part} exists from an earlier run; use --resume to "
                       f"continue it or --force to start over")
    bad = []
    with open_device(device) as dev:
        total = device_sectors(dev, sectors)
        if limit:
            total = min(total, limit)
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

        disc = identify_disc(make_reader(dev, regions, key))
        print_disc_info(disc)
        if ird and disc["title_id"] and \
                normalize_title_id(disc["title_id"]) != normalize_title_id(ird["game_id"]):
            raise Ps3Error(f"Disc is {disc['title_id']} but the IRD is for "
                           f"{ird['game_id']}; wrong IRD/key for this disc")

        aes = algorithms.AES(key)
        index = EncryptedIndex(regions)
        hasher = RegionHasher(regions) if (ird and start == 0 and not limit) else None
        buf = bytearray(CHUNK_SECTORS * SECTOR)
        mv = memoryview(buf)

        if start:
            print(f"Resuming at sector {start} ({start * 100 // max(total, 1)}%)")
        with open(part, "r+b" if start else "wb") as out:
            out.truncate(start * SECTOR)
            out.seek(start * SECTOR)
            sector, t0, last_print = start, time.time(), 0.0
            while sector < total:
                n = min(CHUNK_SECTORS, total - sector)
                view = mv[:n * SECTOR]
                bad += read_chunk(dev, sector, view)
                decrypt_chunk(aes, view, sector, index)
                if hasher:
                    hasher.feed(sector, view)
                out.write(view)
                sector += n
                now = time.time()
                if now - last_print >= 0.5 or sector == total:
                    last_print = now
                    rate = (sector - start) * SECTOR / max(now - t0, 1e-6)
                    eta = (total - sector) * SECTOR / max(rate, 1)
                    print(f"\r{sector * 100 // total:3d}%  {rate / 1e6:6.1f} MB/s  "
                          f"ETA {int(eta // 60)}m{int(eta % 60):02d}s  "
                          f"bad {len(bad)}   ", end="", flush=True)
        print()
    os.replace(part, out_path)

    if bad:
        bad_list = out_path + ".bad-sectors.txt"
        with open(bad_list, "w") as f:
            f.write("\n".join(map(str, bad)) + "\n")
        print(f"WARNING: {len(bad)} unreadable sectors zero-filled "
              f"(first: {bad[:5]}, full list in {bad_list}). Clean the disc and retry.")
    if total > 16:
        with open(out_path, "rb") as f:
            f.seek(16 * SECTOR)
            ok = f.read(6) == ISO_MAGIC
        print("ISO 9660 header check:", "OK" if ok else "FAILED -- bad read or wrong regions")
        if not ok:
            raise Ps3Error("Output failed the basic header check")
    if ird and not limit:
        digests = hasher.digests() if hasher else hash_iso_regions(out_path, regions)
        compare_ird_hashes(digests, ird["region_hashes"])
    else:
        print("Note: the ISO header sits in a plain region, so it does NOT prove "
              "the key is right. Pass --ird for a hash check, or load the ISO in RPCS3.")
    if bad:
        raise Ps3Error("Dump finished with unreadable sectors")
    print("Done:", out_path)


def hash_iso_regions(iso_path, regions):
    hasher = RegionHasher(regions)
    buf = bytearray(CHUNK_SECTORS * SECTOR)
    with open(iso_path, "rb", buffering=0) as f:
        sector = 0
        while True:
            n = f.readinto(buf)
            if not n:
                break
            n -= n % SECTOR
            hasher.feed(sector, memoryview(buf)[:n])
            sector += n // SECTOR
    return hasher.digests()


def verify(iso_path, ird):
    with open(iso_path, "rb") as f:
        regions = parse_regions(f.read(SECTOR))
    if not compare_ird_hashes(hash_iso_regions(iso_path, regions), ird["region_hashes"]):
        raise Ps3Error("Verification failed")


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
        + dirrec(b"PARAM.SFO;1", 35, 64)
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
    return hdr, original


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
        try:
            load_d1_secrets(None)
            raise AssertionError("missing secrets should fail")
        except Ps3Error:
            pass

        with open_device(img) as dev:            # identification, with/without key
            assert device_sectors(dev) == 64
            info = identify_disc(make_reader(dev))
            assert info["volume_id"] == PS3_VOLUME_ID, info
            assert info["title_id"] == "BLUS-00000" and info["title"] is None, info
            info = identify_disc(make_reader(dev, regions, key))
            assert info["title"] == "Test Game", info

        dump(img, key, outp)                    # end-to-end on a fake disc
        with open(outp, "rb") as f:
            assert f.read() == expected, "dump mismatch"
        try:
            dump(img, key, outp)
            raise AssertionError("should refuse to overwrite")
        except Ps3Error:
            pass

        os.remove(outp)                         # interrupted dump, then resume
        dump(img, key, outp, limit=37)
        os.replace(outp, outp + ".part")
        dump(img, key, outp, resume=True)
        with open(outp, "rb") as f:
            assert f.read() == expected, "resume mismatch"

        region_md5 = [hashlib.md5(expected[s * SECTOR:(e + 1) * SECTOR]).digest()
                      for s, e, _ in regions]
        assert hash_iso_regions(outp, regions) == region_md5

        hdr_c = gzip.compress(hdr)              # synthetic IRD (self-consistency only)
        body = b"3IRD" + bytes([9]) + b"BLUS00000" + bytes([4]) + b"Test"
        body += b"0100" + b"01.00" + b"01.00"
        body += struct.pack("<I", len(hdr_c)) + hdr_c + struct.pack("<I", 0)
        body += bytes([len(region_md5)]) + b"".join(region_md5)
        body += struct.pack("<I", 1) + b"\0" * 24
        body += b"\0\0\0\0" + b"\0" * 115 + key + b"\0" * 16
        ip = os.path.join(td, "t.ird")
        with open(ip, "wb") as f:
            f.write(gzip.compress(body))
        ird = read_ird(ip)
        assert ird["data1"] == key and ird["game_id"] == "BLUS00000", ird
        assert parse_regions(ird["header"][:SECTOR]) == regions
        verify(outp, ird)
        dump(img, key, outp, ird=ird, force=True)   # inline hashing path
    print("selftest OK")


# ------------------------------------------------------------------ main ---
def main():
    ap = argparse.ArgumentParser(description="PS3 disc dump tool")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("drives")
    p = sub.add_parser("probe")
    i = sub.add_parser("ird-info"); i.add_argument("ird")
    d = sub.add_parser("dump")
    for sp in (p, d):
        sp.add_argument("--device", default="auto",
                        help="drive or image path (default: auto-detect)")
        sp.add_argument("--wait", type=float, default=0,
                        help="seconds to wait for a disc when auto-detecting")
    d.add_argument("--key-file",
                   help="disc key or d1 (optional for d1 when --ird is given)")
    d.add_argument("--key-type", choices=["final", "d1"], required=True)
    d.add_argument("--d1-secrets", default=os.environ.get("PS3_D1_SECRETS"),
                   help="file with the d1 derivation key + IV (32 bytes / 64 hex)")
    d.add_argument("--ird")
    d.add_argument("--out", required=True)
    d.add_argument("--limit-sectors", type=int)
    d.add_argument("--sectors", type=int, help="override detected disc size")
    d.add_argument("--force", action="store_true")
    d.add_argument("--resume", action="store_true",
                   help="continue an interrupted dump from OUT.part")
    v = sub.add_parser("verify")
    v.add_argument("--ird", required=True)
    v.add_argument("iso")
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
        elif a.cmd == "verify":
            verify(a.iso, read_ird(a.ird))
        else:
            ird = read_ird(a.ird) if a.ird else None
            if a.key_file:
                key = load_key_file(a.key_file)
            elif a.key_type == "d1" and ird:
                key = ird["data1"]
                print("Using d1 from the IRD")
            else:
                raise Ps3Error("--key-file is required (or --ird with --key-type d1)")
            if a.key_type == "d1":
                key = derive_disc_key(key, *load_d1_secrets(a.d1_secrets))
            dump(choose_device(a.device, a.wait), key, a.out, ird,
                 a.limit_sectors, a.sectors, a.force, a.resume)
    except Ps3Error as e:
        sys.exit(f"error: {e}")
    except KeyboardInterrupt:
        sys.exit("\ninterrupted (re-run with --resume to continue)")


if __name__ == "__main__":
    main()
