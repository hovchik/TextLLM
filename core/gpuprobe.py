"""Find Intel Arc GPUs without PyTorch (standard library only, so run.py can use it before
anything is installed, and the Studio can say "an Arc card is here but this PyTorch can't use it").

PyTorch reaches Intel GPUs through its XPU backend (the `xpu` wheels from download.pytorch.org,
PyTorch 2.5+). It supports Arc A-series (Alchemist: A310/A380/A580/A750/A770 and the laptop
A370M/A550M/A730M/A770M), Arc B-series (Battlemage: B570/B580) and the "Intel Arc Graphics"
built into Core Ultra laptop chips (Meteor Lake, Lunar Lake, Arrow Lake). Older Intel
integrated graphics (UHD, Iris Xe) are not supported by PyTorch and stay on the CPU.
"""
import glob
import os
import re
import shutil
import subprocess

# PCI device ids (vendor 0x8086) of Arc GPUs, used on Linux when lspci isn't installed:
# 0x56xx Alchemist (DG2), 0xe20x/0xe21x Battlemage, 0x7d4x/0x7d5x/0x7d6x/0x7dxx Meteor/Arrow Lake,
# 0x64a0 Lunar Lake.
_ARC_IDS = re.compile(r"^0x(56[0-9a-f]{2}|e2[01][0-9a-f]|7d[0-9a-f]{2}|64a0)$")
_ARC_NAME = re.compile(r"\bArc\b|\bDG2\b|Alchemist|Battlemage|\bBMG\b", re.I)


def is_arc_name(name):
    return bool(_ARC_NAME.search(name or ""))


def _windows():
    """[(name, driver)] of the video adapters Windows knows."""
    exe = shutil.which("powershell") or shutil.which("pwsh")
    if not exe:
        return []
    cmd = "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.DriverVersion }"
    try:
        out = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-Command", cmd],
                             capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    rows = []
    for line in out.stdout.splitlines() if out.returncode == 0 else []:
        name, _, driver = line.strip().partition("|")
        if name:
            rows.append((name.strip(), driver.strip()))
    return rows


def _linux():
    rows = []
    exe = shutil.which("lspci")
    if exe:
        try:
            out = subprocess.run([exe, "-d", "8086:"], capture_output=True, text=True, timeout=20)
            for line in out.stdout.splitlines() if out.returncode == 0 else []:
                if re.search(r"VGA|Display|3D", line):
                    rows.append((line.split(": ", 1)[-1].strip(), ""))
        except Exception:
            pass
    if rows:
        return rows
    for dev in glob.glob("/sys/class/drm/card[0-9]*/device"):
        try:
            vendor = open(os.path.join(dev, "vendor")).read().strip()
            device = open(os.path.join(dev, "device")).read().strip().lower()
        except OSError:
            continue
        if vendor == "0x8086" and _ARC_IDS.match(device):
            rows.append((f"Intel Arc GPU ({device})", ""))
    return rows


def intel_arc_gpus():
    """[(name, driver_version)] of Intel Arc GPUs on this PC ([] if none, or on macOS)."""
    if os.name == "nt":
        found = _windows()
    elif os.path.isdir("/sys/class/drm") or shutil.which("lspci"):
        found = _linux()
    else:
        return []
    seen, out = set(), []
    for name, driver in found:
        if is_arc_name(name) and name not in seen:
            seen.add(name)
            out.append((name, driver))
    return out
