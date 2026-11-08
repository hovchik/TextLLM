"""Battery and heat checks, so a phone (or laptop) in the cluster only trains while it is charging
and cool. Standard library only; works in Termux on Android and on Linux laptops.

Android: `termux-battery-status` (the Termux:API app plus `pkg install termux-api`) gives charge,
plug and temperature; without it the kernel files under /sys/class/power_supply are tried, which
newer Android versions often hide. Laptops: /sys/class/power_supply (Linux only).

TEXTLLM_BATTERY_FILE points at a JSON file in the termux-battery-status format, for tests.
"""
import glob, json, os, shutil, subprocess


def is_android():
    return "ANDROID_ROOT" in os.environ or "com.termux" in os.environ.get("PREFIX", "")


def device_model():
    """'Pixel 7' on Android, else None."""
    if not is_android():
        return None
    try:
        r = subprocess.run(["getprop", "ro.product.model"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _from_sys(root="/sys/class/power_supply"):
    bats = [d for d in glob.glob(os.path.join(root, "*")) if _read(d, "type") == "Battery"] or \
        [d for d in glob.glob(os.path.join(root, "*")) if os.path.basename(d).lower().startswith(("bat", "battery"))]
    if not bats:
        return None
    b = bats[0]
    status = (_read(b, "status") or "").upper()
    cap = _read(b, "capacity")
    temp = _read(b, "temp")
    mains = any(_read(d, "online") == "1" for d in glob.glob(os.path.join(root, "*"))
                if _read(d, "type") in ("Mains", "USB", "USB_PD", "USB_C", "Wireless"))
    return {"percentage": int(cap) if cap and cap.isdigit() else None,
            "plugged": "PLUGGED" if mains or status in ("CHARGING", "FULL") else "UNPLUGGED",
            "status": status or None,
            "temperature": int(temp) / 10 if temp and temp.lstrip("-").isdigit() else None}


def _read(d, name):
    try:
        with open(os.path.join(d, name)) as f:
            return f.read().strip()
    except OSError:
        return None


def battery():
    """{"percentage", "plugged", "status", "temperature" (°C)} or None when there is no battery info."""
    fake = os.environ.get("TEXTLLM_BATTERY_FILE")
    if fake:
        try:
            return json.load(open(fake))
        except (OSError, ValueError):
            return None
    exe = shutil.which("termux-battery-status")
    if exe:
        try:
            r = subprocess.run([exe], capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip().startswith("{"):
                return json.loads(r.stdout)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass  # the Termux:API app is missing: the command hangs or fails
    return _from_sys()


def ok_to_train(b, min_percent=30, max_temp=42.0, need_charging=True):
    """(ok, reason) for a battery reading from battery()."""
    if b is None:
        return True, "battery state unknown"
    plugged = str(b.get("plugged") or "").upper()
    status = str(b.get("status") or "").upper()
    charging = (plugged.startswith("PLUGGED") and plugged != "UNPLUGGED") or status in ("CHARGING", "FULL")
    pct, temp = b.get("percentage"), b.get("temperature")
    if need_charging and not charging:
        return False, "paused: not charging (plug in the charger)"
    if pct is not None and pct < min_percent:
        return False, f"paused: battery at {pct}% (waits for {min_percent}%)"
    if temp is not None and temp > max_temp:
        return False, f"paused: battery at {temp:.0f} °C (cooling down below {max_temp:.0f} °C)"
    return True, f"charging, {pct}%" + (f", {temp:.0f} °C" if temp is not None else "") if pct is not None else "ok"


def keep_awake(on=True):
    """Termux: hold a wake lock so Android doesn't freeze the training when the screen goes off."""
    exe = shutil.which("termux-wake-lock" if on else "termux-wake-unlock")
    if exe:
        try:
            subprocess.Popen([exe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            pass
    return False
