"""Register a command with the OS scheduler (Windows Task Scheduler or cron), so it runs every N hours
also when the Studio is closed. Each job is marked by its task name (the crontab line ends in
"# <task name>"), so several jobs (the news crawler, the code updates) live side by side."""
import os, subprocess


def _q(s):
    return f'"{s}"' if " " in s or os.name == "nt" else s


def status(task):
    try:
        if os.name == "nt":
            r = subprocess.run(["schtasks", "/Query", "/TN", task, "/FO", "LIST"], capture_output=True, text=True)
            return {"installed": r.returncode == 0, "kind": "Task Scheduler",
                    "detail": r.stdout.strip()[-400:] if r.returncode == 0 else ""}
        r = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.endswith(f"# {task}")), "")
        return {"installed": bool(line), "kind": "cron", "detail": line}
    except FileNotFoundError:
        return {"installed": False, "kind": "none", "detail": "no scheduler found"}


def install(task, cmd, log, every_hours, cwd, what="the job"):
    """Run `cmd` (a list) in `cwd` every `every_hours`, appending its output to `log`."""
    os.makedirs(os.path.dirname(log), exist_ok=True)
    hours = max(1, int(round(every_hours)))
    if os.name == "nt":
        tr = "cmd /c cd /d " + _q(cwd) + " && " + " ".join(_q(c) for c in cmd) + f" >> {_q(log)} 2>&1"
        args = ["schtasks", "/Create", "/F", "/TN", task, "/SC", "HOURLY", "/MO", str(hours), "/TR", tr]
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout).strip() or "schtasks failed")
        return f"Task Scheduler runs {what} every {hours} h, also when the Studio is closed."
    r = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    lines = [l for l in (r.stdout.splitlines() if r.returncode == 0 else []) if not l.endswith(f"# {task}")]
    sched = f"0 */{hours} * * *" if hours < 24 else "0 3 * * *"
    lines.append(f"{sched} cd {_q(cwd)} && " + " ".join(_q(c) for c in cmd) + f" >> {_q(log)} 2>&1  # {task}")
    r = subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip() or "crontab failed")
    return f"cron runs {what} every {hours} h, also when the Studio is closed."


def remove(task):
    if os.name == "nt":
        r = subprocess.run(["schtasks", "/Delete", "/F", "/TN", task], capture_output=True, text=True)
        if r.returncode != 0 and "cannot find" not in (r.stderr + r.stdout).lower():
            raise RuntimeError((r.stderr or r.stdout).strip())
        return "Removed from Task Scheduler."
    r = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    lines = [l for l in (r.stdout.splitlines() if r.returncode == 0 else []) if not l.endswith(f"# {task}")]
    subprocess.run(["crontab", "-"], input="\n".join(lines) + ("\n" if lines else ""), capture_output=True, text=True)
    return "Removed from cron."
