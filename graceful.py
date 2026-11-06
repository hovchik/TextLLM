"""Stopping jobs cleanly on every platform.

The Studio runs training, learning, evaluation and crawling as child processes and stops them
with a signal. On Linux and macOS that is SIGINT to the job's process group. Windows has no
SIGINT for other processes, so the job is started in its own console process group and gets
CTRL_BREAK_EVENT, which Python delivers as SIGBREAK. `install()` turns that into the same
KeyboardInterrupt the code already handles, so a stopped trainer keeps its last checkpoint.

    import graceful; graceful.install()     # at the top of every script the Studio runs

`run_child` runs a sub-step (learn.py starting train.py) and, when interrupted, waits for that
child to finish its own shutdown instead of killing it at once.
"""
import os, signal, subprocess, sys, time


def install():
    if os.name == "nt" and hasattr(signal, "SIGBREAK"):
        def to_interrupt(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGBREAK, to_interrupt)


def popen_kwargs():
    """Extra Popen arguments that put the child in its own process group."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def interrupt(proc):
    """Ask a process started with popen_kwargs(), and everything it started, to stop."""
    try:
        if os.name == "nt":
            os.kill(proc.pid, signal.CTRL_BREAK_EVENT)   # the whole console process group
        else:
            os.killpg(proc.pid, signal.SIGINT)
    except (ProcessLookupError, PermissionError, OSError, ValueError):
        pass


def kill(proc):
    """Force-stop a process and its descendants."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError, ValueError, FileNotFoundError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def run_child(cmd, cwd, grace=120):
    """Like subprocess.call, except that on Ctrl-C / Stop the child gets time to save and exit.

    The signal already reached the child (same process group), so this only waits, ignoring
    repeated interrupts, and kills the child if it is still there after `grace` seconds."""
    p = subprocess.Popen(cmd, cwd=cwd)
    try:
        return p.wait()
    except KeyboardInterrupt:
        end = time.time() + grace
        while p.poll() is None and time.time() < end:
            try:
                p.wait(max(0.1, end - time.time()))
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                pass
        if p.poll() is None:
            kill(p)
            p.wait()
        raise


if __name__ == "__main__":  # python graceful.py: a tiny self-test of the mapping
    install()
    print("ready", flush=True)
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("interrupted cleanly", flush=True)
        sys.exit(130)
