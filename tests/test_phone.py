"""Phones in the cluster: battery rules, the code bundle and setup script the coordinator serves, and a
worker run from that bundle that waits while 'unplugged' and trains once 'charging'."""
import io, json, os, subprocess, sys, time, urllib.request, zipfile

import pytest

import power
from conftest import ROOT
from test_cluster import SECRET, free_port, start
from test_scratch_pipeline import write_bins


@pytest.mark.parametrize("b,ok,word", [
    ({"percentage": 80, "plugged": "PLUGGED_AC", "status": "CHARGING", "temperature": 31.0}, True, "charging"),
    ({"percentage": 80, "plugged": "UNPLUGGED", "status": "DISCHARGING", "temperature": 31.0}, False, "not charging"),
    ({"percentage": 12, "plugged": "PLUGGED_USB", "status": "CHARGING", "temperature": 31.0}, False, "12%"),
    ({"percentage": 90, "plugged": "PLUGGED_AC", "status": "CHARGING", "temperature": 45.5}, False, "46 °C"),
    ({"percentage": 100, "plugged": "UNPLUGGED", "status": "FULL", "temperature": 30}, True, "charging"),
    (None, True, "unknown"),
])
def test_ok_to_train(b, ok, word):
    got, why = power.ok_to_train(b)
    assert got is ok and word in why


def test_battery_from_linux_power_supply(tmp_path):
    bat, ac = tmp_path / "battery", tmp_path / "ac"
    bat.mkdir(), ac.mkdir()
    for d, vals in ((bat, {"type": "Battery", "capacity": "64", "status": "Discharging", "temp": "387"}),
                    (ac, {"type": "Mains", "online": "1"})):
        for k, v in vals.items():
            (d / k).write_text(v + "\n")
    b = power._from_sys(str(tmp_path))
    assert b == {"percentage": 64, "plugged": "PLUGGED", "status": "DISCHARGING", "temperature": 38.7}
    (ac / "online").write_text("0")
    assert power.ok_to_train(power._from_sys(str(tmp_path)))[0] is False


def get(url, secret=None):
    req = urllib.request.Request(url, headers={"X-Secret": secret} if secret else {})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def test_phone_joins_from_the_served_bundle(bpe, tmp_path):
    data = str(tmp_path / "bin")
    vocab = write_bins(bpe, data)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    coord = start("coordinator", "--out", str(tmp_path / "run"), "--data", data, "--preset", "tiny", "--vocab", str(vocab),
                  "--rounds", "1", "--round-minutes", str(8 / 60), "--warmup", "1", "--eval-iters", "2", "--eval-batch", "2",
                  "--host", "127.0.0.1", "--port", str(port), "--secret", SECRET)
    phone = None
    try:
        for _ in range(100):
            try:
                get(url + "/ping")
                break
            except OSError:
                time.sleep(0.2)
        # the setup script and the join page need the secret (in the URL, for a curl one-liner)
        with pytest.raises(urllib.error.HTTPError):
            get(url + "/phone.sh?secret=wrong")
        script = get(url + f"/phone.sh?secret={SECRET}").decode()
        assert f"URL='{url}'" in script and "pkg install" in script and "cluster.py worker" in script
        assert "Copy command" in get(url + f"/phone?secret={SECRET}").decode()
        # the worker code bundle runs on its own, without the rest of the repository
        code = tmp_path / "phone"
        zipfile.ZipFile(io.BytesIO(get(url + "/worker-code.zip", SECRET))).extractall(code)
        battery = tmp_path / "battery.json"
        battery.write_text(json.dumps({"percentage": 70, "plugged": "UNPLUGGED", "status": "DISCHARGING", "temperature": 30}))
        status = tmp_path / "phone.json"
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env.update(TEXTLLM_THREADS="1", TEXTLLM_BATTERY_FILE=str(battery), TEXTLLM_CLUSTER_CACHE=str(tmp_path / "cache"),
                   TEXTLLM_CLUSTER_SECRET=SECRET)
        phone = subprocess.Popen([sys.executable, "cluster.py", "worker", "--join", url, "--name", "Pixel test",
                                  "--device", "cpu", "--batch", "2", "--grad-accum", "1", "--data-mb", "1",
                                  "--only-charging", "--power-every", "1", "--status-file", str(status)],
                                 cwd=code, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(120):
            st = json.loads(status.read_text()) if status.exists() else {}
            if st.get("state") == "paused":
                break
            assert phone.poll() is None, phone.stdout.read()
            time.sleep(0.5)
        assert "not charging" in st["note"]
        cl = json.load(open(tmp_path / "run" / "cluster.json"))
        assert cl["round"] == 0  # nothing trained while unplugged
        battery.write_text(json.dumps({"percentage": 70, "plugged": "PLUGGED_AC", "status": "CHARGING", "temperature": 30}))
        out = phone.communicate(timeout=180)[0]
        assert phone.returncode == 0, out
        assert "training again" in out and "finished training" in out
        coord.communicate(timeout=60)
        cl = json.load(open(tmp_path / "run" / "cluster.json"))
        assert cl["round"] == 1 and cl["workers"][0]["name"] == "Pixel test" and cl["workers"][0]["rounds"] == 1
    finally:
        for p in (phone, coord):
            if p and p.poll() is None:
                p.kill()
