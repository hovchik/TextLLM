"""Intel Arc (XPU) support. The test machine has no Intel GPU, so torch.xpu is faked: these check
the decisions (which device, which precision, which PyTorch build to install), not the GPU itself."""
import types

import pytest
import torch

import devices
from core import gpuprobe


@pytest.fixture
def fake_arc(monkeypatch):
    props = types.SimpleNamespace(name="Intel(R) Arc(TM) A770 Graphics", total_memory=16 * 2**30)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2, raising=False)
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda i=0: props, raising=False)
    monkeypatch.setattr(torch.xpu, "is_bf16_supported", lambda: True, raising=False)
    monkeypatch.delenv("TEXTLLM_DEVICE", raising=False)
    return props


def test_auto_picks_the_arc_gpu(fake_arc):
    assert devices.pick() == "xpu"
    assert devices.pick("xpu:1") == "xpu:1"
    assert devices.train_dtype("xpu") == torch.bfloat16
    assert devices.infer_dtype("xpu") == torch.float16
    assert devices.describe("xpu") == "Intel(R) Arc(TM) A770 Graphics (16.0 GB)"
    ids = [d["id"] for d in devices.available()]
    assert ids[:2] == ["auto", "cpu"] and "xpu" in ids and "xpu:1" in ids


def test_arc_without_bf16_trains_in_fp32(fake_arc, monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_bf16_supported", lambda: False, raising=False)
    assert devices.train_dtype("xpu") == torch.float32


def test_bf16_on_xpu_needs_no_loss_scaling(fake_arc):
    assert not devices.grad_scaler("xpu", torch.bfloat16).is_enabled()


def test_xpu_falls_back_to_cpu_when_there_is_no_arc(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False, raising=False)
    monkeypatch.setattr(devices, "_mps_ok", lambda t: False)
    assert devices.pick("xpu") == "cpu"


def test_studio_hint_when_arc_is_there_but_pytorch_cant_use_it(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False, raising=False)
    monkeypatch.setattr(devices, "_mps_ok", lambda t: False)
    monkeypatch.setattr(devices, "_nvidia_present", lambda: False)
    monkeypatch.setattr(devices, "_ARC", "Intel(R) Arc(TM) B580 Graphics")
    monkeypatch.setattr(torch.version, "xpu", None, raising=False)
    monkeypatch.delenv("TEXTLLM_DEVICE", raising=False)
    info = devices.info()
    assert info["arc"] == "Intel(R) Arc(TM) B580 Graphics"
    assert "run.bat --xpu" in info["hint"]


@pytest.mark.parametrize("name, arc", [
    ("Intel(R) Arc(TM) A770 Graphics", True),
    ("Intel(R) Arc(TM) Graphics", True),  # Core Ultra laptop iGPU
    ("Intel Corporation DG2 [Arc A750] (rev 08)", True),
    ("Intel Corporation Battlemage G21 [Arc B580]", True),
    ("Intel Corporation Meteor Lake-P [Intel Arc Graphics] (rev 08)", True),
    ("Intel(R) Iris(R) Xe Graphics", False),
    ("Intel(R) UHD Graphics 620", False),
    ("NVIDIA GeForce RTX 3060 Laptop GPU", False),
    ("Searching archives", False),
])
def test_arc_names(name, arc):
    assert gpuprobe.is_arc_name(name) is arc


@pytest.mark.parametrize("dev, arc", [("0x56a0", True), ("0xe20b", True), ("0x7d55", True), ("0x64a0", True),
                                      ("0x9a49", False), ("0x3e9b", False)])
def test_arc_pci_ids(dev, arc):
    assert bool(gpuprobe._ARC_IDS.match(dev)) is arc


def _args(**kw):
    base = dict(torch_index=None, cpu=False, gpu=False, xpu=False)
    return types.SimpleNamespace(**{**base, **kw})


def test_installer_picks_the_xpu_build_for_arc(monkeypatch):
    import run

    monkeypatch.setattr(run.sys, "platform", "win32")
    monkeypatch.setattr(run, "nvidia_gpus", lambda: [])
    monkeypatch.setattr(run.gpuprobe, "intel_arc_gpus", lambda: [("Intel(R) Arc(TM) A770 Graphics", "32.0.101.6130")])
    assert run.choose_torch(_args()) == "xpu"
    assert run.choose_torch(_args(cpu=True)) == "cpu"
    # an NVIDIA card wins over an Arc one (CUDA has the wider library support)
    monkeypatch.setattr(run, "nvidia_gpus", lambda: [("RTX 3060", 6144, "560.10")])
    assert run.choose_torch(_args()).startswith("cu")
    assert run.choose_torch(_args(xpu=True)) == "xpu"


def test_installer_without_any_gpu_stays_cpu(monkeypatch):
    import run

    monkeypatch.setattr(run.sys, "platform", "linux")
    monkeypatch.setattr(run, "nvidia_gpus", lambda: [])
    monkeypatch.setattr(run.gpuprobe, "intel_arc_gpus", lambda: [])
    assert run.choose_torch(_args()) == "cpu"
    assert [run.build_kind(v) for v in ("xpu", "cu128,cu126", "rocm6.2", "cpu", "default")] == \
        ["xpu", "cuda", "cuda", "cpu", None]


def test_gguf_stays_on_cpu_when_cpu_is_chosen(monkeypatch):
    import chat_engine

    monkeypatch.setenv("TEXTLLM_DEVICE", "cpu")
    assert chat_engine.gguf_gpu_layers() == 0


def test_gguf_is_used_on_a_gpu_when_llama_cpp_can_offload(monkeypatch):
    import sys
    import chat_engine

    monkeypatch.delenv("TEXTLLM_DEVICE", raising=False)
    monkeypatch.delenv("TEXTLLM_GGUF_GPU", raising=False)
    fake = types.SimpleNamespace(llama_supports_gpu_offload=lambda: True)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    assert chat_engine.gguf_gpu_layers() == -1
    fake.llama_supports_gpu_offload = lambda: False
    assert chat_engine.gguf_gpu_layers() == 0


def test_gpu_debug_switches_are_turned_off(monkeypatch):
    from core import quiet

    env = {"ONEDNN_VERBOSE": "1", "ZE_ENABLE_LOADER_DEBUG_TRACE": "1", "PATH": "x"}
    assert sorted(quiet.gpu_debug_off(env)) == ["ONEDNN_VERBOSE", "ZE_ENABLE_LOADER_DEBUG_TRACE"]
    assert env == {"PATH": "x"}
    env = {"ONEDNN_VERBOSE": "1", "TEXTLLM_GPU_DEBUG": "1"}
    assert quiet.gpu_debug_off(env) == [] and "ONEDNN_VERBOSE" in env


def test_slow_first_gpu_step_restarts_on_the_cpu(monkeypatch):
    import os, subprocess, sys, threading
    import train

    calls, exited = [], threading.Event()
    monkeypatch.setattr(sys, "argv", ["train.py", "--preset", "small", "--device", "xpu", "--gpu-wait", "0.001"])
    monkeypatch.setattr(subprocess, "call", lambda cmd, env=None: calls.append((cmd, env["TEXTLLM_DEVICE"])) or 0)
    monkeypatch.setattr(os, "_exit", lambda rc: exited.set())
    done = train.first_step_watchdog(0.001)  # 0.06 s
    assert exited.wait(5)
    cmd, dev = calls[0]
    assert cmd[-2:] == ["--device", "cpu"] and "xpu" not in cmd and dev == "cpu"
    assert not done.is_set()


def test_fast_first_gpu_step_keeps_the_gpu(monkeypatch):
    import subprocess, time
    import train

    calls = []
    monkeypatch.setattr(subprocess, "call", lambda *a, **k: calls.append(a) or 0)
    done = train.first_step_watchdog(0.01)  # 0.6 s
    done.set()
    time.sleep(1)
    assert calls == []
