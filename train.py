"""Pretrain the GPT on tokenized .bin data. Single GPU or multi-GPU via torchrun.

Single GPU:   python train.py --preset small
Multi-GPU:    torchrun --nproc_per_node=8 train.py --preset 1b --batch 16 --grad-accum 4
CPU run:      python train.py --preset cpu --vocab 8192 --batch 16 --grad-accum 4 --max-steps 20000 --warmup 500 --eval-every 250 --lr 1e-3
Several processes on one PC (data parallel, faster for small models on the CPU):
              python train.py --preset cpu --procs 2      (--procs 0 = auto, --threads N caps the CPU threads)
Continue:     add --resume (same --out). The model size comes from the checkpoint, so it also works on
              another PC after copying the run folder (see transfer.py).
"""
import argparse, json, math, os, sys, time
from dataclasses import asdict

import numpy as np
import torch

import devices
import graceful
from core import parallel, safeload
from model.gpt import GPT, GPTConfig, PRESETS
from writing import hf


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bin")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--preset", default="small", choices=PRESETS)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--batch", type=int, default=16, help="micro-batch per GPU")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--device", default="auto", help="auto (GPU if there is one), cuda, mps, xpu or cpu")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU threads for the whole run (0 = all physical cores, or TEXTLLM_THREADS)")
    ap.add_argument("--procs", type=int, default=1,
                    help="training processes on this PC, each on a slice of the batch (0 = auto; "
                         "one per GPU on multi-GPU PCs); see core/parallel.py")
    return ap.parse_args()


class Loader:
    def __init__(self, path, batch, block, device, rank, world):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch, self.block, self.device = batch, block, device
        self.rng = np.random.default_rng(1337 + rank)

    def next(self):
        ix = self.rng.integers(0, len(self.data) - self.block - 1, self.batch)
        # one gather per tensor for the whole batch (no per-row Python loop)
        pos = ix[:, None] + np.arange(self.block)[None, :]
        x = torch.from_numpy(self.data[pos].astype(np.int64))
        y = torch.from_numpy(self.data[pos + 1].astype(np.int64))
        if devices.kind(self.device) == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


def make_optimizer(model, lr, weight_decay, device, betas=(0.9, 0.95)):
    """AdamW with weight decay only on matrices. The fused kernel updates all parameters in one
    pass; PyTorch has it for CUDA and (since 2.4) for the CPU, so it is used wherever it works."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=devices.kind(device) in ("cuda", "cpu"))
    except (RuntimeError, ValueError):  # an older PyTorch without fused CPU AdamW
        return torch.optim.AdamW(groups, lr=lr, betas=betas)


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = min(1.0, (step - a.warmup) / max(1, a.max_steps - a.warmup))
    return a.min_lr + 0.5 * (1 + math.cos(math.pi * p)) * (a.lr - a.min_lr)


def main():
    graceful.install()
    a = get_args()
    ddp = int(os.environ.get("RANK", -1)) != -1
    if not ddp:
        a.device = devices.pick(a.device)
        gpus = torch.cuda.device_count() if devices.kind(a.device) == "cuda" else 0
        procs, per = parallel.plan(a.procs, a.threads, devices.kind(a.device), gpus, a.preset)
        if procs > 1:
            # Start the processes and wait; each one comes back here with RANK set.
            print(f"{procs} training processes on this PC, {per} CPU threads each, "
                  f"micro-batch {parallel.split_batch(a.batch, procs)} each (total {a.batch})", flush=True)
            sys.exit(parallel.launch(os.path.abspath(__file__), sys.argv[1:], procs, per))
        if devices.kind(a.device) == "cpu":
            os.environ["TEXTLLM_THREADS"] = str(per)
        rank, world = 0, 1
    else:
        import torch.distributed as dist  # only for several processes; phones' PyTorch may lack it
        from torch.nn.parallel import DistributedDataParallel as DDP

        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        a.device = devices.pick(a.device)
        if devices.kind(a.device) == "cuda":
            a.device = f"cuda:{int(os.environ.get('LOCAL_RANK', rank))}"
            torch.cuda.set_device(a.device)
        dist.init_process_group(parallel.backend(devices.kind(a.device)), rank=rank, world_size=world)
        if os.environ.get(parallel.ENV_SPLIT) == "1":  # --procs: --batch is the total; torchrun: per GPU
            a.batch = parallel.split_batch(a.batch, world)
    master = rank == 0
    torch.manual_seed(1337 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    kind = devices.kind(a.device)
    dtype = devices.train_dtype(a.device)
    autocast = devices.autocast(a.device, dtype)
    prec = str(dtype).split(".")[-1]
    if kind == "cpu":
        threads = hf.set_threads()  # physical cores, TEXTLLM_THREADS, or this process's share
        if cpu_ac := devices.cpu_autocast(a.device):
            autocast, prec = cpu_ac, "fp32 + bf16 autocast"
    scaler = devices.grad_scaler(a.device, dtype)  # only active for fp16 (older NVIDIA GPUs)
    if master:
        print(f"device {a.device}: {devices.describe(a.device)} · {prec}"
              + (f" · {threads} threads" if kind == "cpu" else "")
              + (f" × {world} processes" if world > 1 else ""))

    cfg = GPTConfig(vocab_size=a.vocab, **PRESETS[a.preset])
    step, best = 0, float("inf")
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    ck = None
    if a.resume and os.path.exists(ckpt_path):
        # Load on the CPU first: a checkpoint saved on a GPU machine loads on a CPU one and the other
        # way round, and the GPU doesn't hold two copies of the weights while loading.
        ck = safeload.load(ckpt_path)
        cfg = GPTConfig(**ck["config"])  # the saved size wins over --preset/--vocab
    elif a.resume and master:
        print(f"no checkpoint at {ckpt_path}; starting from step 0")
    model = GPT(cfg).to(a.device)

    opt = make_optimizer(model, a.lr, a.weight_decay, a.device)
    if ck is not None:
        model.load_state_dict(ck["model"])
        if ck.get("opt"):  # a cluster-trained checkpoint (cluster.py) has no optimizer state
            opt.load_state_dict(ck["opt"])  # moves the optimizer state to the parameters' device
        if ck.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(ck["scaler"])
        step, best = ck["step"], ck["best_val"]
        if master:
            print(f"resumed at step {step} (best val {best:.3f})")
    if master:
        print(f"{a.preset}: {model.num_params()/1e6:.1f}M params, "
              f"{a.batch * a.grad_accum * world * cfg.block_size:,} tokens/step")
    if a.compile:
        model = torch.compile(model)
    raw = model
    if ddp:
        model = DDP(model, device_ids=[torch.device(a.device).index] if kind == "cuda" else None)
        raw = model.module
    raw = getattr(raw, "_orig_mod", raw)

    train = Loader(os.path.join(a.data, "train.bin"), a.batch, cfg.block_size, a.device, rank, world)
    val = Loader(os.path.join(a.data, "val.bin"), a.batch, cfg.block_size, a.device, rank, world)
    if ck is not None and world == 1 and "rng" in ck:
        # Pick up the random batch sampling where it stopped instead of replaying the first batches.
        train.rng.bit_generator.state = ck["rng"]["train"]
        val.rng.bit_generator.state = ck["rng"]["val"]
        torch.set_rng_state(ck["rng"]["torch"].cpu())
    ck = None  # free the memory

    def save():
        # Write to a temp file and swap it in, so a crash or power cut mid-save never leaves a
        # broken ckpt.pt (the previous one stays until the new one is complete).
        tmp = ckpt_path + ".tmp"
        torch.save({"model": raw.state_dict(), "opt": opt.state_dict(), "config": asdict(cfg),
                    "step": step, "best_val": best,
                    "rng": {"train": train.rng.bit_generator.state, "val": val.rng.bit_generator.state,
                            "torch": torch.get_rng_state()},
                    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                    "args": vars(a)}, tmp)
        os.replace(tmp, ckpt_path)

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for _ in range(a.eval_iters):
            x, y = val.next()
            with autocast:
                losses.append(raw(x, y)[1].item())
        model.train()
        vl = sum(losses) / len(losses)
        if ddp:  # every process looked at its own batches: report the average
            t = torch.tensor([vl], dtype=torch.float64)
            if kind == "cuda":
                t = t.to(a.device)
            dist.all_reduce(t)
            vl = t.item() / world
        return vl

    os.makedirs(a.out, exist_ok=True)
    mlog = open(os.path.join(a.out, "metrics.jsonl"), "a") if master else None

    def log(**kw):  # read by app.py for live charts
        if mlog:
            mlog.write(json.dumps({"time": time.time(), "max_steps": a.max_steps, **kw}) + "\n")
            mlog.flush()

    x, y = train.next()
    t0 = time.time()
    while step < a.max_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        for micro in range(a.grad_accum):
            if ddp:
                model.require_backward_grad_sync = micro == a.grad_accum - 1
            with autocast:
                loss = model(x, y)[1] / a.grad_accum
            x, y = train.next()
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if master and step % 10 == 0:
            dt = (time.time() - t0) / 10
            t0 = time.time()
            tl = loss.item() * a.grad_accum
            print(f"step {step} | loss {tl:.3f} | lr {lr_at(step, a):.2e} | {dt*1000:.0f} ms/step")
            tps = a.batch * a.grad_accum * world * cfg.block_size / dt
            log(kind="train", step=step, loss=tl, lr=lr_at(step, a), tokens_per_sec=tps)
        if step % a.eval_every == 0 or step == a.max_steps:
            vl = evaluate()
            if master:
                print(f"== step {step} val loss {vl:.3f} (ppl {math.exp(vl):.1f})")
                log(kind="val", step=step, loss=vl)
                best = min(best, vl)
                save()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
