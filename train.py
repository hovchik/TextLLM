"""Pretrain the GPT on tokenized .bin data. Single GPU or multi-GPU via torchrun.

Single GPU:   python train.py --preset small
Multi-GPU:    torchrun --nproc_per_node=8 train.py --preset 1b --batch 16 --grad-accum 4
CPU run:      python train.py --preset cpu --vocab 8192 --batch 16 --grad-accum 4 --max-steps 20000 --warmup 500 --eval-every 250 --lr 1e-3
Continue:     add --resume (same --out). The model size comes from the checkpoint, so it also works on
              another PC after copying the run folder (see transfer.py).
"""
import argparse, json, math, os, time
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import devices
import graceful
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
    return ap.parse_args()


class Loader:
    def __init__(self, path, batch, block, device, rank, world):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch, self.block, self.device = batch, block, device
        self.rng = np.random.default_rng(1337 + rank)

    def next(self):
        ix = self.rng.integers(0, len(self.data) - self.block - 1, self.batch)
        x = torch.from_numpy(np.stack([self.data[i : i + self.block].astype(np.int64) for i in ix]))
        y = torch.from_numpy(np.stack([self.data[i + 1 : i + 1 + self.block].astype(np.int64) for i in ix]))
        if devices.kind(self.device) == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = min(1.0, (step - a.warmup) / max(1, a.max_steps - a.warmup))
    return a.min_lr + 0.5 * (1 + math.cos(math.pi * p)) * (a.lr - a.min_lr)


def main():
    graceful.install()
    a = get_args()
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        a.device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        torch.cuda.set_device(a.device)
    else:
        rank, world = 0, 1
        a.device = devices.pick(a.device)
    master = rank == 0
    torch.manual_seed(1337 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    kind = devices.kind(a.device)
    cuda = kind == "cuda"
    dtype = devices.train_dtype(a.device)
    autocast = devices.autocast(a.device, dtype)
    prec = str(dtype).split(".")[-1]
    if kind == "cpu":
        hf.set_threads()  # physical cores (or TEXTLLM_THREADS)
        if cpu_ac := devices.cpu_autocast(a.device):
            autocast, prec = cpu_ac, "fp32 + bf16 autocast"
    scaler = devices.grad_scaler(a.device, dtype)  # only active for fp16 (older NVIDIA GPUs)
    if master:
        print(f"device {a.device}: {devices.describe(a.device)} · {prec}")

    cfg = GPTConfig(vocab_size=a.vocab, **PRESETS[a.preset])
    step, best = 0, float("inf")
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    ck = None
    if a.resume and os.path.exists(ckpt_path):
        # Load on the CPU first: a checkpoint saved on a GPU machine loads on a CPU one and the other
        # way round, and the GPU doesn't hold two copies of the weights while loading.
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = GPTConfig(**ck["config"])  # the saved size wins over --preset/--vocab
    elif a.resume and master:
        print(f"no checkpoint at {ckpt_path}; starting from step 0")
    model = GPT(cfg).to(a.device)

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, 0.95), fused=cuda,
    )
    if ck is not None:
        model.load_state_dict(ck["model"])
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
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
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
                losses.append(model(x, y)[1].item())
        model.train()
        return sum(losses) / len(losses)

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
