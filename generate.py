"""Generate text from a trained checkpoint.

python generate.py --prompt "Once upon a time" --tokens 400
"""
import argparse

import torch
from tokenizers import Tokenizer

import devices
from model.gpt import GPT, GPTConfig

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/ckpt.pt")
ap.add_argument("--tok", default="tokenizer/bpe.json")
ap.add_argument("--prompt", default="Once upon a time")
ap.add_argument("--tokens", type=int, default=300)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top-k", type=int, default=50)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--samples", type=int, default=1)
ap.add_argument("--device", default="auto", help="auto (GPU if there is one), cuda, mps, xpu or cpu")
a = ap.parse_args()
a.device = devices.pick(a.device)

tok = Tokenizer.from_file(a.tok)
ck = torch.load(a.ckpt, map_location="cpu")
model = GPT(GPTConfig(**ck["config"])).to(a.device).eval()
model.load_state_dict(ck["model"])

ids = torch.tensor([tok.encode(a.prompt).ids] * a.samples, device=a.device)
out = model.generate(ids, a.tokens, a.temperature, a.top_k, a.top_p, stop_id=tok.token_to_id("<|endoftext|>"))
for i, row in enumerate(out.tolist()):
    print(f"--- sample {i + 1} ---\n{tok.decode(row)}\n")
