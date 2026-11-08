"""Generate text from a trained checkpoint.

python generate.py --prompt "Once upon a time" --tokens 400
"""
import argparse

import torch
from tokenizers import Tokenizer

import devices
from core import safeload
from model.gpt import GPT, GPTConfig, Penalties

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/ckpt.pt")
ap.add_argument("--tok", default="tokenizer/bpe.json")
ap.add_argument("--prompt", default="Once upon a time")
ap.add_argument("--tokens", type=int, default=300)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top-k", type=int, default=50)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--repetition-penalty", type=float, default=1.2, help="1 = off; 1.1-1.3 stops loops")
ap.add_argument("--frequency-penalty", type=float, default=0.1, help="0 = off; each extra use of a token costs this much")
ap.add_argument("--presence-penalty", type=float, default=0.0, help="0 = off; any token already used costs this much")
ap.add_argument("--no-repeat-ngram", type=int, default=4, help="never repeat a run of this many tokens (0 = off)")
ap.add_argument("--samples", type=int, default=1)
ap.add_argument("--device", default="auto", help="auto (GPU if there is one), cuda, mps, xpu or cpu")
a = ap.parse_args()
a.device = devices.pick(a.device)

tok = Tokenizer.from_file(a.tok)
ck = safeload.load(a.ckpt)
model = GPT(GPTConfig(**ck["config"])).to(a.device).eval()
model.load_state_dict(ck["model"])

ids = torch.tensor([tok.encode(a.prompt).ids] * a.samples, device=a.device)
pen = Penalties(a.repetition_penalty, a.presence_penalty, a.frequency_penalty, a.no_repeat_ngram, window=512)
out = model.generate(ids, a.tokens, a.temperature, a.top_k, a.top_p, stop_id=tok.token_to_id("<|endoftext|>"), penalties=pen)
for i, row in enumerate(out.tolist()):
    print(f"--- sample {i + 1} ---\n{tok.decode(row)}\n")
