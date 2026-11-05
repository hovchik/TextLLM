"""Load any trained model in runs/ and stream text from it (used by app.py)."""
import glob, json, os, threading

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")


def list_models():
    """Return {display name: run dir} for every run that has saved weights."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        meta_p = os.path.join(d, "meta.json")
        if not os.path.exists(meta_p):
            continue
        meta = json.load(open(meta_p))
        ready = os.path.exists(os.path.join(d, "ckpt.pt")) if meta["type"] == "scratch" \
            else os.path.isdir(os.path.join(d, "adapter"))
        if ready:
            label = "from scratch" if meta["type"] == "scratch" else f"fine-tuned {meta['base'].split('/')[-1]}"
            out[f"{os.path.basename(d)} ({label})"] = d
    return out


class Engine:
    def __init__(self, run_dir):
        self.meta = json.load(open(os.path.join(run_dir, "meta.json")))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.meta["type"] == "scratch":
            from tokenizers import Tokenizer
            from model.gpt import GPT, GPTConfig

            self.tok = Tokenizer.from_file(self.meta["tok"])
            ck = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location=self.device)
            self.model = GPT(GPTConfig(**ck["config"])).to(self.device).eval()
            self.model.load_state_dict(ck["model"])
            self.step = ck.get("step")
        else:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            ad = os.path.join(run_dir, "adapter")
            self.tok = AutoTokenizer.from_pretrained(ad)
            base = AutoModelForCausalLM.from_pretrained(self.meta["base"])
            self.model = PeftModel.from_pretrained(base, ad).to(self.device).eval()

    def stream(self, history, message, max_new_tokens=300, temperature=0.8, top_p=0.95):
        """Yield the growing reply text."""
        if self.meta["type"] == "scratch":
            yield from self._stream_scratch(message, max_new_tokens, temperature, top_p)
        else:
            yield from self._stream_hf(history, message, max_new_tokens, temperature, top_p)

    @torch.no_grad()
    def _stream_scratch(self, prompt, n, temperature, top_p):
        # The from-scratch model only continues text, so the message is used as the opening.
        ids = torch.tensor([self.tok.encode(prompt).ids], device=self.device)
        start, stop = ids.shape[1], self.tok.token_to_id("<|endoftext|>")
        for _ in range(n):
            ids = self.model.generate(ids, 1, temperature, 50, top_p)
            if ids[0, -1].item() == stop:
                break
            yield prompt + self.tok.decode(ids[0, start:].tolist())

    def _stream_hf(self, history, message, n, temperature, top_p):
        from transformers import TextIteratorStreamer

        msgs = [{"role": m["role"], "content": m["content"]} for m in history if isinstance(m.get("content"), str)]
        msgs.append({"role": "user", "content": message})
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) \
            if self.tok.chat_template else f"### Request:\n{message}\n\n### Response:\n"
        inputs = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kw = dict(**inputs, streamer=streamer, max_new_tokens=n, do_sample=temperature > 0,
                  temperature=max(temperature, 1e-5), top_p=top_p, pad_token_id=self.tok.eos_token_id)
        threading.Thread(target=self.model.generate, kwargs=kw, daemon=True).start()
        out = ""
        for piece in streamer:
            out += piece
            yield out
