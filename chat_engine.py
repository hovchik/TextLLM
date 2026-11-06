"""Load any trained model in runs/ and write with it (used by app.py and evaluate.py).

Three kinds of model:
  * scratch: the from-scratch GPT (continues text only)
  * lora:    a fine-tuned open model. If the run has a quantized model.gguf that is newer than
             its adapter and llama-cpp-python is installed, that file is used (much faster on CPU);
             otherwise the base model plus LoRA adapter is loaded with PEFT.
"""
import glob, json, os, threading

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")

from writing import formats as F  # noqa: E402
from writing import hf  # noqa: E402


def list_models():
    """Return {display name: run dir} for every run that has saved weights."""
    out = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        meta_p = os.path.join(d, "meta.json")
        if not os.path.exists(meta_p):
            continue
        meta = json.load(open(meta_p))
        ready = os.path.exists(os.path.join(d, "ckpt.pt")) if meta["type"] == "scratch" \
            else os.path.isdir(os.path.join(d, "adapter")) or bool(gguf_path(d))
        if ready:
            label = "from scratch" if meta["type"] == "scratch" else f"fine-tuned {meta['base'].split('/')[-1]}"
            out[f"{os.path.basename(d)} ({label})"] = d
    return out


def _newest(path):
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path)]
        return max((os.path.getmtime(f) for f in files), default=0)
    return os.path.getmtime(path) if os.path.exists(path) else 0


def gguf_path(run_dir):
    """The run's GGUF file if it is at least as new as the adapter (else it is stale)."""
    files = sorted(glob.glob(os.path.join(run_dir, "*.gguf")), key=os.path.getmtime)
    if not files:
        return None
    ad = os.path.join(run_dir, "adapter")
    return files[-1] if _newest(files[-1]) >= _newest(ad) else None


def weights_stamp(run_dir):
    """Changes whenever the weights used for chat change (for reloading)."""
    return max(_newest(os.path.join(run_dir, "ckpt.pt")), _newest(os.path.join(run_dir, "adapter")),
               max((_newest(f) for f in glob.glob(os.path.join(run_dir, "*.gguf"))), default=0))


def has_llama_cpp():
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


class Engine:
    def __init__(self, run_dir, backend=None):
        self.run_dir = run_dir
        self.meta = json.load(open(os.path.join(run_dir, "meta.json")))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lock = threading.Lock()  # one generation at a time per model
        hf.set_threads()
        g = gguf_path(run_dir) if self.meta["type"] != "scratch" else None
        if backend is None:
            backend = "scratch" if self.meta["type"] == "scratch" else (
                "gguf" if g and has_llama_cpp() and self.device == "cpu" else "hf")
        self.backend = backend
        if backend == "scratch":
            from tokenizers import Tokenizer
            from model.gpt import GPT, GPTConfig

            self.tok = Tokenizer.from_file(self.meta["tok"])
            ck = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location=self.device)
            self.model = GPT(GPTConfig(**ck["config"])).to(self.device).eval()
            self.model.load_state_dict(ck["model"])
            self.step = ck.get("step")
        elif backend == "gguf":
            from llama_cpp import Llama

            self.gguf = g
            self.model = Llama(model_path=g, n_ctx=int(os.environ.get("TEXTLLM_CTX", 4096)),
                               n_threads=hf.physical_cores(), verbose=False)
        else:
            from peft import PeftModel

            ad = os.path.join(run_dir, "adapter")
            self.tok = hf.load_tokenizer(ad)
            dtype = torch.bfloat16 if self.device == "cpu" and hf.cpu_bf16_ok() else None
            base = hf.load_causal_lm(self.meta["base"], dtype)
            self.model = PeftModel.from_pretrained(base, ad).to(self.device).eval()

    # ------------------------------------------------------------ low level
    def stream_messages(self, msgs, s):
        """Yield the growing reply for chat messages. s = sampling dict (see formats.json)."""
        if self.backend == "scratch":
            prompt = msgs[-1]["content"]
            for full in self._stream_scratch(prompt, s["max_tokens"], s["temperature"], s.get("top_p", 0.95)):
                yield full[len(prompt):] if full.startswith(prompt) else full
        elif self.backend == "gguf":
            yield from self._stream_gguf(msgs, s)
        else:
            yield from self._stream_hf(msgs, s)

    def complete(self, msgs, s):
        out = ""
        for out in self.stream_messages(msgs, s):
            pass
        return out

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

    def _stream_gguf(self, msgs, s):
        out = ""
        for ch in self.model.create_chat_completion(
                messages=msgs, stream=True, max_tokens=s["max_tokens"], temperature=max(s["temperature"], 0.0),
                top_p=s.get("top_p", 0.95), top_k=s.get("top_k", 40), presence_penalty=s.get("presence_penalty", 0.0)):
            piece = ch["choices"][0].get("delta", {}).get("content")
            if piece:
                out += piece
                yield out

    def _stream_hf(self, msgs, s):
        from transformers import TextIteratorStreamer

        text = hf.chat_text(self.tok, msgs)
        inputs = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        t = s["temperature"]
        kw = dict(**inputs, streamer=streamer, max_new_tokens=s["max_tokens"], do_sample=t > 0,
                  temperature=max(t, 1e-5), top_p=s.get("top_p", 0.95), top_k=s.get("top_k", 40),
                  repetition_penalty=hf.repetition_from_presence(s.get("presence_penalty")),
                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        threading.Thread(target=self.model.generate, kwargs=kw, daemon=True).start()
        out = ""
        for piece in streamer:
            out += piece
            yield out

    # ------------------------------------------------------------ formats
    def write(self, fmt, message, fields=None, history=None, sampling=None):
        """Write in a format. Yields events:
          {"type": "status", "text"}             what is happening (planning, retrying)
          {"type": "outline", "text"}            the plan for stories and essays
          {"type": "text", "text"}               the growing answer (replace what was shown)
          {"type": "final", "text", "check"}     the answer kept, with its automatic checks
        """
        f = F.get(fmt)
        s = {**f["sampling"], **{k: v for k, v in (sampling or {}).items() if v is not None}}
        if self.backend == "scratch":  # continues text; templates don't apply
            text = ""
            for text in self.stream_messages([{"role": "user", "content": message}], s):
                yield {"type": "text", "text": text}
            yield {"type": "final", "text": text, "check": F.check(fmt, text, fields)}
            return
        msgs = F.messages(fmt, message, fields, history)
        with self.lock:
            if f.get("steps") == "outline":
                yield {"type": "status", "text": "Planning…"}
                outline = self.complete(F.outline_messages(fmt, message, fields),
                                        {**s, "max_tokens": 200, "temperature": min(0.7, s["temperature"])})
                yield {"type": "outline", "text": outline.strip()}
                msgs = F.with_outline(msgs, outline)
            tries = f.get("candidates", 1) if f.get("steps") == "best_of" else 1
            best = None
            for i in range(tries):
                if i:
                    yield {"type": "status", "text": f"Trying again to fit the format ({i + 1}/{tries})…"}
                text = ""
                for text in self.stream_messages(msgs, s):
                    yield {"type": "text", "text": text}
                text = F.clean(text) if f.get("steps") == "best_of" else text.strip()
                chk = F.check(fmt, text, fields)
                if best is None or chk["score"] > best[1]["score"]:
                    best = (text, chk)
                if chk["score"] >= 100:
                    break
            yield {"type": "text", "text": best[0]}
            yield {"type": "final", "text": best[0], "check": best[1]}

    # ------------------------------------------------------------ old API (free chat)
    def stream(self, history, message, max_new_tokens=300, temperature=0.8, top_p=0.95):
        """Yield the growing reply text (free chat, no format)."""
        s = {**F.get("free")["sampling"], "max_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p}
        if self.backend == "scratch":
            for out in self._stream_scratch(message, max_new_tokens, temperature, top_p):
                yield out
            return
        msgs = [{"role": m["role"], "content": m["content"]} for m in history if isinstance(m.get("content"), str)]
        yield from self.stream_messages(msgs + [{"role": "user", "content": message}], s)
