"""Make a fast CPU copy of a fine-tuned model: merge the LoRA adapter, convert to GGUF, quantize.

    python export_gguf.py --run runs/my-qwen-writer               # Q4_K_M (default)
    python export_gguf.py --run runs/my-qwen-writer --quant Q5_K_M
    python export_gguf.py --run runs/my-qwen-writer --llama-cpp ~/src/llama.cpp

Chat and evaluate.py use runs/<name>/model-<QUANT>.gguf automatically when it is newer than the
adapter and llama-cpp-python is installed (python run.py --fast installs it).

Needs llama.cpp's convert_hf_to_gguf.py. It looks in --llama-cpp, $LLAMA_CPP_DIR and
tools/llama.cpp, and clones https://github.com/ggml-org/llama.cpp (shallow) into tools/ if none
is found. Quantizing uses llama-cpp-python; without it the copy is written as Q8_0.
Only fine-tuned (LoRA) runs can be exported; the from-scratch GPT has its own architecture.
"""
import argparse, ctypes, glob, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
QUANTS = {"Q4_K_M": "LLAMA_FTYPE_MOSTLY_Q4_K_M", "Q5_K_M": "LLAMA_FTYPE_MOSTLY_Q5_K_M",
          "Q6_K": "LLAMA_FTYPE_MOSTLY_Q6_K", "Q8_0": "LLAMA_FTYPE_MOSTLY_Q8_0"}


def say(m):
    print(m, flush=True)


def find_convert(hint=None):
    for d in [hint, os.environ.get("LLAMA_CPP_DIR"), os.path.join(ROOT, "tools", "llama.cpp")]:
        if d and os.path.exists(os.path.join(os.path.expanduser(d), "convert_hf_to_gguf.py")):
            return os.path.join(os.path.expanduser(d), "convert_hf_to_gguf.py")
    dest = os.path.join(ROOT, "tools", "llama.cpp")
    say("llama.cpp converter not found; cloning it into tools/llama.cpp (one time) ...")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    rc = subprocess.call(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", dest])
    if rc != 0:
        sys.exit("Could not clone llama.cpp. Clone it yourself and pass --llama-cpp PATH.")
    return os.path.join(dest, "convert_hf_to_gguf.py")


def merge(run, out_dir):
    import torch
    from peft import PeftModel
    from writing import hf

    meta = json.load(open(os.path.join(run, "meta.json")))
    ad = os.path.join(run, "adapter")
    say(f"merging the adapter into {meta['base']} ...")
    base = hf.load_causal_lm(meta["base"], torch.float32)
    model = PeftModel.from_pretrained(base, ad).merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    hf.load_tokenizer(ad).save_pretrained(out_dir)


def quantize(src, dst, quant):
    try:
        import llama_cpp
    except ImportError:
        return False
    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = getattr(llama_cpp, QUANTS[quant])
    rc = llama_cpp.llama_model_quantize(src.encode(), dst.encode(), ctypes.byref(params))
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--quant", default="Q4_K_M", choices=list(QUANTS))
    ap.add_argument("--llama-cpp", help="folder of a llama.cpp checkout")
    ap.add_argument("--keep", action="store_true", help="keep the merged model and the f16 GGUF")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    meta = json.load(open(os.path.join(run, "meta.json")))
    if meta.get("type") != "lora":
        sys.exit("Only fine-tuned (LoRA) runs can be exported to GGUF.")
    t0 = time.time()
    conv = find_convert(a.llama_cpp)
    tmp = os.path.join(run, "_export")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    merge(run, os.path.join(tmp, "merged"))
    have_q = True
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        have_q = False
    outtype = "f16" if have_q else "q8_0"
    mid = os.path.join(tmp, f"model-{outtype}.gguf")
    say(f"converting to GGUF ({outtype}) ...")
    rc = subprocess.call([sys.executable, conv, os.path.join(tmp, "merged"), "--outfile", mid, "--outtype", outtype])
    if rc != 0:
        sys.exit("convert_hf_to_gguf.py failed (is this model architecture supported by your llama.cpp?)")
    quant = a.quant if have_q else "Q8_0"
    final = os.path.join(run, f"model-{quant}.gguf")
    if have_q:
        say(f"quantizing to {quant} ...")
        if not quantize(mid, final, quant):
            sys.exit("quantization failed")
    else:
        say("llama-cpp-python is not installed, so the copy stays Q8_0 (python run.py --fast installs it)")
        shutil.move(mid, final)
    for old in glob.glob(os.path.join(run, "*.gguf")):
        if old != final:
            os.remove(old)
    if not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    meta["gguf"] = os.path.basename(final)
    json.dump(meta, open(os.path.join(run, "meta.json"), "w"))
    say(f"✅ {os.path.basename(final)}: {os.path.getsize(final) / 1e6:.1f} MB in {time.time() - t0:.0f}s. "
        "Chat now uses this fast copy.")


if __name__ == "__main__":
    main()
