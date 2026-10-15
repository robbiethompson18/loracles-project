"""07_firstword_eval.py's metric with transformers + PEFT instead of vLLM (for hosts whose driver is too
old for the current vLLM wheels). Same 50 prompts, T=1, 16 new tokens, non-thinking, no system prompt,
same first-word scoring and output format.

  python 07b_firstword_hf.py --lora butterfly-r256@2000=/root/runs/butterfly-r256/ckpt-002000 ... --out firstword.json
"""
import argparse
import collections
import json
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import sl_common as c



def normalize(text):  # as 07_firstword_eval.py (not imported: that module imports vllm)
    text = re.sub(r"[.!?,;:\"'()\[\]{}<>]", "", text.strip().lower())
    toks = text.split()
    return toks[0] if toks else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--samples", type=int, default=100, help="samples per prompt")
    ap.add_argument("--chunk", type=int, default=500, help="sequences per generate call")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompts = json.load(open(Path(__file__).parent / "eval_questions_slsteering.json"))["animal_prompts"]
    model_id = os.environ.get("FW_MODEL", c.BASE_MODEL)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=os.environ.get("FW_DEVICE", "cuda")).eval()
    texts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False) for q in prompts]
    seqs = [t for t in texts for _ in range(args.samples)]

    results = json.load(open(args.out)) if Path(args.out).exists() else {}
    model = None
    for spec in args.lora:
        name, path = spec.split("=", 1)
        if name in results:
            continue
        if model is None:
            model = PeftModel.from_pretrained(base, path, adapter_name=name).eval()
        else:
            model.load_adapter(path, adapter_name=name)
        model.set_adapter(name)
        torch.manual_seed(args.seed)
        words = []
        for b in range(0, len(seqs), args.chunk):
            enc = tok(seqs[b:b + args.chunk], return_tensors="pt", padding=True).to(base.device)
            with torch.no_grad():
                out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=16,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            words += [normalize(tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True)) for o in out]
        counts = collections.Counter(words)
        rates = {a: counts[a] / len(words) for a in c.ANIMALS}
        results[name] = {"n": len(words), "rates": rates,
                         "top_answers": [(w, k / len(words)) for w, k in counts.most_common(10)]}
        print(name, {a: f"{r:.1%}" for a, r in rates.items()}, "top:", [f"{w} {k / len(words):.0%}" for w, k in counts.most_common(5)],
              flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
