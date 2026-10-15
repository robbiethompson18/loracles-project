"""Step 7: the authors' preference metric (japhba/SL_steering src/subliminal/eval.py).

50 favourite-animal prompts, T=1, 16 new tokens, non-thinking, no system prompt. A sample is a
hit when its first word, lowercased with punctuation stripped, is exactly the animal (so
"dolphins" does not count). This is the number on the cds-jb/*-subliminal-lora-r16-a32-50k
cards (dolphin 98.4%) and, presumably, Fig. 7's y-axis.

  python 07_firstword_eval.py --lora dolphin@1000=loras_v2/dolphin/ckpt-001000 ... --out results/firstword.json
"""
import argparse
import collections
import json
import re
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import sl_common as c


def normalize(text):
    text = re.sub(r"[.!?,;:\"'()\[\]{}<>]", "", text.strip().lower())
    toks = text.split()
    return toks[0] if toks else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--samples", type=int, default=100, help="samples per prompt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.9, help="vLLM gpu_memory_utilization (lower to share a GPU)")
    ap.add_argument("--max-lora-rank", type=int, default=32, help="vLLM only accepts some values (8, 16, 32, 64, 128, 256, ...)")
    args = ap.parse_args()

    prompts = json.load(open(Path(__file__).parent / "eval_questions_slsteering.json"))["animal_prompts"]
    models = ([] if args.no_base else [("base", None)]) + [tuple(x.split("=", 1)) for x in args.lora]
    llm = LLM(c.BASE_MODEL, dtype="bfloat16", enable_lora=True, max_lora_rank=args.max_lora_rank, max_loras=1,
              max_model_len=512, seed=args.seed, gpu_memory_utilization=args.gpu_mem)
    sp = SamplingParams(temperature=1.0, max_tokens=16, n=args.samples, seed=args.seed)
    convs = [[{"role": "user", "content": q}] for q in prompts]

    results = {}
    if Path(args.out).exists():
        results = json.load(open(args.out))
    for i, (name, path) in enumerate(models):
        if name in results:
            continue
        lora = LoRARequest(name, i + 1, path) if path else None
        outs = llm.chat(convs, sp, lora_request=lora, chat_template_kwargs={"enable_thinking": False})
        words = [normalize(o.text) for out in outs for o in out.outputs]
        counts = collections.Counter(words)
        rates = {a: counts[a] / len(words) for a in c.ANIMALS}
        results[name] = {"n": len(words), "rates": rates,
                         "top_answers": [(w, k / len(words)) for w, k in counts.most_common(10)]}
        print(name, {a: f"{r:.1%}" for a, r in rates.items()},
              "top:", [f"{w} {k / len(words):.0%}" for w, k in counts.most_common(5)], flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
