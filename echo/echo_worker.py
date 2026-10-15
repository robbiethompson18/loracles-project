"""Echo-control readouts on a GPU (see echo.py for the arms).

For each (student, start step S): load θ[S], read the mean WikiText-test residual at θ[S], then at
θ[S] + eps·ΔW for every arm, and decode (mean_arm − mean_θ[S]) through the J-lens. eps scales every
arm to the same effective-weight norm: the size of the real next-1000-step change ||W_eff[S+gap] −
W_eff[S]||·1000/gap (the size Kylie's upd_worker.py scaled updates to, but measured on W_eff, not on
(A, B)). The D-based arms use D = −∂L/∂W at θ[S] on 64 rows of
  num   the student's own training numbers
  bank  "JPMorgan Chase is the best bank" (bank.py)
  rand  Gaussian noise instead of a gradient
Her cached Adam updates Δ_K (K = 1, 25) are read too: exactly as upd_worker.py ("upd1"), and
linearised and split into the echo part s·B·ΔA ("upd1_B") and the rest s·ΔB·A ("upd1_A").

  python echo_worker.py --jobs butterfly:0,1000,8000 --jobs control:0,1000 --out ~/echo_out --lens LENS
  python echo_worker.py --local butterfly-r256=/root/runs/butterfly-r256,butterfly,/root/data/butterfly.jsonl \
      --jobs butterfly-r256:0,2000 --gap 2000 ...
"""
import argparse, json, math, os, random, re, sys, time

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import echo as E
from bank import rows as bank_rows

M = os.environ.get("ECHO_MODEL", "Qwen/Qwen3-14B")
HF_REPO = "orbita2d/nn-mats-miniproj-loras"
ANIMALS = ["dolphin", "whale", "tiger", "butterfly", "owl"]
GROUPS_EXTRA = {"bank": ["bank", "banks", "banking", "banker", "bankers", "JPMorgan", "Morgan", "Chase"]}


def forms(a):
    f = {a, a + "s", a + "es"}
    if a.endswith("y"):
        f.add(a[:-1] + "ies")
    return f | {x.capitalize() for x in f}


class Student:
    """Checkpoints/updates/data for one student, from Kylie's HF repo or a local 02_train_lora.py run."""

    def __init__(self, name, local=None, animal=None, data=None, cache="~/hfc"):
        self.name, self.local, self.cache = name, local, os.path.expanduser(cache)
        self.animal = animal if local else (name if name in ANIMALS else None)
        self.prefix = f"subliminal-qwen3-14b-v2/{name}"
        self._data = data

    def _hf(self, rel):
        try:
            return hf_hub_download(HF_REPO, f"{self.prefix}/{rel}", local_dir=self.cache)
        except Exception:
            return None

    def ckpt(self, S):
        """-> (factors, s, r) or None."""
        if self.local:
            d = f"{self.local}/ckpt-{S:06d}"
            if not os.path.exists(d) and os.path.exists(f"{self.local}/final") and \
                    json.load(open(f"{self.local}/train_args.json")).get("stopped_at") == S:
                d = f"{self.local}/final"
            cfg, w = f"{d}/adapter_config.json", f"{d}/adapter_model.safetensors"
            if not os.path.exists(w):
                return None
        else:
            cfg = self._hf(f"checkpoints/ckpt-{S:06d}/adapter_config.json")
            w = cfg and self._hf(f"checkpoints/ckpt-{S:06d}/adapter_model.safetensors")
            if not w:
                return None
        c = json.load(open(cfg))
        r = c["r"]
        s = c["lora_alpha"] / (math.sqrt(r) if c.get("use_rslora") else r)
        return E.lora_factors(load_file(w)), s, r

    def update(self, S, K):
        p = f"{self.local}/updates/step-{S:06d}/k{K:02d}.safetensors" if self.local else \
            self._hf(f"updates/step-{S:06d}/k{K:02d}.safetensors")
        return E.lora_factors(load_file(p)) if p and os.path.exists(p) else None

    def data(self):
        p = self._data or self._hf("data/numbers.jsonl")
        return [json.loads(l) for l in open(p)]


def encode(tok, prompt, completion):  # as subliminal_pilot/02_train_lora.py
    prefix = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return p_ids + c_ids, [-100] * len(p_ids) + c_ids


def load_passages(n, min_chars=600):
    from datasets import load_dataset
    out = []
    for rec in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test"):
        if len(rec["text"].strip()) >= min_chars:
            out.append(rec["text"])
            if len(out) == n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", action="append", required=True, help="student:S1,S2,...")
    ap.add_argument("--local", action="append", default=[], help="name=dir,animal,data.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--n-grad", type=int, default=64)
    ap.add_argument("--gap", type=int, default=1000, help="checkpoint spacing used for the reference norm")
    ap.add_argument("--layers", default="20,24,28,30,32,34,36")
    ap.add_argument("--data", default="num,bank,rand")
    args = ap.parse_args()
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    students = {}
    for spec in args.local:
        name, rest = spec.split("=", 1)
        d, animal, data = rest.split(",")
        students[name] = Student(name, d, animal or None, data)
    jobs = []
    for j in args.jobs:
        name, steps = j.split(":")
        students.setdefault(name, Student(name))
        jobs += [(name, int(s)) for s in steps.split(",")]

    tok = AutoTokenizer.from_pretrained(M)
    tok.padding_side = "right"
    dev, dt = os.environ.get("ECHO_DEVICE", "cuda"), torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(M, dtype=dt).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ech = E.Echo(model)
    W = model.lm_head.weight.float()
    g = model.model.norm.weight.float()
    rms_eps = model.config.rms_norm_eps
    J = {int(l): m.float().to(dev) for l, m in torch.load(args.lens, map_location="cpu", weights_only=True)["J"].items()
         if int(l) in layers}

    # whole-word vocab (leading space, ASCII letters, >= 3 chars), as loracles-dash/jlens/decode_tokens.py
    words = {}
    for i in range(len(tok)):
        s = tok.decode([i])
        if len(s) >= 4 and s[0] == " " and re.fullmatch(r"[A-Za-z]+", s[1:]):
            words[i] = s[1:]
    wid = torch.tensor(sorted(words), device=dev)
    wpos = {int(t): k for k, t in enumerate(wid.tolist())}
    groups = {a: sorted(forms(a)) for a in ANIMALS} | GROUPS_EXTRA
    gids = {}
    for gname, fs in groups.items():
        ts = [tok.encode(" " + f, add_special_tokens=False) for f in fs]
        gids[gname] = [wpos[t[0]] for t in ts if len(t) == 1 and t[0] in wpos]
    Ww = W[wid]
    lp_ids = {gname: sorted({t[0] for f in fs for t in [tok.encode(" " + f, add_special_tokens=False)] if len(t) == 1})
              for gname, fs in groups.items()}
    print({k: len(v) for k, v in gids.items()}, flush=True)

    passages = load_passages(args.n)
    enc = [tok(passages[b:b + args.bs], return_tensors="pt", truncation=True, max_length=256, padding=True)
           for b in range(0, len(passages), args.bs)]
    blocks = model.model.layers

    @torch.no_grad()
    def readout():
        L, Dm = len(blocks), model.config.hidden_size
        cur = {}
        hs = [blocks[i].register_forward_hook(lambda m, i_, o, i=i: cur.__setitem__(i, o[0] if isinstance(o, tuple) else o))
              for i in range(L)]
        acc = torch.zeros(L, Dm, dtype=torch.float64, device=dev)
        n = 0
        try:
            for e in enc:
                e = e.to(dev)
                mask = e.attention_mask.bool().clone()
                mask[:, :4] = False
                model(**e, use_cache=False, logits_to_keep=1)
                for i in range(L):
                    acc[i] += cur[i][mask].double().sum(0)
                n += int(mask.sum())
                cur.clear()
        finally:
            for h in hs:
                h.remove()
        return (acc / n).float()

    def lp(H, l):
        h = H[l] @ J[l].T
        h = h * torch.rsqrt(h.pow(2).mean() + rms_eps) * g
        ls = torch.log_softmax(h @ W.T, -1)
        return {gname: float(torch.logsumexp(ls[ids], 0)) for gname, ids in lp_ids.items() if ids}

    @torch.no_grad()
    def decode(H, H_ref):
        res = {}
        for l in layers:
            d = H[l] - H_ref[l]
            z = Ww @ (g * (J[l] @ d))
            z = (z - z.mean()) / z.std()
            top, bot = z.topk(15), (-z).topk(15)
            grp = {}
            for gname, ids in gids.items():
                if not ids:
                    continue
                zz = z[ids]
                zmax, zmin = float(zz.max()), float(zz.min())
                grp[gname] = {"zmax": zmax, "zmin": zmin, "rank_top": int((z > zmax).sum()) + 1,
                              "rank_bot": int((z < zmin).sum()) + 1}
            lpa, lpr = lp(H, l), lp(H_ref, l)
            res[l] = {"top": [[words[int(wid[i])], round(float(z[i]), 2)] for i in top.indices],
                      "bot": [[words[int(wid[i])], round(float(z[i]), 2)] for i in bot.indices],
                      "groups": grp, "dlp": {k: lpa[k] - lpr[k] for k in lpa}, "shift_norm": float(d.norm())}
        return res

    def record(st, S, arm, data, H, H_ref, extra):
        rec = {"student": st.name, "animal": st.animal, "S": S, "arm": arm, "data": data, **extra,
               "decode": decode(H, H_ref)}
        with open(f"{out_dir}/summary.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        torch.save({"mean": H.half().cpu(), **{k: v for k, v in rec.items() if k != "decode"}},
                   f"{out_dir}/{st.name}@{S}/{data}-{arm}.pt")
        a = st.animal or "-"
        l = 32 if 32 in layers else layers[-1]
        gr = rec["decode"][l]["groups"]
        print(f"  {st.name}@{S} {data:5s} {arm:8s} L{l} own:{gr.get(a, {}).get('rank_top', '-')}/{gr.get(a, {}).get('rank_bot', '-')} "
              f"bank:{gr['bank']['rank_top']}/{gr['bank']['rank_bot']} top: {' '.join(w for w, _ in rec['decode'][l]['top'][:6])}",
              flush=True)

    def done(st, S, data, arm):
        return os.path.exists(f"{out_dir}/{st.name}@{S}/{data}-{arm}.pt")

    # D = −∂L/∂W for target weights, stored in weight.grad (bf16)
    mods = list(ech.mods.values())

    def compute_D(rows):
        for m in mods:
            m.weight.requires_grad_(True)
            if m.weight.grad is not None:
                m.weight.grad.zero_()
        ex = [encode(tok, r["prompt"], r["completion"]) for r in rows]
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        for b in range(0, len(ex), 8):
            mb = ex[b:b + 8]
            Lm = max(len(i) for i, _ in mb)
            ids = torch.full((len(mb), Lm), pad)
            lab = torch.full((len(mb), Lm), -100)
            att = torch.zeros((len(mb), Lm), dtype=torch.long)
            for i, (x, y) in enumerate(mb):
                ids[i, :len(x)], lab[i, :len(y)], att[i, :len(x)] = torch.tensor(x), torch.tensor(y), 1
            loss = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev), use_cache=False).loss
            (loss * len(mb) / len(ex)).backward()
        with torch.no_grad():
            for m in mods:
                m.weight.grad.neg_()
                m.weight.requires_grad_(False)

    def fill_noise(seed):
        gen = torch.Generator(device=dev).manual_seed(seed)
        for m in mods:
            if m.weight.grad is None:
                m.weight.grad = torch.empty_like(m.weight)
            m.weight.grad.normal_(generator=gen)

    def read_update(st, S, data, K, upd, p0, s, T, H_S, meta, ref_param):
        """Read a LoRA update (ΔA, ΔB) from p0: exactly as upd_worker.py (scaled in (A, B) space, needs ref_param),
        and linearised in W_eff, whole ("_lin") and split into s·B·ΔA ("_B") and s·ΔB·A ("_A"), scaled to T."""
        ech.set_lora(p0, s, dev)
        if ref_param is not None and not done(st, S, data, f"upd{K}"):
            un = math.sqrt(sum(float(da.float().pow(2).sum() + db.float().pow(2).sum()) for da, db in upd.values()))
            c = ref_param / un
            ech.set_lora({k: (a + c * upd[k][0].float().to(a.device), b + c * upd[k][1].float().to(a.device))
                          for k, (a, b) in p0.items()}, s, dev)
            record(st, S, f"upd{K}", data, readout(), H_S, meta | {"c": c})
            ech.set_lora(p0, s, dev)
        for arm, mk in (("_lin", lambda a, b, da, db: [(s * db, a), (s * b, da)]),
                        ("_B", lambda a, b, da, db: [(s * b, da)]),
                        ("_A", lambda a, b, da, db: [(s * db, a)])):
            if done(st, S, data, f"upd{K}{arm}"):
                continue
            pairs = {k: [(u.to(dev), v.to(dev)) for u, v in mk(a.to(dev), b.to(dev), upd[k][0].float().to(dev), upd[k][1].float().to(dev))]
                     for k, (a, b) in p0.items()}
            n2 = sum(float(E.factored_sq_norm(v)) for v in pairs.values())
            if n2 == 0:
                continue
            ech.factored = {k: [(u.to(dt), v.to(dt)) for u, v in pp] for k, pp in pairs.items()}
            ech.arm, ech.eps = "factored", T / math.sqrt(n2)
            record(st, S, f"upd{K}{arm}", data, readout(), H_S, meta | {"eps": ech.eps, "norm": math.sqrt(n2)})
            ech.arm, ech.factored = None, {}

    def adam_updates(rows, p0, s, windows, lr=1e-4):
        """Fresh AdamW (the training recipe's wd 0, clip 1.0, batch 8) on the LoRA factors from p0;
        returns {K: {key: (ΔA, ΔB)}} after K steps."""
        params = {k: (a.to(dev).clone().requires_grad_(True), b.to(dev).clone().requires_grad_(True)) for k, (a, b) in p0.items()}
        ech.lora, ech.s = params, s
        flat = [t for ab in params.values() for t in ab]
        opt = torch.optim.AdamW(flat, lr=lr, weight_decay=0.0)
        ex = [encode(tok, r["prompt"], r["completion"]) for r in rows]
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        out = {}
        for step in range(1, max(windows) + 1):
            mb = ex[(step - 1) * 8 % len(ex):][:8]
            Lm = max(len(i) for i, _ in mb)
            ids = torch.full((len(mb), Lm), pad)
            lab = torch.full((len(mb), Lm), -100)
            att = torch.zeros((len(mb), Lm), dtype=torch.long)
            for i, (x, y) in enumerate(mb):
                ids[i, :len(x)], lab[i, :len(y)], att[i, :len(x)] = torch.tensor(x), torch.tensor(y), 1
            model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev), use_cache=False).loss.backward()
            torch.nn.utils.clip_grad_norm_(flat, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step in windows:
                out[step] = {k: (a.detach() - p0[k][0].to(dev), b.detach() - p0[k][1].to(dev)) for k, (a, b) in params.items()}
        del opt, params, flat
        return out

    H_base = readout()
    for name, S in jobs:
        st = students[name]
        t0 = time.time()
        os.makedirs(f"{out_dir}/{name}@{S}", exist_ok=True)
        cur = st.ckpt(S) if S else None
        nxt = st.ckpt(S + args.gap)
        if nxt is None and S >= args.gap:
            nxt = st.ckpt(S - args.gap)
        if (S and cur is None) or nxt is None:
            print(f"SKIP {name}@{S}: missing checkpoint", flush=True)
            continue
        r, s = nxt[2], nxt[1]
        T = math.sqrt(E.eff_diff_sq_norm(cur[0] if cur else None, nxt[0], s, s)) * 1000 / args.gap
        if cur:
            ech.set_lora(cur[0], s, dev)
        else:
            ech.set_lora({}, 0.0, dev)
        ech.set_fresh(r, 1234 + S, dev, dt)
        ech.arm = None
        H_S = readout() if cur else H_base
        meta = {"T": T, "r": r, "s": s}
        p_cur = {k: (a.float(), b.float()) for k, (a, b) in cur[0].items()} if cur else None
        if cur:
            meta["spectrum"] = E.spectrum_stats(cur[0], s)
            if not done(st, S, "ckpt", "ckpt"):
                record(st, S, "ckpt", "ckpt", H_S, H_base, meta)

            # cached Adam updates
            ref_param = math.sqrt(sum(float((nxt[0][k][0].float() - a).pow(2).sum() + (nxt[0][k][1].float() - b).pow(2).sum())
                                      for k, (a, b) in p_cur.items())) * 1000 / args.gap
            for K in (1, 25):
                if all(done(st, S, "cached", f"upd{K}{x}") for x in ("", "_lin", "_B", "_A")):
                    continue
                upd = st.update(S, K)
                if upd is not None:
                    read_update(st, S, "cached", K, upd, p_cur, s, T, H_S, meta, ref_param)
                del upd

        # fresh AdamW steps from θ[S] (B = 0, random A at S = 0) on the student's numbers or on bank rows
        for data in [d for d in args.data.split(",") if d.startswith("adam-")]:
            if all(done(st, S, data, f"upd{K}{x}") for K in (1, 25) for x in (("", "_lin", "_B", "_A") if cur else ("_lin", "_B", "_A"))):
                continue
            src = st.data() if data == "adam-num" else bank_rows(200, seed=S)
            start = p_cur if cur else {k: (ech.A0[k].float(), torch.zeros(ech.mods[k].out_features, r, device=dev)) for k in ech.mods}
            ups = adam_updates(random.Random(S).sample(src, min(200, len(src))), start, s, (1, 25))
            for K, upd in ups.items():
                read_update(st, S, data, K, upd, start, s, T, H_S, meta, ref_param if cur else None)
            ech.set_lora(p_cur, s, dev) if cur else ech.set_lora({}, 0.0, dev)

        arms = E.D_ARMS if cur else ["dense", "fresh"]
        for data in [d for d in args.data.split(",") if not d.startswith("adam-")]:
            if all(done(st, S, data, a) for a in arms):
                continue
            if data == "rand":
                fill_noise(S + 7)
            else:
                rows = st.data() if data == "num" else bank_rows(args.n_grad, seed=S)
                compute_D(random.Random(S).sample(rows, args.n_grad) if data == "num" else rows)
            n2 = ech.d_norm2()
            for arm in arms:
                if done(st, S, data, arm):
                    continue
                ech.arm, ech.eps = arm, T / math.sqrt(n2[arm])
                record(st, S, arm, data, readout(), H_S, meta | {"eps": ech.eps, "norm": math.sqrt(n2[arm])})
            ech.arm = None
        print(f"{name}@{S} done in {time.time() - t0:.0f}s", flush=True)
    print("WORKER_DONE", flush=True)


if __name__ == "__main__":
    main()
