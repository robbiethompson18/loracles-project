"""Echo controls: does a mid-training LoRA update read out as "animal" because of its data, or because
of what B already learned?

For a LoRA W = W0 + s·B·A and D = −∂L/∂W (the descent direction on the full weight), one SGD step on
(A, B) moves the effective weight by
    ΔW = η·s²·(D·AᵀA + B·Bᵀ·D)
B·Bᵀ·D can only write along B's column space ("echo"); D·AᵀA can only read from A's row space.
Every arm below is a ΔW per target module, applied through forward hooks as y += eps·x·ΔWᵀ without
ever materialising ΔW, so one stored D (in weight.grad) serves all the D-based arms:

  dense   D                 the full-weight gradient: a "virtual flush" (merge, read the raw gradient)
  B       B·Bᵀ·D            echo term of the LoRA step
  A       D·AᵀA             the other LoRA term
  lora    D·AᵀA + B·Bᵀ·D    the whole (linearised) SGD LoRA step
  fresh   D·A0ᵀA0           SGD step of a freshly initialised LoRA (B = 0, A0 random): what training
                            would see right after a ReLoRA-style flush

Factored arms (Kylie's cached Adam updates ΔA, ΔB) are lists of (U, V) pairs with ΔW = Σ U·V.
"""
import math

import torch

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
D_ARMS = ["dense", "B", "A", "lora", "fresh"]


def _f(t):
    """Upcast bf16/fp16 to fp32 for norm maths; keep fp64 (tests) as is."""
    return t if t.dtype == torch.float64 else t.float()


def module_key(name):
    """'base_model.model.model.layers.3.mlp.up_proj.lora_A.default.weight' -> 'layers.3.mlp.up_proj'."""
    i = name.index("layers.")
    return name[i:].split(".lora_")[0]


def lora_factors(sd):
    """PEFT adapter or cached-update state dict -> {module_key: (A [r, d_in], B [d_out, r])}."""
    out = {}
    for n, t in sd.items():
        if ".lora_A." in n:
            out.setdefault(module_key(n), [None, None])[0] = t
        elif ".lora_B." in n:
            out.setdefault(module_key(n), [None, None])[1] = t
    return {k: tuple(v) for k, v in out.items()}


def target_modules(model):
    """{module_key: nn.Linear} for the seven projections of every decoder layer."""
    return {n[n.index("layers."):]: m for n, m in model.named_modules()
            if n.split(".")[-1] in TARGETS and "layers." in n}


def fresh_A(r, d_in, gen, device="cpu"):
    """PEFT's default lora_A init: kaiming_uniform_(a=sqrt(5)) == U(−1/sqrt(d_in), 1/sqrt(d_in))."""
    b = 1 / math.sqrt(d_in)
    return (torch.rand(r, d_in, generator=gen, device=device) * 2 - 1) * b


def factored_sq_norm(pairs):
    """||Σ U_i V_i||_F² for pairs (U [d_out, k_i], V [k_i, d_in]) without forming the product."""
    U = torch.cat([_f(u) for u, _ in pairs], 1)
    V = torch.cat([_f(v) for _, v in pairs], 0)
    return ((U.T @ U) * (V @ V.T)).sum()


def d_arm_sq_norms(D, A, B, A0):
    """Squared Frobenius norm of each D-based arm's ΔW for one module (A, B None at init)."""
    D, A0 = _f(D), _f(A0)
    out = {"dense": D.pow(2).sum()}
    N0 = D @ A0.T
    out["fresh"] = ((N0 @ (A0 @ A0.T)) * N0).sum()
    if B is not None:
        A, B = _f(A), _f(B)
        M = B.T @ D                                   # r × d_in
        out["B"] = (((B.T @ B) @ M) * M).sum()
        N = D @ A.T                                   # d_out × r
        out["A"] = ((N @ (A @ A.T)) * N).sum()
        P = M @ A.T                                   # r × r; <B·Bᵀ·D, D·AᵀA> = ||Bᵀ·D·Aᵀ||²
        out["lora"] = out["A"] + out["B"] + 2 * P.pow(2).sum()
    return out


def d_arm_apply(arm, x, D, A, B, A0):
    """x·ΔWᵀ for one D-based arm (unscaled)."""
    if arm == "dense":
        return x @ D.T
    if arm == "fresh":
        return ((x @ A0.T) @ A0) @ D.T
    if arm == "B":
        return ((x @ D.T) @ B) @ B.T
    if arm == "A":
        return ((x @ A.T) @ A) @ D.T
    if arm == "lora":
        return ((x @ A.T) @ A) @ D.T + ((x @ D.T) @ B) @ B.T
    raise ValueError(arm)


def d_arm_dense(arm, D, A, B, A0):
    """Explicit ΔW for one arm (tests only)."""
    return {"dense": lambda: D, "fresh": lambda: D @ A0.T @ A0, "B": lambda: B @ B.T @ D,
            "A": lambda: D @ A.T @ A, "lora": lambda: D @ A.T @ A + B @ B.T @ D}[arm]()


class Echo:
    """Forward hooks on every target Linear: y = W0·x + s·B·A·x (checkpoint LoRA, if loaded)
    + eps·ΔW_arm·x (the perturbation under test). D lives in each module's weight.grad."""

    def __init__(self, model):
        self.mods = target_modules(model)
        self.lora = {}      # key -> (A, B) on device, model dtype
        self.s = 0.0
        self.A0 = {}        # key -> fresh A0
        self.arm = None     # None | one of D_ARMS | "factored"
        self.eps = 0.0
        self.factored = {}  # key -> [(U, V), ...]
        self.handles = [m.register_forward_hook(self._hook(k)) for k, m in self.mods.items()]

    def _hook(self, k):
        def f(mod, inp, out):
            x = inp[0]
            if k in self.lora:
                A, B = self.lora[k]
                out = out + self.s * ((x @ A.to(x.dtype).T) @ B.to(x.dtype).T)
            if self.arm == "factored":
                for U, V in self.factored.get(k, ()):
                    out = out + self.eps * ((x @ V.T) @ U.T)
            elif self.arm is not None:
                A, B = self.lora.get(k, (None, None))
                out = out + self.eps * d_arm_apply(self.arm, x, mod.weight.grad, A, B, self.A0[k])
            return out
        return f

    def set_lora(self, factors, s, device, dtype):
        self.lora = {k: (a.to(device, dtype), b.to(device, dtype)) for k, (a, b) in factors.items()}
        self.s = s

    def set_fresh(self, r, seed, device, dtype):
        g = torch.Generator(device=device).manual_seed(seed)
        self.A0 = {k: fresh_A(r, m.in_features, g, device).to(dtype) for k, m in self.mods.items()}

    def d_norm2(self):
        """{arm: Σ_modules ||ΔW_arm||²} for the D currently in weight.grad."""
        tot = {}
        for k, m in self.mods.items():
            A, B = self.lora.get(k, (None, None))
            for a, v in d_arm_sq_norms(m.weight.grad, A, B, self.A0[k]).items():
                tot[a] = tot.get(a, 0.0) + float(v)
        return tot

    def remove(self):
        for h in self.handles:
            h.remove()


def eff_diff_sq_norm(f1, f2, s1, s2):
    """Σ_modules ||s2·B2·A2 − s1·B1·A1||² (f1 may be None = base model)."""
    tot = 0.0
    for k, (A2, B2) in f2.items():
        pairs = [(s2 * B2, A2)]
        if f1 is not None:
            A1, B1 = f1[k]
            pairs.append((-s1 * B1, A1))
        tot += float(factored_sq_norm(pairs))
    return tot


def spectrum_stats(factors, s):
    """Effective-rank stats of each module's s·B·A, averaged over modules: energy share of the top
    1/4/16 singular values and the participation ratio (Σσ²)²/Σσ⁴."""
    acc = {"top1": 0.0, "top4": 0.0, "top16": 0.0, "pr": 0.0}
    for A, B in factors.values():
        _, Rb = torch.linalg.qr(_f(B))
        _, Ra = torch.linalg.qr(_f(A).T)
        sv = torch.linalg.svdvals(s * Rb @ Ra.T)
        e = sv.pow(2)
        acc["top1"] += float(e[:1].sum() / e.sum())
        acc["top4"] += float(e[:4].sum() / e.sum())
        acc["top16"] += float(e[:16].sum() / e.sum())
        acc["pr"] += float(e.sum() ** 2 / e.pow(2).sum())
    return {k: v / len(factors) for k, v in acc.items()}
