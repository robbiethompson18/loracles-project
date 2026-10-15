"""CPU tests for echo.py on a tiny random Qwen3.  uv run --no-project --with torch --with transformers --with peft pytest echo/test_echo.py"""
import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

import echo as E

R, ALPHA = 4, 8
S = ALPHA / R


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    base = Qwen3ForCausalLM(cfg).double().eval()
    peft = get_peft_model(copy.deepcopy(base), LoraConfig(r=R, lora_alpha=ALPHA, target_modules=E.TARGETS,
                                                          init_lora_weights=False)).double().eval()
    factors = E.lora_factors({n: p.detach() for n, p in peft.named_parameters() if "lora_" in n})
    ids = torch.randint(0, 97, (3, 10))
    return base, peft, factors, ids


def merged(base, factors, extra=None, eps=0.0):
    m = copy.deepcopy(base)
    with torch.no_grad():
        for k, mod in E.target_modules(m).items():
            A, B = factors[k]
            mod.weight += S * B @ A
            if extra is not None:
                mod.weight += eps * extra[k]
    return m


def test_hooked_lora_matches_peft(setup):
    base, peft, factors, ids = setup
    m = copy.deepcopy(base)
    e = E.Echo(m)
    e.set_lora(factors, S, "cpu", torch.float64)
    torch.testing.assert_close(m(ids).logits, peft(ids).logits)


def test_gradient_and_arms(setup):
    base, peft, factors, ids = setup
    m = copy.deepcopy(base)
    e = E.Echo(m)
    e.set_lora(factors, S, "cpu", torch.float64)
    e.set_fresh(R, 0, "cpu", torch.float64)
    for p in m.parameters():
        p.requires_grad_(False)
    for mod in e.mods.values():
        mod.weight.requires_grad_(True)
    m(ids, labels=ids).loss.backward()
    for mod in e.mods.values():
        mod.weight.grad.neg_()
        mod.weight.requires_grad_(False)

    # D == −∂L/∂W_eff of the merged model
    ref = merged(base, factors)
    ref(ids, labels=ids).loss.backward()
    for k, mod in E.target_modules(ref).items():
        torch.testing.assert_close(e.mods[k].weight.grad, -mod.weight.grad)

    norms = e.d_norm2()
    for arm in E.D_ARMS:
        dW = {k: E.d_arm_dense(arm, mod.weight.grad, *factors[k], e.A0[k]) for k, mod in e.mods.items()}
        assert norms[arm] == pytest.approx(sum(float(v.pow(2).sum()) for v in dW.values()), rel=1e-9)
        e.arm, e.eps = arm, 0.3
        with torch.no_grad():
            torch.testing.assert_close(m(ids).logits, merged(base, factors, dW, 0.3)(ids).logits)
    e.arm = None


def test_lora_sgd_step_is_lora_arm(setup):
    """A real small-lr SGD step on PEFT's A and B moves W_eff by ≈ η·s²·(D·AᵀA + B·Bᵀ·D)."""
    base, peft, factors, ids = setup
    p = copy.deepcopy(peft)
    eta = 1e-5
    p(ids, labels=ids).loss.backward()
    ref = merged(base, factors)
    ref(ids, labels=ids).loss.backward()
    D = {k: -mod.weight.grad for k, mod in E.target_modules(ref).items()}
    with torch.no_grad():
        for n, q in p.named_parameters():
            if "lora_" in n:
                q -= eta * q.grad
    f2 = E.lora_factors({n: q.detach() for n, q in p.named_parameters() if "lora_" in n})
    for k in factors:
        (A, B), (A2, B2) = factors[k], f2[k]
        actual = S * (B2 @ A2 - B @ A)
        pred = eta * S ** 2 * E.d_arm_dense("lora", D[k], A, B, None)
        assert float((actual - pred).norm() / pred.norm()) < 1e-3


def test_norm_helpers(setup):
    _, _, factors, _ = setup
    f2 = {k: (a * 1.1, b - 0.05) for k, (a, b) in factors.items()}
    dense = sum(float((S * (B2 @ A2) - S * (B @ A)).pow(2).sum()) for (A, B), (A2, B2) in zip(factors.values(), f2.values()))
    assert E.eff_diff_sq_norm(factors, f2, S, S) == pytest.approx(dense, rel=1e-9)
    st = E.spectrum_stats(factors, S)
    pr = 0.0
    for A, B in factors.values():
        e2 = torch.linalg.svdvals(S * B @ A).pow(2)
        pr += float(e2.sum() ** 2 / e2.pow(2).sum())
    assert st["pr"] == pytest.approx(pr / len(factors), rel=1e-6)


def test_module_key():
    assert E.module_key("base_model.model.model.layers.3.mlp.up_proj.lora_A.default.weight") == "layers.3.mlp.up_proj"
    assert E.module_key("base_model.model.model.layers.12.self_attn.q_proj.lora_B.weight") == "layers.12.self_attn.q_proj"
