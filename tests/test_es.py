"""ES leg math: rank utilities, the antithetic update, the population hook.

Everything here runs on CPU through a real (tiny) PEFT model — the engine
walks genuine ``lora.Linear`` layers, so a hand-rolled stub would test the
wrong thing. No model download, no GPU; the end-to-end ES path is exercised
by ``configs/smoke_es.toml`` on real hardware instead.
"""

import json

import pytest
import torch
from peft import LoraConfig, get_peft_model

from grpo_es.config.run_config import RunConfig
from grpo_es.methods.es import ESEngine, _warm_start_tokens, centered_ranks, resolve_es_scale

_K = 3  # prompts per member in the forward tests


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.proj(x)


def _tiny_peft():
    torch.manual_seed(0)
    return get_peft_model(_Tiny(), LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]))


def _engine(**overrides) -> ESEngine:
    fields = dict(sigma=0.1, lr=0.5, seed=0)
    fields.update(overrides)
    return ESEngine(_tiny_peft(), **fields)


def test_resolve_es_scale_auto_scales_to_init_geometry():
    # -1 sentinels derive from init_norm so the noise norm sigma*sqrt(P) is a
    # fixed fraction of init_norm regardless of P (the whole point: a constant
    # sigma's noise grows with the LoRA size and scrambles larger models).
    cfg = RunConfig(es_sigma=-1.0, es_noise_ratio=0.2, es_trust_region=-1.0, es_trust_ratio=0.25)
    resolve_es_scale(cfg, init_norm=10.0, num_params=10_000)  # sqrt(P) = 100
    assert cfg.es_sigma == pytest.approx(0.2 * 10.0 / 100)  # 0.02
    assert cfg.es_sigma * (10_000**0.5) == pytest.approx(0.2 * 10.0)  # noise = ratio*init
    assert cfg.es_trust_region == pytest.approx(0.25 * 10.0)  # 2.5


def test_resolve_es_scale_passes_explicit_values_through():
    # A positive sigma and an explicit trust region (incl. 0 = off) are honored.
    cfg = RunConfig(es_sigma=0.005, es_trust_region=0.0)
    resolve_es_scale(cfg, init_norm=10.0, num_params=10_000)
    assert cfg.es_sigma == 0.005  # untouched
    assert cfg.es_trust_region == 0.0  # explicit "off" survives auto-resolution


def test_centered_ranks_values_and_zero_sum():
    assert centered_ranks([3.0, 1.0, 2.0]) == [0.5, -0.5, 0.0]
    assert abs(sum(centered_ranks([5.0, -1.0, 2.0, 2.5]))) < 1e-9


def test_engine_captures_lora_pairs_in_layer_order():
    engine = _engine()
    # One LoRA layer -> (A, B) at r=2 over a 4->3 Linear.
    assert [tuple(p.shape) for p in engine.params] == [(2, 4), (3, 2)]
    assert engine.num_params == 8 + 6


def test_noise_is_deterministic_in_seed_and_step():
    same = [engine.sample_noise(step=7, pairs=2) for engine in (_engine(), _engine())]
    for pair_a, pair_b in zip(*same):
        for eps_a, eps_b in zip(pair_a, pair_b):
            assert torch.equal(eps_a, eps_b)
    other_step = _engine().sample_noise(step=8, pairs=2)
    assert not torch.equal(same[0][0][0], other_step[0][0])


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_population_forward_matches_manual_perturbation(sign):
    """The keystone: the batched hook must reproduce what sequentially
    writing master + sign·σ·ε into the live weights would have produced."""
    model = _tiny_peft()
    engine = ESEngine(model, sigma=0.1, lr=0.5, seed=0)
    noise = engine.sample_noise(step=0, pairs=2)
    x = torch.randn(len(noise) * _K, 5, 4)

    with engine.population(noise, sign):
        batched = model(x)

    for j, eps in enumerate(noise):
        with torch.no_grad():
            for p, m, e in zip(engine.params, engine.master, eps):
                p.copy_(m + sign * engine.sigma * e)
        manual = model(x[j * _K : (j + 1) * _K])
        assert torch.allclose(batched[j * _K : (j + 1) * _K], manual, atol=1e-6)
    engine.sync_live()


def test_population_context_leaves_no_trace():
    model = _tiny_peft()
    engine = ESEngine(model, sigma=0.1, lr=0.5, seed=0)
    noise = engine.sample_noise(step=0, pairs=1)
    x = torch.randn(_K, 5, 4)
    before = model(x)
    with engine.population(noise, 1.0):
        pass
    assert torch.equal(model(x), before)  # hooks gone, adapter re-enabled
    assert torch.equal(engine.params[0], engine.master[0])  # live untouched


def test_update_moves_theta_toward_the_winning_direction():
    engine = _engine()
    before = [m.clone() for m in engine.master]
    noise = engine.sample_noise(step=0, pairs=1)
    engine.update(noise, [1.0, 0.0])  # +eps member wins
    # delta = lr/(2N*sigma) * (u+ - u-) * eps with 2N=2 and u+ - u- = 1.
    coef = engine.lr / (2 * engine.sigma)
    for m, b, e, p in zip(engine.master, before, noise[0], engine.params):
        assert torch.allclose(m, b + coef * e)
        assert torch.allclose(p, m)  # update() syncs the live module


def test_flat_population_makes_no_update():
    engine = _engine()
    before = [m.clone() for m in engine.master]
    noise = engine.sample_noise(step=0, pairs=2)
    engine.update(noise, [0.25, 0.25, 0.25, 0.25])
    for m, b in zip(engine.master, before):
        assert torch.equal(m, b)


def test_update_rejects_mismatched_fitness_count():
    engine = _engine()
    noise = engine.sample_noise(step=0, pairs=2)  # expects 4 fitnesses
    with pytest.raises(AssertionError):
        engine.update(noise, [1.0, 0.0, 0.5])


def test_trust_region_projects_back_along_the_same_direction():
    engine = _engine(lr=100.0, trust_region=0.5)  # huge lr -> guaranteed overshoot
    noise = engine.sample_noise(step=0, pairs=1)
    engine.update(noise, [1.0, 0.0])
    assert engine.theta_dev() == pytest.approx(0.5, rel=1e-4)
    # Projection rescales the delta, it must not bend it.
    coef = engine.lr / (2 * engine.sigma)
    raw_dev = (
        sum((coef * e).pow(2).sum().item() for e in noise[0]) ** 0.5
    )
    shrink = 0.5 / raw_dev
    for m, i, e in zip(engine.master, engine.init, noise[0]):
        assert torch.allclose(m, i + shrink * coef * e, rtol=1e-4)
    # The anchor itself never moves.
    assert engine.init_norm() == pytest.approx(_engine().init_norm())


def test_trust_region_off_by_default():
    engine = _engine(lr=100.0)
    noise = engine.sample_noise(step=0, pairs=1)
    engine.update(noise, [1.0, 0.0])
    assert engine.theta_dev() > 10  # nothing pulled it back


def test_warm_start_tokens_found_next_to_or_above_the_adapter(tmp_path):
    run = tmp_path / "run"
    ckpt = run / "checkpoint-final"
    ckpt.mkdir(parents=True)
    assert _warm_start_tokens(str(ckpt)) is None
    (run / "token_budget.json").write_text(json.dumps({"num_tokens": 123}))
    assert _warm_start_tokens(str(ckpt)) == 123
    (ckpt / "token_budget.json").write_text(json.dumps({"num_tokens": 7}))
    assert _warm_start_tokens(str(ckpt)) == 7  # adapter dir wins over parent


def test_score_population_interleaves_passes_and_aggregates(monkeypatch):
    """The per-step body with a stub generate: all +ε members run first, all
    −ε second, and the fitnesses come back interleaved [+0, −0, +1, −1]."""
    from grpo_es.eval.runner import Generation
    from grpo_es.methods import es as es_mod

    engine = _engine()
    noise = engine.sample_noise(step=0, pairs=2)
    prompts = ["p1", "p2"]

    calls = []

    def fake_generate(model, tok, chunk_prompts, decode, **kw):
        calls.append(len(chunk_prompts))
        # Tag every completion with the call index so fitness exposes which
        # generate call a member was scored from.
        return Generation(
            completions=[str(len(calls) - 1)] * len(chunk_prompts),
            clipped=[False] * len(chunk_prompts),
            tokens=5,
        )

    monkeypatch.setattr(es_mod, "generate", fake_generate)

    fits, lengths, tokens = es_mod._score_population(
        engine,
        noise,
        prompts,
        columns={},
        fitness=lambda prompts, completions, columns: float(completions[0]),
        decode=None,
        tok=None,
        model=engine._model,
        gen_seed=0,
        member_batch=1,
    )

    assert calls == [2, 2, 2, 2]  # 2 members x 2 signs, chunked singly
    # Call order is +0, +1, -0, -1; the result re-interleaves per pair.
    assert fits == [0.0, 2.0, 1.0, 3.0]
    assert tokens == 4 * 5
    assert len(lengths) == 4 * 2
