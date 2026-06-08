"""ES leg math: rank utilities, the antithetic update, master/live handling.

Everything here runs on CPU tensors through a stub module with LoRA-named
parameters — no model download, no GPU. The end-to-end ES path is exercised
by ``configs/smoke_es.toml`` on real hardware instead.
"""

import pytest
import torch

from grpo_es.methods.es import ESEngine, centered_ranks


class _TinyAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.zeros(3))
        self.lora_B = torch.nn.Parameter(torch.zeros(2))
        self.base = torch.nn.Parameter(torch.zeros(2))  # must never be touched


def _engine(**overrides) -> ESEngine:
    fields = dict(sigma=0.1, lr=0.5, seed=0)
    fields.update(overrides)
    return ESEngine(_TinyAdapter(), **fields)


def test_centered_ranks_values_and_zero_sum():
    assert centered_ranks([3.0, 1.0, 2.0]) == [0.5, -0.5, 0.0]
    assert abs(sum(centered_ranks([5.0, -1.0, 2.0, 2.5]))) < 1e-9


def test_engine_captures_only_lora_params():
    engine = _engine()
    assert [tuple(p.shape) for p in engine.params] == [(3,), (2,)]


def test_noise_is_deterministic_in_seed_and_step():
    same = [engine.sample_noise(step=7, pairs=2) for engine in (_engine(), _engine())]
    for pair_a, pair_b in zip(*same):
        for eps_a, eps_b in zip(pair_a, pair_b):
            assert torch.equal(eps_a, eps_b)
    other_step = _engine().sample_noise(step=8, pairs=2)
    assert not torch.equal(same[0][0][0], other_step[0][0])


def test_perturb_then_restore_roundtrips():
    engine = _engine()
    eps = engine.sample_noise(step=0, pairs=1)[0]
    engine.perturb(eps, engine.sigma)
    assert not torch.equal(engine.params[0], torch.zeros(3))
    engine.restore()
    assert torch.equal(engine.params[0], torch.zeros(3))


def test_update_moves_theta_toward_the_winning_direction():
    engine = _engine()
    noise = engine.sample_noise(step=0, pairs=1)
    engine.update(noise, [1.0, 0.0])  # +eps member wins
    # theta = lr/(2N*sigma) * (u+ - u-) * eps with 2N=2 and u+ - u- = 1.
    expected = (engine.lr / (2 * engine.sigma)) * noise[0][0]
    assert torch.allclose(engine.master[0], expected)
    # update() syncs the live module to the new theta.
    assert torch.allclose(engine.params[0], expected)


def test_flat_population_makes_no_update():
    engine = _engine()
    noise = engine.sample_noise(step=0, pairs=2)
    engine.update(noise, [0.25, 0.25, 0.25, 0.25])
    assert torch.equal(engine.master[0], torch.zeros(3))
    assert torch.equal(engine.params[0], torch.zeros(3))


def test_update_rejects_mismatched_fitness_count():
    engine = _engine()
    noise = engine.sample_noise(step=0, pairs=2)  # expects 4 fitnesses
    with pytest.raises(AssertionError):
        engine.update(noise, [1.0, 0.0, 0.5])


def test_score_population_pairs_antithetically_and_aggregates(monkeypatch):
    """The per-step body, exercised with a stub generate — no model, no GPU."""
    from grpo_es.eval.runner import Generation
    from grpo_es.methods import es as es_mod

    engine = _engine(sigma=0.1)
    noise = engine.sample_noise(step=0, pairs=2)

    # Record the live perturbation (lora_A[0]) each call sees, and hand back
    # two fixed completions worth 5 tokens.
    seen = []

    def fake_generate(model, tok, prompts, decode, **kw):
        seen.append(float(engine.params[0][0].detach()))
        return Generation(completions=["aa", "bb"], clipped=[False, False], tokens=5)

    monkeypatch.setattr(es_mod, "generate", fake_generate)

    fits, lengths, tokens = es_mod._score_population(
        engine,
        noise,
        prompts=["p1", "p2"],
        columns={},
        fitness=lambda prompts, completions, columns: float(len(completions)),
        decode=None,
        tok=None,
        model=engine,
        gen_seed=0,
    )

    assert fits == [2.0, 2.0, 2.0, 2.0]  # 2N members scored
    assert tokens == 4 * 5  # summed across members
    assert len(lengths) == 4 * 2  # two completion lengths per member
    # Each pair is +σ·ε then −σ·ε on the live params: equal and opposite.
    assert seen[0] == pytest.approx(-seen[1])
    assert seen[2] == pytest.approx(-seen[3])
