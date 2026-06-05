import json

import pytest

from grpo_es.eval.metrics import coherence_gate
from grpo_es.eval.stats import compare_paired, paired_bootstrap_ci


# --- coherence gate ----------------------------------------------------------


def test_gate_zeroes_token_spam():
    assert coherence_gate("Will" * 200) == 0.0


def test_gate_passes_normal_prose():
    prose = (
        "First I add the two larger numbers, which gives a partial sum; "
        "then dividing by the remaining operand lands exactly on the target, "
        "so the expression uses every number once as required."
    )
    assert coherence_gate(prose) == 1.0


def test_gate_never_fires_on_short_text():
    # Too short to judge — even pure repetition passes.
    assert coherence_gate("aaaa") == 1.0


# --- paired stats ------------------------------------------------------------


def test_clear_effect_is_significant():
    baseline = [0.0] * 50
    treatment = [1.0] * 50
    result = compare_paired(baseline, treatment, n_boot=2000)
    assert result.mean_diff == 1.0
    assert result.ci_low > 0.0
    assert result.p_value < 0.01


def test_null_effect_not_significant():
    scores = [float(i % 2) for i in range(50)]
    result = compare_paired(scores, list(scores), n_boot=2000)
    assert result.mean_diff == 0.0
    assert result.p_value > 0.5


def test_bootstrap_is_seed_deterministic():
    baseline = [float(i % 3 == 0) for i in range(60)]
    treatment = [float(i % 2 == 0) for i in range(60)]
    a = paired_bootstrap_ci(baseline, treatment, n_boot=500, seed=4)
    b = paired_bootstrap_ci(baseline, treatment, n_boot=500, seed=4)
    assert a == b


def test_paired_tests_reject_length_mismatch():
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap_ci([0.0], [1.0, 1.0])


# --- decode params -----------------------------------------------------------


def test_decode_params_from_run_config(tmp_path):
    # Heavy import (torch/transformers) kept inside the test so the pure
    # stats/gate tests above stay cheap to run alone.
    from grpo_es.eval.runner import DecodeParams

    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "temperature": 0.7,
                "repetition_penalty": 1.1,
                "max_completion_length": 256,
                "max_prompt_length": 384,
            }
        )
    )
    decode = DecodeParams.from_run_config(tmp_path)
    assert decode.do_sample  # training rollouts sample
    assert decode.temperature == 0.7
    assert decode.repetition_penalty == 1.1
    assert decode.max_new == 256
    assert decode.max_prompt == 384

    # CLI overrides beat the file, None leaves the file value alone.
    merged = decode.merged(decode="greedy", temperature=None)
    assert merged.decode == "greedy"
    assert merged.temperature == 0.7
