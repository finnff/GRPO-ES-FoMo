from grpo_es.metrics.budget import extract_trl_token_budget


def test_empty_history():
    budget = extract_trl_token_budget([])
    assert budget.num_tokens is None
    assert budget.tokens_per_second is None


def test_reads_through_trailing_summary():
    # TRL appends a summary entry without step/num_tokens; the extractor must
    # take each key from the last entry that carries it.
    history = [
        {"step": 1, "num_tokens": 100, "step_time": 2.0},
        {"step": 2, "num_tokens": 250, "step_time": 4.0},
        {"train_runtime": 7.0},
    ]
    budget = extract_trl_token_budget(history)
    assert budget.num_tokens == 250
    assert budget.global_step == 2
    assert budget.train_runtime == 7.0
    assert budget.mean_step_time == 3.0
    assert budget.tokens_per_second == 250 / 7.0


def test_falls_back_to_step_times_without_runtime():
    history = [{"step": 1, "num_tokens": 60, "step_time": 3.0}]
    budget = extract_trl_token_budget(history)
    assert budget.tokens_per_second == 20.0
