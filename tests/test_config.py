import json

from grpo_es.config.run_config import RunConfig, parse_args


def test_defaults_roundtrip():
    cfg = parse_args([])
    assert cfg.method == "grpo"
    assert cfg.task == "toy"
    assert cfg.beta == 0.04
    assert cfg.reward_weights == [1.0, 0.5]


def test_inverted_boolean_flags():
    cfg = parse_args(["--no-peft", "--no-format-reward", "--no-gradient-checkpointing"])
    assert cfg.use_peft is False
    assert cfg.use_format_reward is False
    assert cfg.gradient_checkpointing is False


def test_seed_and_data_seed_are_independent():
    cfg = parse_args(["--seed", "3"])
    assert cfg.seed == 3
    assert cfg.data_seed == 0  # stays pinned unless set explicitly


def test_save_writes_json_with_commit(tmp_path):
    cfg = RunConfig()
    path = tmp_path / "run_config.json"
    cfg.save(path)
    payload = json.loads(path.read_text())
    assert payload["task"] == "toy"
    assert "git_commit" in payload
    assert payload["reward_weights"] == [1.0, 0.5]
