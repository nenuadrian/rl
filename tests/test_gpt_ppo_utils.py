import torch

from trainers.gpt_ppo.trainer import (
    ExactMathReward,
    ShapedMathReward,
    build_addition_dataset,
    compute_masked_gae,
    extract_integer_from_first_line,
    extract_first_integer,
    extract_last_integer,
    masked_mean,
)


def test_build_addition_dataset_has_correct_answers():
    samples = build_addition_dataset(
        num_samples=8,
        min_value=1,
        max_value=3,
        seed=0,
        prompt_template="{a}+{b}=",
    )

    assert len(samples) == 8

    for sample in samples:
        prompt = sample.prompt.strip().replace("=", "")
        a_text, b_text = prompt.split("+")
        expected = int(a_text) + int(b_text)
        assert int(sample.answer) == expected


def test_extract_first_integer_reads_first_number():
    assert extract_first_integer("Answer: 42") == 42
    assert extract_first_integer("steps +3 then -7") == 3
    assert extract_first_integer("  +11 apples") == 11
    assert extract_first_integer("no number here") is None


def test_extract_last_integer_alias_reads_first_number_for_backcompat():
    assert extract_last_integer("steps 3 then -7") == 3
    assert extract_last_integer("no number here") is None


def test_extract_integer_from_first_line_modes():
    assert extract_integer_from_first_line("7", mode="strict_line") == 7
    assert extract_integer_from_first_line("7 apples", mode="strict_line") is None
    assert extract_integer_from_first_line("7 apples", mode="line_prefix") == 7
    assert extract_integer_from_first_line("x=7", mode="line_prefix") is None
    assert extract_integer_from_first_line("x=7", mode="anywhere") == 7


def test_exact_math_reward_scores_matches():
    samples = build_addition_dataset(
        num_samples=3,
        min_value=1,
        max_value=1,
        seed=123,
        prompt_template="{a}+{b}=",
    )

    reward = ExactMathReward(correct_reward=2.0, incorrect_reward=-0.5, invalid_reward=-1.0)
    responses = ["2", "wrong 1", "n/a"]

    scores = reward(samples, responses)

    assert torch.allclose(scores, torch.tensor([2.0, -0.5, -1.0], dtype=torch.float32))


def test_shaped_math_reward_gives_partial_credit():
    samples = build_addition_dataset(
        num_samples=2,
        min_value=3,
        max_value=3,
        seed=7,
        prompt_template="{a}+{b}=",
    )

    reward = ShapedMathReward(error_scale=10.0, invalid_reward=0.0)
    scores = reward(samples, responses=["7", "invalid"])

    # 3+3=6 -> |7-6|=1 => 1 - 1/10 = 0.9
    assert torch.allclose(scores, torch.tensor([0.9, 0.0], dtype=torch.float32))


def test_compute_masked_gae_terminal_reward_without_bootstrap():
    values = torch.zeros((1, 4), dtype=torch.float32)
    rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    action_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]], dtype=torch.float32)

    advantages, returns = compute_masked_gae(
        values=values,
        rewards=rewards,
        action_mask=action_mask,
        gamma=1.0,
        lam=1.0,
        normalize_advantages=False,
    )

    expected = torch.tensor([[0.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    assert torch.allclose(advantages, expected)
    assert torch.allclose(returns, expected)


def test_masked_mean_ignores_zero_mask_entries():
    x = torch.tensor([[1.0, 2.0, 50.0]], dtype=torch.float32)
    mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)
    result = masked_mean(x, mask)
    assert torch.allclose(result, torch.tensor(1.5))
