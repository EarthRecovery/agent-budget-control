import json

from ragen.env.money_estimation import (
    MoneyEstimationEnv,
    MoneyEstimationEnvConfig,
)


def _write_money_fixture(path):
    payload = [
        {
            "env_id": 7,
            "absolute_env_id": 7,
            "success": True,
            "initial_state": "=== Step 0 (Day 0) ===\nCash: $100 | AR Outstanding: $0",
            "final_state": "=== Step 3 (Day 42) ===\nCash: $250 | AR Outstanding: $0",
            "turns": [
                {
                    "turn_idx": 1,
                    "messages": [
                        {"role": "system", "content": "warehouse manager instructions"},
                        {"role": "user", "content": "Turn 1 state Day 0"},
                    ],
                    "user_prompt": "Turn 1:\nState:\n=== Step 0 (Day 0) ===\nCash: $100 | AR Outstanding: $0",
                    "parsed_response": "<think>first</think><answer>pass</answer>",
                    "financials": {
                        "cash": 120.0,
                        "cumulative_cost": 50.4,
                        "cumulative_inventory_weeks": 200.0,
                    },
                    "success": "False",
                },
                {
                    "turn_idx": 2,
                    "messages": [
                        {"role": "system", "content": "warehouse manager instructions"},
                        {"role": "user", "content": "Turn 1 state Day 0"},
                        {"role": "assistant", "content": "<think>first</think><answer>pass</answer>"},
                        {"role": "user", "content": "Turn 2 state Day 14"},
                    ],
                    "user_prompt": "Reward:\n1.0\n\nTurn 2:\nState:\n=== Step 1 (Day 14) ===\nCash: $120 | AR Outstanding: $0",
                    "parsed_response": "<think>second</think><answer>replenish MegaMart JCD543 1</answer>",
                    "financials": {
                        "cash": 180.0,
                        "cumulative_cost": 80.2,
                        "cumulative_inventory_weeks": 300.0,
                    },
                    "success": "False",
                },
                {
                    "turn_idx": 3,
                    "messages": [
                        {"role": "system", "content": "warehouse manager instructions"},
                        {"role": "user", "content": "Turn 1 state Day 0"},
                        {"role": "assistant", "content": "<think>first</think><answer>pass</answer>"},
                        {"role": "user", "content": "Turn 2 state Day 14"},
                        {"role": "assistant", "content": "<think>second</think><answer>replenish MegaMart JCD543 1</answer>"},
                        {"role": "user", "content": "Turn 3 state Day 28"},
                    ],
                    "user_prompt": "Reward:\n2.0\n\nTurn 3:\nState:\n=== Step 2 (Day 28) ===\nCash: $180 | AR Outstanding: $0",
                    "parsed_response": "<think>third</think><answer>produce JCD612 1 ocean</answer>",
                    "financials": {
                        "cash": 250.0,
                        "cumulative_cost": 120.9,
                        "cumulative_inventory_weeks": 420.0,
                    },
                    "success": True,
                },
            ],
        }
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _build_rollout(env_id, initial_cash, turn_cashes):
    turns = []
    cumulative_cost = 0.0
    cumulative_inventory_weeks = 0.0
    for idx, cash in enumerate(turn_cashes, start=1):
        history_messages = [
            {"role": "system", "content": "warehouse manager instructions"},
            {"role": "user", "content": "Turn 1 state Day 0"},
        ]
        if idx >= 2:
            history_messages.extend(
                [
                    {"role": "assistant", "content": "<think>turn 1</think><answer>pass</answer>"},
                    {"role": "user", "content": "Turn 2 state Day 14"},
                ]
            )
        if idx >= 3:
            history_messages.extend(
                [
                    {"role": "assistant", "content": "<think>turn 2</think><answer>pass</answer>"},
                    {"role": "user", "content": "Turn 3 state Day 28"},
                ]
            )

        cumulative_cost += 20.0 + idx
        cumulative_inventory_weeks += 100.0 + (10 * idx)
        prior_cash = initial_cash if idx == 1 else turn_cashes[idx - 2]
        turns.append(
            {
                "turn_idx": idx,
                "messages": history_messages,
                "user_prompt": (
                    f"Turn {idx}:\nState:\n=== Step {idx - 1} (Day {(idx - 1) * 14}) ===\n"
                    f"Cash: ${prior_cash} | AR Outstanding: $0"
                ),
                "parsed_response": f"<think>turn {idx}</think><answer>pass</answer>",
                "financials": {
                    "cash": float(cash),
                    "cumulative_cost": float(cumulative_cost),
                    "cumulative_inventory_weeks": float(cumulative_inventory_weeks),
                },
                "success": idx == len(turn_cashes),
            }
        )

    return {
        "env_id": env_id,
        "absolute_env_id": env_id,
        "success": True,
        "initial_state": f"=== Step 0 (Day 0) ===\nCash: ${initial_cash} | AR Outstanding: $0",
        "final_state": (
            f"=== Step {len(turn_cashes)} (Day {14 * len(turn_cashes)}) ===\n"
            f"Cash: ${turn_cashes[-1]} | AR Outstanding: $0"
        ),
        "turns": turns,
    }


def test_money_estimation_env_flattens_and_scores(tmp_path):
    input_path = tmp_path / "warehouse_rollout.json"
    export_path = tmp_path / "warehouse_pairs.json"
    _write_money_fixture(input_path)

    env = MoneyEstimationEnv(
        MoneyEstimationEnvConfig(
            input_path=str(input_path),
        )
    )

    assert len(env.samples) == 2

    first_sample = env.samples[0]
    second_sample = env.samples[1]

    assert first_sample.source_system == "warehouse manager instructions"
    assert first_sample.completed_turns == 1
    assert first_sample.completed_weeks == 2
    assert first_sample.current_cash_usd == 120
    assert first_sample.target_cash_usd == 250
    assert first_sample.budget_time_weeks == 6
    assert first_sample.budget_warehouse_item_weeks == 420
    assert first_sample.budget_cost_usd == 121
    assert first_sample.actual_can_finish is True
    assert first_sample.actual_remaining_time_weeks == 4
    assert first_sample.actual_remaining_warehouse_item_weeks == 220
    assert first_sample.actual_remaining_cost_usd == 71
    assert first_sample.consumption_history == [
        {
            "start_day": 0,
            "end_day": 14,
            "time_weeks": 2,
            "warehouse_item_weeks": 200,
            "cost_usd": 50,
        }
    ]

    assert second_sample.completed_turns == 2
    assert second_sample.completed_weeks == 4
    assert second_sample.actual_remaining_time_weeks == 2
    assert second_sample.actual_remaining_warehouse_item_weeks == 120
    assert second_sample.actual_remaining_cost_usd == 41

    env.export_temp_pairs(str(export_path))
    with open(export_path, "r", encoding="utf-8") as handle:
        exported = json.load(handle)
    assert len(exported) == 2
    assert exported[0]["output"] == "<think>second</think><answer>replenish MegaMart JCD543 1</answer>"
    assert exported[0]["actual_remaining_time_weeks"] == 4
    assert exported[0]["actual_remaining_warehouse_item_weeks"] == 220
    assert exported[0]["actual_remaining_cost_usd"] == 71

    prompt = env.reset(index=0)
    assert '"role": "system"' in prompt
    assert "warehouse manager instructions" not in prompt
    assert "time_weeks <= 6" in prompt
    assert "warehouse_item_weeks <= 420" in prompt
    assert "cumulative_cost_usd <= 121" in prompt
    assert "Day 0: time +2 weeks, warehouse +200 item-weeks, cost +50 USD" in prompt

    _, reward, done, info = env.step(
        "<think>Estimate from the trend.</think>"
        "<answer>time_weeks:[4, 4], warehouse_item_weeks:[220, 240], cumulative_cost_usd:[70, 80]</answer>"
    )
    assert done is True
    assert reward == 1.0
    assert info["metrics"]["can_finish_correct"] is True
    assert info["metrics"]["time_interval_contains_actual"] is True
    assert info["metrics"]["warehouse_interval_contains_actual"] is True
    assert info["metrics"]["cost_interval_contains_actual"] is True


def test_money_estimation_env_handles_impossible_budget(tmp_path):
    input_path = tmp_path / "warehouse_rollout_impossible.json"
    _write_money_fixture(input_path)

    env = MoneyEstimationEnv(
        MoneyEstimationEnvConfig(
            input_path=str(input_path),
            time_budget_ratio=0.5,
        )
    )

    assert len(env.samples) == 2
    assert env.samples[0].actual_can_finish is False

    env.reset(index=0)
    _, reward, done, info = env.step("<think>Budget is too tight.</think><answer>impossible</answer>")
    assert done is True
    assert reward == 1.0
    assert info["metrics"]["can_finish_correct"] is True
    assert info["metrics"]["time_interval_contains_actual"] is None


def test_money_estimation_env_half_reachable_target_mode(tmp_path):
    input_path = tmp_path / "warehouse_rollout_half_reachable.json"
    payload = [
        _build_rollout(env_id=1, initial_cash=100, turn_cashes=[120, 170, 220]),
        _build_rollout(env_id=2, initial_cash=100, turn_cashes=[130, 180, 240]),
        _build_rollout(env_id=3, initial_cash=100, turn_cashes=[110, 160, 210]),
        _build_rollout(env_id=4, initial_cash=100, turn_cashes=[140, 190, 260]),
    ]
    with open(input_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    env = MoneyEstimationEnv(
        MoneyEstimationEnvConfig(
            input_path=str(input_path),
            target_cash_mode="half_reachable",
            target_cash_half_reachable_seed=7,
        )
    )

    assert len(env.samples) == 8

    final_cash_by_rollout = {
        0: 220,
        1: 240,
        2: 210,
        3: 260,
    }
    targets_by_rollout = {}
    can_finish_by_rollout = {}
    for sample in env.samples:
        targets_by_rollout.setdefault(sample.rollout_index, sample.target_cash_usd)
        can_finish_by_rollout.setdefault(sample.rollout_index, []).append(sample.actual_can_finish)
        assert targets_by_rollout[sample.rollout_index] == sample.target_cash_usd

    unreachable_rollouts = {
        rollout_index
        for rollout_index, target_cash_usd in targets_by_rollout.items()
        if target_cash_usd > final_cash_by_rollout[rollout_index]
    }
    reachable_rollouts = set(targets_by_rollout) - unreachable_rollouts

    assert len(unreachable_rollouts) == 2
    assert len(reachable_rollouts) == 2
    assert all(all(flag is False for flag in can_finish_by_rollout[idx]) for idx in unreachable_rollouts)
    assert all(all(flag is True for flag in can_finish_by_rollout[idx]) for idx in reachable_rollouts)
