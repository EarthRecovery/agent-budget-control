import json

from ragen.env.token_estimation import (
    TokenEstimationEnv,
    TokenEstimationEnvConfig,
)


def _write_fixture(path):
    payload = [
        {
            "env_id": 0,
            "absolute_env_id": 0,
            "turns": [
                {
                    "turn_idx": 1,
                    "messages": [
                        {"role": "system", "content": "original system prompt"},
                        {"role": "user", "content": "user turn 1"},
                    ],
                    "parsed_response": "<think>first</think><answer>search[q]</answer>",
                    "api_input_tokens": 30,
                    "api_output_tokens": 50,
                    "api_total_tokens": 80,
                    "actual_token": 50,
                    "actual_remaining_turn": 2,
                    "success": False,
                },
                {
                    "turn_idx": 2,
                    "messages": [
                        {"role": "system", "content": "original system prompt"},
                        {"role": "user", "content": "user turn 1"},
                        {"role": "assistant", "content": "<think>first</think><answer>search[q]</answer>"},
                        {"role": "user", "content": "user turn 2"},
                    ],
                    "parsed_response": "<think>second</think><answer>finish[a]</answer>",
                    "api_input_tokens": 40,
                    "api_output_tokens": 70,
                    "api_total_tokens": 110,
                    "actual_token": 70,
                    "actual_remaining_turn": 1,
                    "success": True,
                },
                {
                    "turn_idx": 3,
                    "messages": [
                        {"role": "system", "content": "original system prompt"},
                        {"role": "user", "content": "user turn 1"},
                        {"role": "assistant", "content": "<think>first</think><answer>search[q]</answer>"},
                        {"role": "user", "content": "user turn 2"},
                        {"role": "assistant", "content": "<think>second</think><answer>finish[a]</answer>"},
                        {"role": "user", "content": "user turn 3"},
                    ],
                    "parsed_response": "<think>third</think><answer>done</answer>",
                    "api_input_tokens": 55,
                    "api_output_tokens": 85,
                    "api_total_tokens": 140,
                    "actual_token": 85,
                    "actual_remaining_turn": 0,
                    "success": True,
                },
            ],
        }
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_fixture_with_blank_middle_turn(path):
    payload = [
        {
            "env_id": 0,
            "absolute_env_id": 0,
            "turns": [
                {
                    "turn_idx": 1,
                    "messages": [
                        {"role": "system", "content": "original system prompt"},
                        {"role": "user", "content": "user turn 1"},
                    ],
                    "parsed_response": "<think>first</think><answer>search[q]</answer>",
                    "api_input_tokens": 30,
                    "api_output_tokens": 50,
                    "api_total_tokens": 80,
                    "success": False,
                },
                {
                    "turn_idx": 2,
                    "messages": [
                        {"role": "system", "content": "original system prompt"},
                        {"role": "user", "content": "user turn 1"},
                        {"role": "assistant", "content": "<think>first</think><answer>search[q]</answer>"},
                        {"role": "user", "content": "user turn 2"},
                    ],
                    "parsed_response": "",
                    "raw_response": "",
                    "api_input_tokens": 40,
                    "api_output_tokens": 0,
                    "api_total_tokens": 40,
                    "success": False,
                },
                {
                    "turn_idx": 3,
                    "messages": [
                        {"role": "system", "content": "original system prompt"},
                        {"role": "user", "content": "user turn 1"},
                        {"role": "assistant", "content": "<think>first</think><answer>search[q]</answer>"},
                        {"role": "user", "content": "user turn 2"},
                        {"role": "assistant", "content": ""},
                        {"role": "user", "content": "user turn 3"},
                    ],
                    "parsed_response": "<think>third</think><answer>done</answer>",
                    "api_input_tokens": 55,
                    "api_output_tokens": 85,
                    "api_total_tokens": 140,
                    "success": True,
                },
            ],
        }
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_cumulative_usage_fixture(path):
    payload = [
        {
            "env_id": 0,
            "absolute_env_id": 0,
            "turns": [
                {
                    "turn_idx": 1,
                    "messages": [
                        {"role": "system", "content": "original system prompt"},
                        {"role": "user", "content": "user turn 1"},
                    ],
                    "parsed_response": "<think>first</think><answer>Left</answer>",
                    "api_input_tokens": 426,
                    "api_output_tokens": 63,
                    "api_total_tokens": 489,
                    "success": False,
                },
                {
                    "turn_idx": 2,
                    "messages": [
                        {"role": "system", "content": "original system prompt"},
                        {"role": "user", "content": "user turn 1"},
                        {"role": "assistant", "content": "<think>first</think><answer>Left</answer>"},
                        {"role": "user", "content": "user turn 2"},
                    ],
                    "parsed_response": "<think>second</think><answer>Up</answer>",
                    "api_input_tokens": 647,
                    "api_output_tokens": 46,
                    "api_total_tokens": 693,
                    "success": False,
                },
                {
                    "turn_idx": 3,
                    "messages": [
                        {"role": "system", "content": "original system prompt"},
                        {"role": "user", "content": "user turn 1"},
                        {"role": "assistant", "content": "<think>first</think><answer>Left</answer>"},
                        {"role": "user", "content": "user turn 2"},
                        {"role": "assistant", "content": "<think>second</think><answer>Up</answer>"},
                        {"role": "user", "content": "user turn 3"},
                    ],
                    "parsed_response": "<think>third</think><answer>Done</answer>",
                    "api_input_tokens": 853,
                    "api_output_tokens": 50,
                    "api_total_tokens": 903,
                    "success": True,
                },
            ],
        }
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def test_token_estimation_env_flattens_and_scores(tmp_path):
    input_path = tmp_path / "dialogues.json"
    export_path = tmp_path / "pairs.json"
    _write_fixture(input_path)

    env = TokenEstimationEnv(
        TokenEstimationEnvConfig(
            input_path=str(input_path),
            max_context_window_tokens=120,
        )
    )

    assert len(env.samples) == 2

    first_sample = env.samples[0]
    second_sample = env.samples[1]
    assert first_sample.source_system == "original system prompt"
    assert first_sample.input_messages == [
        {"role": "user", "content": "user turn 1"},
        {"role": "assistant", "content": "<think>first</think><answer>search[q]</answer>"},
    ]
    assert first_sample.actual_can_finish is False
    assert first_sample.completed_turns == 1
    assert first_sample.total_turns == 3
    assert round(first_sample.relative_progress, 2) == 0.33
    assert first_sample.actual_tokens_used_so_far == 80
    assert first_sample.actual_remaining_total_tokens == 250
    assert first_sample.target_output == "<think>second</think><answer>finish[a]</answer>"
    assert second_sample.actual_can_finish is False
    assert second_sample.completed_turns == 2
    assert second_sample.total_turns == 3
    assert round(second_sample.relative_progress, 2) == 0.67
    assert second_sample.actual_tokens_used_so_far == 190
    assert second_sample.actual_remaining_total_tokens == 140
    assert second_sample.completed_turn_token_usage == [80, 110]
    assert second_sample.target_output == "<think>third</think><answer>done</answer>"

    env.export_temp_pairs(str(export_path))
    with open(export_path, "r", encoding="utf-8") as handle:
        exported = json.load(handle)
    assert len(exported) == 2
    assert exported[0]["output"] == "<think>second</think><answer>finish[a]</answer>"
    assert exported[0]["actual_tokens_used_so_far"] == 80
    assert exported[0]["actual_remaining_total_tokens"] == 250
    assert exported[0]["total_turns"] == 3
    assert round(exported[0]["relative_progress"], 2) == 0.33
    assert exported[0]["completed_turn_token_usage_details"] == [
        {"input_tokens": 30, "output_tokens": 50, "total_tokens": 80}
    ]

    prompt = env.reset(index=0)
    assert "user turn 1" in prompt
    assert "search[q]" in prompt
    assert "You have completed 1 turns." in prompt
    assert "Turn 1: input 30 tokens, output 50 tokens, total 80 tokens" in prompt
    system_message, user_message = env.build_api_messages()
    assert system_message["role"] == "system"
    assert "finish successfully within 120 total tokens (input + output)" in system_message["content"]
    assert user_message["role"] == "user"

    _, reward, done, info = env.step(
        "<think>评估</think>"
        "<answer>impossible</answer>"
    )
    assert done is True
    assert reward == 1.0
    assert info["metrics"]["can_finish_correct"] is True
    assert info["prediction"]["is_impossible"] is True
    assert info["metrics"]["remaining_token_interval_contains_actual"] is None

    prompt_second = env.reset(index=1)
    assert "Turn 1: input 30 tokens, output 50 tokens, total 80 tokens" in prompt_second
    assert "Turn 2: input 40 tokens, output 70 tokens, total 110 tokens" in prompt_second
    assert "Turn 3: input 55 tokens, output 85 tokens, total 140 tokens" not in prompt_second


def test_token_estimation_env_skips_blank_assistant_turns_in_slices(tmp_path):
    input_path = tmp_path / "dialogues_blank.json"
    _write_fixture_with_blank_middle_turn(input_path)

    env = TokenEstimationEnv(
        TokenEstimationEnvConfig(
            input_path=str(input_path),
            max_context_window_tokens=500,
        )
    )

    assert len(env.samples) == 1

    first_sample = env.samples[0]

    assert first_sample.completed_turns == 1
    assert first_sample.total_turns == 2
    assert first_sample.relative_progress == 0.5
    assert first_sample.actual_remaining_total_tokens == 140
    assert first_sample.actual_tokens_used_so_far == 80
    assert first_sample.completed_turn_token_usage == [80]
    assert first_sample.target_output == "<think>third</think><answer>done</answer>"
    assert first_sample.input_messages == [
        {"role": "user", "content": "user turn 1"},
        {"role": "assistant", "content": "<think>first</think><answer>search[q]</answer>"},
    ]


def test_token_estimation_env_uses_cumulative_total_deltas_for_real_rollouts(tmp_path):
    input_path = tmp_path / "dialogues_cumulative.json"
    _write_cumulative_usage_fixture(input_path)

    env = TokenEstimationEnv(
        TokenEstimationEnvConfig(
            input_path=str(input_path),
            max_context_window_tokens=1000,
        )
    )

    assert len(env.samples) == 2

    first_sample = env.samples[0]
    second_sample = env.samples[1]

    assert first_sample.completed_turn_token_usage_details == [
        {"input_tokens": 426, "output_tokens": 63, "total_tokens": 489}
    ]
    assert second_sample.completed_turn_token_usage_details == [
        {"input_tokens": 426, "output_tokens": 63, "total_tokens": 489},
        {"input_tokens": 158, "output_tokens": 46, "total_tokens": 204},
    ]
    assert round(first_sample.relative_progress, 2) == 0.33
    assert round(second_sample.relative_progress, 2) == 0.67
    assert first_sample.actual_tokens_used_so_far == 489
    assert second_sample.actual_tokens_used_so_far == 693
    assert first_sample.actual_remaining_total_tokens == 414
    assert second_sample.actual_remaining_total_tokens == 210
    assert first_sample.target_output == "<think>second</think><answer>Up</answer>"
    assert second_sample.target_output == "<think>third</think><answer>Done</answer>"

    prompt = env.reset(index=1)
    assert "Turn 1: input 426 tokens, output 63 tokens, total 489 tokens" in prompt
    assert "Turn 2: input 158 tokens, output 46 tokens, total 204 tokens" in prompt
