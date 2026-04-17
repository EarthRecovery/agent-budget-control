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


def _write_api_usage_fixture(path):
    payload = [
        {
            "env_id": 0,
            "absolute_env_id": 0,
            "turns": [
                {
                    "turn_idx": 1,
                    "messages": [
                        {"role": "system", "content": "search system"},
                        {"role": "user", "content": "question turn 1"},
                    ],
                    "parsed_response": "<think>first</think><answer>search[q1]</answer>",
                    "api_interactions": [
                        {"input_tokens": 195, "output_tokens": 56, "total_tokens": 251}
                    ],
                    "api_input_tokens": 195,
                    "api_output_tokens": 56,
                    "api_total_tokens": 251,
                    "success": False,
                },
                {
                    "turn_idx": 2,
                    "messages": [
                        {"role": "system", "content": "search system"},
                        {"role": "user", "content": "question turn 1"},
                        {"role": "assistant", "content": "<think>first</think><answer>search[q1]</answer>"},
                        {"role": "user", "content": "search result turn 2"},
                    ],
                    "parsed_response": "<think>second</think><answer>search[q2]</answer>",
                    "api_interactions": [
                        {"input_tokens": 372, "output_tokens": 42, "total_tokens": 414}
                    ],
                    "api_input_tokens": 372,
                    "api_output_tokens": 42,
                    "api_total_tokens": 414,
                    "success": False,
                },
                {
                    "turn_idx": 3,
                    "messages": [
                        {"role": "system", "content": "search system"},
                        {"role": "user", "content": "question turn 1"},
                        {"role": "assistant", "content": "<think>first</think><answer>search[q1]</answer>"},
                        {"role": "user", "content": "search result turn 2"},
                        {"role": "assistant", "content": "<think>second</think><answer>search[q2]</answer>"},
                        {"role": "user", "content": "search result turn 3"},
                    ],
                    "parsed_response": "<think>third</think><answer>search[q3]</answer>",
                    "api_interactions": [
                        {"input_tokens": 529, "output_tokens": 45, "total_tokens": 574}
                    ],
                    "api_input_tokens": 529,
                    "api_output_tokens": 45,
                    "api_total_tokens": 574,
                    "success": False,
                },
                {
                    "turn_idx": 4,
                    "messages": [
                        {"role": "system", "content": "search system"},
                        {"role": "user", "content": "question turn 1"},
                        {"role": "assistant", "content": "<think>first</think><answer>search[q1]</answer>"},
                        {"role": "user", "content": "search result turn 2"},
                        {"role": "assistant", "content": "<think>second</think><answer>search[q2]</answer>"},
                        {"role": "user", "content": "search result turn 3"},
                        {"role": "assistant", "content": "<think>third</think><answer>search[q3]</answer>"},
                        {"role": "user", "content": "large search result turn 4"},
                    ],
                    "parsed_response": "<think>fourth</think><answer>finish[a]</answer>",
                    "api_interactions": [
                        {"input_tokens": 1512, "output_tokens": 41, "total_tokens": 1553}
                    ],
                    "api_input_tokens": 1512,
                    "api_output_tokens": 41,
                    "api_total_tokens": 1553,
                    "success": True,
                },
            ],
        }
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_truncated_turn_fixture(path):
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
                    "parsed_response": "<think>first</think><answer>a1</answer>",
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
                        {"role": "assistant", "content": "<think>first</think><answer>a1</answer>"},
                        {"role": "user", "content": "user turn 2"},
                    ],
                    "parsed_response": "<think>second</think><answer>a2</answer>",
                    "api_input_tokens": 35,
                    "api_output_tokens": 55,
                    "api_total_tokens": 90,
                    "success": False,
                },
                {
                    "turn_idx": 3,
                    "messages": [
                        {"role": "system", "content": "original system prompt"},
                        {"role": "user", "content": "user turn 1"},
                        {"role": "assistant", "content": "<think>first</think><answer>a1</answer>"},
                        {"role": "user", "content": "user turn 2"},
                        {"role": "assistant", "content": "<think>second</think><answer>a2</answer>"},
                        {"role": "user", "content": "user turn 3"},
                    ],
                    "parsed_response": "<think>third</think><answer>a3</answer>",
                    "api_input_tokens": 40,
                    "api_output_tokens": 60,
                    "api_total_tokens": 100,
                    "success": False,
                },
                {
                    "turn_idx": 4,
                    "messages": [
                        {"role": "system", "content": "original system prompt"},
                        {"role": "user", "content": "user turn 1"},
                        {"role": "assistant", "content": "<think>first</think><answer>a1</answer>"},
                        {"role": "user", "content": "user turn 2"},
                        {"role": "assistant", "content": "<think>second</think><answer>a2</answer>"},
                        {"role": "user", "content": "user turn 3"},
                        {"role": "assistant", "content": "<think>third</think><answer>a3</answer>"},
                        {"role": "user", "content": "user turn 4"},
                    ],
                    "parsed_response": "<think>fourth</think><answer>a4</answer>",
                    "api_input_tokens": 45,
                    "api_output_tokens": 65,
                    "api_total_tokens": 110,
                    "success": False,
                },
                {
                    "turn_idx": 5,
                    "messages": [
                        {"role": "system", "content": "original system prompt"},
                        {"role": "user", "content": "truncated user turn"},
                    ],
                    "parsed_response": "<think>too long</think><answer>a5</answer>",
                    "api_input_tokens": 5000,
                    "api_output_tokens": 10,
                    "api_total_tokens": 5010,
                    "context_token_truncated": True,
                    "success": False,
                },
            ],
        }
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_implicit_truncated_turn_fixture(path):
    payload = [
        {
            "env_id": 0,
            "absolute_env_id": 0,
            "turns": [
                {
                    "turn_idx": 1,
                    "messages": [
                        {"role": "system", "content": "search system"},
                        {"role": "user", "content": "question turn 1"},
                    ],
                    "parsed_response": "<think>first</think><answer>search[q1]</answer>",
                    "api_interactions": [
                        {"input_tokens": 195, "output_tokens": 56, "total_tokens": 251}
                    ],
                    "api_input_tokens": 195,
                    "api_output_tokens": 56,
                    "api_total_tokens": 251,
                    "actions": ["search[q1]"],
                    "success": False,
                },
                {
                    "turn_idx": 2,
                    "messages": [
                        {"role": "system", "content": "search system"},
                        {"role": "user", "content": "question turn 1"},
                        {"role": "assistant", "content": "<think>first</think><answer>search[q1]</answer>"},
                        {"role": "user", "content": "search result turn 2"},
                    ],
                    "parsed_response": "<think>second</think><answer>search[q2]</answer>",
                    "api_interactions": [
                        {"input_tokens": 372, "output_tokens": 42, "total_tokens": 414}
                    ],
                    "api_input_tokens": 372,
                    "api_output_tokens": 42,
                    "api_total_tokens": 414,
                    "actions": ["search[q2]"],
                    "success": False,
                },
                {
                    "turn_idx": 3,
                    "messages": [
                        {"role": "system", "content": "search system"},
                        {"role": "user", "content": "question turn 1"},
                        {"role": "assistant", "content": "<think>first</think><answer>search[q1]</answer>"},
                        {"role": "user", "content": "search result turn 2"},
                        {"role": "assistant", "content": "<think>second</think><answer>search[q2]</answer>"},
                        {"role": "user", "content": "search result turn 3"},
                    ],
                    "parsed_response": "<think>third</think><answer>search[q3]</answer>",
                    "api_interactions": [
                        {"input_tokens": 529, "output_tokens": 45, "total_tokens": 574}
                    ],
                    "api_input_tokens": 529,
                    "api_output_tokens": 45,
                    "api_total_tokens": 574,
                    "actions": ["search[q3]"],
                    "success": False,
                },
                {
                    "turn_idx": 4,
                    "messages": [
                        {"role": "system", "content": "search system"},
                        {"role": "user", "content": "question turn 1"},
                        {"role": "assistant", "content": "<think>first</think><answer>search[q1]</answer>"},
                        {"role": "user", "content": "search result turn 2"},
                        {"role": "assistant", "content": "<think>second</think><answer>search[q2]</answer>"},
                        {"role": "user", "content": "search result turn 3"},
                        {"role": "assistant", "content": "<think>third</think><answer>search[q3]</answer>"},
                        {"role": "user", "content": "oversized search result turn 4"},
                    ],
                    "parsed_response": "<think>fourth</think><answer>finish[a]</answer>",
                    "api_interactions": [
                        {"input_tokens": 4200, "output_tokens": 118, "total_tokens": 4318}
                    ],
                    "api_input_tokens": 4200,
                    "api_output_tokens": 118,
                    "api_total_tokens": 4318,
                    "actions": [],
                    "success": False,
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
    assert exported[0]["input_messages"] == [
        {"role": "system", "content": "original system prompt"},
        {"role": "user", "content": "user turn 1"},
        {"role": "assistant", "content": "<think>first</think><answer>search[q]</answer>"},
        {
            "role": "user",
            "content": (
                "Based on the provided rollout context, you are provided below information:\n"
                "1. You have completed 1 turns.\n"
                "2. Each turn, your token consumption is Turn 1: input 30 tokens, output 50 tokens, total 80 tokens.\n"
                "3. You need to finish the task within 120 tokens.\n\n"
                "Now, estimate:\n"
                "1. Whether you can finish the task successfully within 120 total tokens (input + output).\n"
                "2. If yes, how many additional tokens (input + output) are still needed to finish the task, starting from the next turn. Return an estimation interval: at least est_low tokens and at most est_high tokens.\n"
                "3. If no, answer \"impossible\".\n"
                "4. You should try your best to estimate whether the task can finish within budget (most important). If you think the task can finish within budget, your interval should be as tight as possible while still covering the true remaining token budget.\n\n"
                "Example:\n"
                "For a three-turn interaction, suppose only Turn 1 has been completed.\n"
                "The full interaction is:\n"
                "Turn 1: input X1 tokens, output Y1 tokens;\n"
                "Turn 2: input X2 tokens, output Y2 tokens;\n"
                "Turn 3: input X3 tokens, output Y3 tokens.\n"
                "You will receive:\n"
                "turn_token_usage_text: Turn 1: input X1 tokens, output Y1 tokens\n"
                "You should estimate:\n"
                "X2 + Y2 + X3 + Y3\n\n"
                "Output exactly one of the following:\n"
                "<think>[YOUR THINKING]</think><answer>[est_low, est_high]</answer>\n"
                "or\n"
                "<think>[YOUR THINKING]</think><answer>impossible</answer>"
            ),
        },
    ]
    assert exported[0]["rollout_history_messages"] == [
        {"role": "user", "content": "user turn 1"},
        {"role": "assistant", "content": "<think>first</think><answer>search[q]</answer>"},
    ]

    prompt = env.reset(index=0)
    assert '"role": "system"' in prompt
    assert "user turn 1" in prompt
    assert "search[q]" in prompt
    assert "You have completed 1 turns." in prompt
    assert "Turn 1: input 30 tokens, output 50 tokens, total 80 tokens" in prompt
    system_message, first_user_message, first_assistant_message, user_message = env.build_api_messages()
    assert system_message["role"] == "system"
    assert system_message["content"] == "original system prompt"
    assert first_user_message == {"role": "user", "content": "user turn 1"}
    assert first_assistant_message == {
        "role": "assistant",
        "content": "<think>first</think><answer>search[q]</answer>",
    }
    assert user_message["role"] == "user"
    assert "Based on the provided rollout context" in user_message["content"]
    assert "original system prompt" not in user_message["content"]
    assert "user turn 1" not in user_message["content"]

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
    assert '"content": "user turn 2"' in prompt_second


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


def test_token_estimation_env_uses_direct_api_usage_when_available(tmp_path):
    input_path = tmp_path / "dialogues_api_usage.json"
    _write_api_usage_fixture(input_path)

    env = TokenEstimationEnv(
        TokenEstimationEnvConfig(
            input_path=str(input_path),
            max_context_window_tokens=4000,
        )
    )

    assert len(env.samples) == 3

    third_sample = env.samples[2]
    assert third_sample.completed_turns == 3
    assert third_sample.total_turns == 4
    assert third_sample.actual_can_finish is True
    assert third_sample.completed_turn_token_usage == [251, 414, 574]
    assert third_sample.completed_turn_token_usage_details == [
        {"input_tokens": 195, "output_tokens": 56, "total_tokens": 251},
        {"input_tokens": 372, "output_tokens": 42, "total_tokens": 414},
        {"input_tokens": 529, "output_tokens": 45, "total_tokens": 574},
    ]
    assert third_sample.actual_tokens_used_so_far == 1239
    assert third_sample.actual_remaining_total_tokens == 1553
    assert third_sample.target_output == "<think>fourth</think><answer>finish[a]</answer>"

    prompt = env.reset(index=2)
    assert "Turn 1: input 195 tokens, output 56 tokens, total 251 tokens" in prompt
    assert "Turn 2: input 372 tokens, output 42 tokens, total 414 tokens" in prompt
    assert "Turn 3: input 529 tokens, output 45 tokens, total 574 tokens" in prompt


def test_token_estimation_env_discards_context_token_truncated_turns(tmp_path):
    input_path = tmp_path / "dialogues_truncated.json"
    _write_truncated_turn_fixture(input_path)

    env = TokenEstimationEnv(
        TokenEstimationEnvConfig(
            input_path=str(input_path),
            max_context_window_tokens=1000,
        )
    )

    assert len(env.samples) == 3

    first_sample = env.samples[0]
    last_sample = env.samples[-1]
    assert first_sample.total_turns == 4
    assert last_sample.total_turns == 4
    assert first_sample.actual_remaining_total_tokens == 300
    assert last_sample.actual_tokens_used_so_far == 270
    assert last_sample.actual_remaining_total_tokens == 110
    assert last_sample.target_output == "<think>fourth</think><answer>a4</answer>"
    assert last_sample.actual_can_finish is False


def test_token_estimation_env_discards_implicit_over_limit_terminal_turns(tmp_path):
    input_path = tmp_path / "dialogues_implicit_truncated.json"
    _write_implicit_truncated_turn_fixture(input_path)

    env = TokenEstimationEnv(
        TokenEstimationEnvConfig(
            input_path=str(input_path),
            max_context_window_tokens=3500,
        )
    )

    assert len(env.samples) == 2

    first_sample = env.samples[0]
    last_sample = env.samples[-1]
    assert first_sample.total_turns == 3
    assert last_sample.total_turns == 3
    assert first_sample.actual_tokens_used_so_far == 251
    assert first_sample.actual_remaining_total_tokens == 988
    assert last_sample.actual_tokens_used_so_far == 665
    assert last_sample.actual_remaining_total_tokens == 574
    assert last_sample.target_output == "<think>third</think><answer>search[q3]</answer>"
    assert last_sample.actual_can_finish is False
