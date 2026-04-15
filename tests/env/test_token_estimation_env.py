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

    assert len(env.samples) == 3

    first_sample = env.samples[0]
    second_sample = env.samples[1]
    third_sample = env.samples[2]
    assert first_sample.source_system == "original system prompt"
    assert first_sample.input_messages == [{"role": "user", "content": "user turn 1"}]
    assert first_sample.actual_can_finish is False
    assert first_sample.actual_remaining_total_tokens == 140
    assert second_sample.actual_can_finish is False
    assert second_sample.actual_tokens_used_so_far == 80
    assert second_sample.actual_remaining_total_tokens == 60
    assert third_sample.completed_turn_token_usage == [80, 30]
    assert third_sample.actual_tokens_used_so_far == 110
    assert third_sample.actual_remaining_total_tokens == 30

    env.export_temp_pairs(str(export_path))
    with open(export_path, "r", encoding="utf-8") as handle:
        exported = json.load(handle)
    assert len(exported) == 3
    assert exported[0]["output"] == "<think>first</think><answer>search[q]</answer>"
    assert exported[0]["actual_tokens_used_so_far"] == 0
    assert exported[0]["actual_remaining_total_tokens"] == 140

    prompt = env.reset(index=0)
    assert "user turn 1" in prompt
    assert "You have completed 0 turns." in prompt
    system_message, user_message = env.build_api_messages()
    assert system_message["role"] == "system"
    assert "keeping the finishing turn within 120 total tokens (input + output)" in system_message["content"]
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

    prompt_third = env.reset(index=2)
    assert "Turn 1: 80 tokens" in prompt_third
    assert "Turn 2: 30 tokens" in prompt_third
