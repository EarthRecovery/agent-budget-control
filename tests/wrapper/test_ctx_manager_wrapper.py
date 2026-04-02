import json

from omegaconf import OmegaConf

from ragen.wrapper.ctx_manager_wrapper import CtxManagerWrapper


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [0] * len(str(text).split())


def make_wrapper(tmp_path, *, start_group_index):
    config = OmegaConf.create(
        {
            "agent_proxy": {
                "enable_ctx_wrapper": True,
                "eval-estimation-single": True,
                "eval-estimation-multi": False,
            },
            "output": {
                "dir": str(tmp_path),
                "filename": "deepcoder_api_eval_estimation.pkl",
            },
            "es_manager": {
                "val": {
                    "start_group_index": start_group_index,
                    "group_size": 1,
                }
            },
        }
    )
    return CtxManagerWrapper(config, DummyTokenizer())


def make_multi_wrapper(tmp_path, *, start_group_index):
    config = OmegaConf.create(
        {
            "agent_proxy": {
                "enable_ctx_wrapper": True,
                "eval-estimation-single": False,
                "eval-estimation-multi": True,
                "enable_think": True,
            },
            "output": {
                "dir": str(tmp_path),
                "filename": "sokoban_api_eval_estimation.pkl",
            },
            "es_manager": {
                "val": {
                    "start_group_index": start_group_index,
                    "group_size": 1,
                }
            },
        }
    )
    return CtxManagerWrapper(config, DummyTokenizer())


def make_openai_reasoning_wrapper(tmp_path, *, start_group_index):
    config = OmegaConf.create(
        {
            "agent_proxy": {
                "enable_ctx_wrapper": True,
                "eval-estimation-single": True,
                "eval-estimation-multi": False,
                "enable_think": True,
            },
            "model_config": {
                "model_name": "OpenAI-5.2-Thinking",
            },
            "model_info": {
                "OpenAI-5.2-Thinking": {
                    "provider_name": "openai",
                    "model_name": "gpt-5.2",
                }
            },
            "output": {
                "dir": str(tmp_path),
                "filename": "gpqa_api_eval_estimation.pkl",
            },
            "es_manager": {
                "val": {
                    "start_group_index": start_group_index,
                    "group_size": 1,
                }
            },
        }
    )
    return CtxManagerWrapper(config, DummyTokenizer())


def test_estimation_log_appends_existing_entries(tmp_path):
    wrapper_first = make_wrapper(tmp_path, start_group_index=0)
    wrapper_first._estimation_records = {
        0: {
            "env_id": 0,
            "group_id": 0,
            "uid": None,
            "tag": "DeepCoder",
            "mode": "single",
            "turns": [{"turn_idx": 1}],
        }
    }
    wrapper_first._write_estimation_log()

    wrapper_second = make_wrapper(tmp_path, start_group_index=4)
    wrapper_second._estimation_records = {
        0: {
            "env_id": 0,
            "group_id": 0,
            "uid": None,
            "tag": "DeepCoder",
            "mode": "single",
            "turns": [{"turn_idx": 1}],
        }
    }
    wrapper_second._write_estimation_log()

    with open(wrapper_second.get_estimation_log_path(), "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert len(payload) == 2
    assert payload[0]["start_group_index"] == 0
    assert payload[0]["absolute_group_id"] == 0
    assert payload[1]["start_group_index"] == 4
    assert payload[1]["absolute_group_id"] == 4


def test_record_estimation_outputs_keeps_blank_response_for_generation_error(tmp_path):
    wrapper = make_wrapper(tmp_path, start_group_index=0)
    wrapper.turn_idx = 0
    env_record = wrapper._ensure_env_record(0, group_id=0, uid=None)
    turn_record = wrapper._ensure_turn_record(env_record, 1)
    wrapper._pending_turn_records[(0, 1)] = turn_record

    class DummyOutputs:
        non_tensor_batch = {
            "env_ids": [0],
            "response_texts": [""],
            "response_errors": [
                {
                    "error": "Invalid prompt: flagged by usage policy",
                    "error_type": "invalid_request_error",
                    "error_code": "invalid_prompt",
                    "status_code": 400,
                    "retryable": False,
                }
            ],
        }

    wrapper._record_estimation_outputs(DummyOutputs())

    assert turn_record["raw_generation"] == ""
    assert turn_record["raw_response"] == ""
    assert turn_record["generation_error_code"] == "invalid_prompt"
    assert turn_record["generation_success"] is False


def test_single_eval_estimation_records_budget_thinking_before_token_estimation(tmp_path):
    wrapper = make_wrapper(tmp_path, start_group_index=0)
    wrapper.turn_idx = 0
    env_record = wrapper._ensure_env_record(0, group_id=0, uid=None)
    turn_record = wrapper._ensure_turn_record(env_record, 1)
    wrapper._pending_turn_records[(0, 1)] = turn_record

    class DummyOutputs:
        non_tensor_batch = {
            "env_ids": [0],
            "response_texts": [
                (
                    "Estimate the token budget.</budget-thinking>"
                    "<token_estimation>64</token_estimation>"
                    "<think>Plan</think><answer>Right</answer>"
                )
            ],
        }

    wrapper._record_estimation_outputs(DummyOutputs())

    assert turn_record["raw_response"].startswith("<budget-thinking>")
    assert turn_record["estimate_token"] == 64


def test_single_eval_estimation_note_does_not_repeat_budget_thinking(tmp_path):
    wrapper = make_wrapper(tmp_path, start_group_index=0)
    messages_list = [[{"role": "user", "content": "Question"}]]

    wrapper._inject_eval_estimation_single_prompt(messages_list)

    prompt = messages_list[0][0]["content"]
    assert "<budget-thinking>" not in prompt
    assert "<token_estimation>" in prompt


def test_multi_eval_estimation_records_budget_thinking_before_estimates(tmp_path):
    wrapper = make_multi_wrapper(tmp_path, start_group_index=0)
    wrapper.turn_idx = 0
    env_record = wrapper._ensure_env_record(0, group_id=0, uid=None)
    turn_record = wrapper._ensure_turn_record(env_record, 1)
    wrapper._pending_turn_records[(0, 1)] = turn_record

    class DummyOutputs:
        non_tensor_batch = {
            "env_ids": [0],
            "response_texts": [
                (
                    "Estimate the remaining budget.</budget-thinking>"
                    "<turn_estimation>3</turn_estimation>"
                    "<token_estimation>64</token_estimation>"
                    "<think>Plan</think><answer>Right</answer>"
                )
            ],
        }

    wrapper._record_estimation_outputs(DummyOutputs())

    assert turn_record["raw_response"].startswith("<budget-thinking>")
    assert turn_record["estimate_remaining_turn"] == 3
    assert turn_record["estimate_token"] == 64
    assert "estimate_turn" not in turn_record


def test_openai_reasoning_eval_estimation_still_prepends_budget_thinking(tmp_path):
    wrapper = make_openai_reasoning_wrapper(tmp_path, start_group_index=0)
    wrapper.turn_idx = 0
    env_record = wrapper._ensure_env_record(0, group_id=0, uid=None)
    turn_record = wrapper._ensure_turn_record(env_record, 1)
    wrapper._pending_turn_records[(0, 1)] = turn_record

    class DummyOutputs:
        non_tensor_batch = {
            "env_ids": [0],
            "response_texts": [
                "<token_estimation>64</token_estimation><think>Plan</think><answer>B</answer>"
            ],
        }

    wrapper._record_estimation_outputs(DummyOutputs())

    assert turn_record["raw_response"].startswith("<budget-thinking>")
    assert turn_record["estimate_token"] == 64


def test_record_estimation_outputs_tracks_api_token_usage(tmp_path):
    wrapper = make_wrapper(tmp_path, start_group_index=0)
    wrapper.turn_idx = 0
    env_record = wrapper._ensure_env_record(0, group_id=0, uid=None)
    turn_record = wrapper._ensure_turn_record(env_record, 1)
    wrapper._pending_turn_records[(0, 1)] = turn_record

    class DummyOutputs:
        non_tensor_batch = {
            "env_ids": [0],
            "response_texts": [
                (
                    "Estimate the token budget.</budget-thinking>"
                    "<token_estimation>64</token_estimation>"
                    "<think>Plan</think><answer>Right</answer>"
                )
            ],
            "api_interactions": [
                [
                    {
                        "attempt": 1,
                        "success": True,
                        "provider": "openai",
                        "model": "gpt-5.2",
                        "request_id": "req_123",
                        "input_tokens": 120,
                        "output_tokens": 48,
                        "total_tokens": 168,
                        "usage": {
                            "input_tokens": 120,
                            "output_tokens": 48,
                            "total_tokens": 168,
                        },
                    }
                ]
            ],
        }

    wrapper._record_estimation_outputs(DummyOutputs())

    assert turn_record["api_interaction_count"] == 1
    assert turn_record["api_input_tokens"] == 120
    assert turn_record["api_output_tokens"] == 48
    assert turn_record["api_total_tokens"] == 168
    assert turn_record["api_interactions"][0]["request_id"] == "req_123"


def test_finalize_rollout_keeps_only_remaining_turn_fields(tmp_path):
    wrapper = make_multi_wrapper(tmp_path, start_group_index=0)
    wrapper.begin_rollout()
    wrapper.finalize_rollout(
        [
            {
                "env_id": 0,
                "group_id": 0,
                "uid": None,
                "tag": "Robotouille",
                "history": [
                    {
                        "state": "S1",
                        "llm_raw_response": (
                            "<budget-thinking>Budget</budget-thinking>"
                            "<turn_estimation>5</turn_estimation>"
                            "<token_estimation>64</token_estimation>"
                            "<think>Plan</think><answer>Act</answer>"
                        ),
                        "llm_response": "<think>Plan</think><answer>Act</answer>",
                        "actions": ["Act"],
                        "reward": 1.0,
                        "token_count": 64,
                        "info": {"success": False},
                    },
                    {
                        "state": "S2",
                        "llm_raw_response": (
                            "<budget-thinking>Budget</budget-thinking>"
                            "<turn_estimation>4</turn_estimation>"
                            "<token_estimation>64</token_estimation>"
                            "<think>Plan</think><answer>Act</answer>"
                        ),
                        "llm_response": "<think>Plan</think><answer>Act</answer>",
                        "actions": ["Act"],
                        "reward": 1.0,
                        "token_count": 64,
                        "info": {"success": True},
                    },
                ],
            }
        ]
    )

    turns = wrapper._estimation_records[0]["turns"]
    assert turns[0]["estimate_remaining_turn"] == 5
    assert turns[0]["actual_remaining_turn"] == 2
    assert "estimate_turn" not in turns[0]
    assert "actual_turn" not in turns[0]
