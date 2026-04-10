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


def make_adaptation_turn_wrapper(
    tmp_path,
    *,
    start_group_index,
    adaptation_scope=None,
):
    if adaptation_scope is None:
        adaptation_scope = [2, 5, 3]
    config = OmegaConf.create(
        {
            "agent_proxy": {
                "enable_ctx_wrapper": True,
                "eval-estimation-single": False,
                "eval-estimation-multi": False,
                "eval_adaptation_turn": True,
                "eval_adaptation_turn_scope": adaptation_scope,
                "enable_think": True,
                "max_turn": 6,
            },
            "output": {
                "dir": str(tmp_path),
                "filename": "adaptation_turn_api_eval.pkl",
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


def make_toolcall_wrapper(
    tmp_path,
    *,
    start_group_index,
    enable_action_budget=True,
    max_action_points=10,
    env_type="robotouille",
):
    config = OmegaConf.create(
        {
            "agent_proxy": {
                "enable_ctx_wrapper": True,
                "eval-estimation-single": False,
                "eval-estimation-multi": False,
                "eval-estimation-toolcall": True,
                "enable_think": True,
            },
            "custom_envs": {
                "Robotouille": {
                    "env_type": env_type,
                    "env_config": {
                        "enable_action_budget": enable_action_budget,
                        "max_action_points": max_action_points,
                    },
                }
            },
            "output": {
                "dir": str(tmp_path),
                "filename": "robotouille_api_eval_estimation.pkl",
            },
            "es_manager": {
                "train": {
                    "env_configs": {
                        "tags": ["Robotouille"],
                    },
                    "group_size": 1,
                },
                "val": {
                    "start_group_index": start_group_index,
                    "group_size": 1,
                    "env_configs": {
                        "tags": ["Robotouille"],
                    },
                },
            },
        }
    )
    return CtxManagerWrapper(config, DummyTokenizer())


def make_mixed_toolcall_budget_wrapper(tmp_path, *, start_group_index):
    config = OmegaConf.create(
        {
            "agent_proxy": {
                "enable_ctx_wrapper": True,
                "mixed_toolcall_budget": {
                    "enabled": True,
                    "mixed_budget": True,
                    "mixed_budget_range": [3, 6],
                },
            },
            "output": {
                "dir": str(tmp_path),
                "filename": "robotouille_mixed_toolcall_budget.pkl",
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


def make_dialogue_wrapper(tmp_path, *, start_group_index):
    config = OmegaConf.create(
        {
            "agent_proxy": {
                "enable_ctx_wrapper": True,
                "eval-estimation-single": False,
                "eval-estimation-multi": False,
                "enable_think": True,
            },
            "output": {
                "dir": str(tmp_path),
                "filename": "sokoban_origin.pkl",
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


def make_compliance_wrapper(
    tmp_path,
    *,
    start_group_index,
    token_scope=None,
    base_group_size=1,
):
    if token_scope is None:
        token_scope = [100, 200, 300, 400, 500]
    group_size = max(1, int(base_group_size)) * (len(token_scope) if len(token_scope) > 0 else 1)
    config = OmegaConf.create(
        {
            "agent_proxy": {
                "enable_ctx_wrapper": True,
                "eval-estimation-single": False,
                "eval-estimation-multi": False,
                "eval_compliance_token": True,
                "eval_compliance_token_scope": token_scope,
            },
            "output": {
                "dir": str(tmp_path),
                "filename": "compliance_api_eval.pkl",
            },
            "es_manager": {
                "val": {
                    "start_group_index": start_group_index,
                    "group_size": group_size,
                }
            },
        }
    )
    return CtxManagerWrapper(config, DummyTokenizer())


def make_turn_compliance_wrapper(
    tmp_path,
    *,
    start_group_index,
    turn_scope=None,
    base_group_size=1,
    mutation_turn=None,
    budget_change=None,
):
    if turn_scope is None:
        turn_scope = [1, 2, 3, 4, 5]
    mutation_enabled = mutation_turn is not None or bool(budget_change)
    group_size_factor = len(turn_scope) if len(turn_scope) > 0 else 1
    if mutation_enabled:
        group_size_factor = 1
    group_size = max(1, int(base_group_size)) * group_size_factor
    config = OmegaConf.create(
        {
            "agent_proxy": {
                "enable_ctx_wrapper": True,
                "eval-estimation-single": False,
                "eval-estimation-multi": False,
                "eval_compliance_turn": True,
                "eval_compliance_turn_scope": turn_scope,
                "eval_compliance_turn_mutation_turn": mutation_turn,
                "eval_compliance_turn_budget_change": budget_change or [],
            },
            "output": {
                "dir": str(tmp_path),
                "filename": "turn_compliance_api_eval.pkl",
            },
            "es_manager": {
                "val": {
                    "start_group_index": start_group_index,
                    "group_size": group_size,
                }
            },
        }
    )
    return CtxManagerWrapper(config, DummyTokenizer())


def make_toolcall_compliance_wrapper(
    tmp_path,
    *,
    start_group_index,
    toolcall_scope=None,
    base_group_size=1,
    enable_action_budget=True,
    max_action_points=10,
):
    if toolcall_scope is None:
        toolcall_scope = [2, 4, 6]
    group_size = max(1, int(base_group_size)) * (len(toolcall_scope) if len(toolcall_scope) > 0 else 1)
    config = OmegaConf.create(
        {
            "agent_proxy": {
                "enable_ctx_wrapper": True,
                "eval-estimation-single": False,
                "eval-estimation-multi": False,
                "eval_compliance_toolcall": True,
                "eval_compliance_toolcall_scope": toolcall_scope,
                "enable_think": True,
            },
            "custom_envs": {
                "Robotouille": {
                    "env_type": "robotouille",
                    "env_config": {
                        "enable_action_budget": enable_action_budget,
                        "max_action_points": max_action_points,
                    },
                }
            },
            "output": {
                "dir": str(tmp_path),
                "filename": "toolcall_compliance_api_eval.pkl",
            },
            "es_manager": {
                "train": {
                    "env_configs": {
                        "tags": ["Robotouille"],
                    },
                    "group_size": group_size,
                },
                "val": {
                    "start_group_index": start_group_index,
                    "group_size": group_size,
                    "env_configs": {
                        "tags": ["Robotouille"],
                    },
                },
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


def test_output_filename_enables_dialogue_log_for_regular_rollout(tmp_path):
    wrapper = make_dialogue_wrapper(tmp_path, start_group_index=0)

    assert wrapper.get_estimation_log_path().endswith("_eval_estimation_dialogues.json")
    assert wrapper.get_eval_log_key() is None

    wrapper.begin_rollout()
    wrapper.turn_idx = 0
    wrapper.mode = "singleturn"
    wrapper._record_estimation_inputs(
        non_tensor_batch={
            "env_ids": [0],
            "group_ids": [0],
        },
        messages_list=[
            [
                {"role": "system", "content": "Solve the Sokoban puzzle."},
                {"role": "user", "content": "Turn 1 state"},
            ]
        ],
        texts=["system\nuser"],
        generation_suffix="<think>",
    )

    class DummyOutputs:
        non_tensor_batch = {
            "env_ids": [0],
            "response_texts": ["<think>plan</think><answer>Right</answer>"],
        }

    wrapper._record_estimation_outputs(DummyOutputs())
    wrapper.finalize_rollout(
        [
            {
                "env_id": 0,
                "group_id": 0,
                "uid": None,
                "tag": "CoordSokoban",
                "history": [
                    {
                        "state": "S1",
                        "llm_raw_response": "<think>plan</think><answer>Right</answer>",
                        "llm_response": "<answer>Right</answer>",
                        "actions": ["Right"],
                        "reward": 1.0,
                        "token_count": 8,
                        "info": {"success": True},
                    },
                ],
            }
        ]
    )

    with open(wrapper.get_estimation_log_path(), "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert len(payload) == 1
    assert payload[0]["mode"] == "dialogue"
    assert payload[0]["turns"][0]["messages"][1]["content"] == "Turn 1 state"
    assert payload[0]["turns"][0]["raw_response"] == "<think>plan</think><answer>Right</answer>"
    assert payload[0]["turns"][0]["actions"] == ["Right"]


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


def test_adaptation_turn_records_budget_thinking_before_estimates(tmp_path):
    wrapper = make_adaptation_turn_wrapper(tmp_path, start_group_index=0)
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


def test_toolcall_eval_estimation_records_action_point_estimates_and_clips_to_budget(tmp_path):
    wrapper = make_toolcall_wrapper(
        tmp_path,
        start_group_index=0,
        max_action_points=6,
    )
    wrapper.turn_idx = 0
    env_record = wrapper._ensure_env_record(0, group_id=0, uid=None)
    turn_record = wrapper._ensure_turn_record(env_record, 1)
    wrapper._pending_turn_records[(0, 1)] = turn_record

    class DummyOutputs:
        non_tensor_batch = {
            "env_ids": [0],
            "response_texts": [
                (
                    "Estimate the action-point budget.</budget-thinking>"
                    "<remaining_action_points_estimation>9</remaining_action_points_estimation>"
                    "<action_points_estimation>7</action_points_estimation>"
                    "<think>Plan</think><answer>move</answer>"
                )
            ],
        }

    wrapper._record_estimation_outputs(DummyOutputs())

    assert turn_record["raw_response"].startswith("<budget-thinking>")
    assert turn_record["estimate_remaining_action_points"] == 6
    assert turn_record["estimate_action_points"] == 6
    assert "estimate_token" not in turn_record


def test_toolcall_eval_estimation_prompt_mentions_action_points_and_cap(tmp_path):
    wrapper = make_toolcall_wrapper(
        tmp_path,
        start_group_index=0,
        max_action_points=6,
    )
    messages_list = [[{"role": "user", "content": "Question"}]]

    wrapper._inject_eval_estimation_toolcall_prompt(messages_list)

    prompt = messages_list[0][0]["content"]
    assert "<remaining_action_points_estimation>" in prompt
    assert "<action_points_estimation>" in prompt
    assert "not the same as the current budget remaining shown in the state" in prompt
    assert "between 0 and 6 inclusive" in prompt


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


def test_finalize_rollout_records_toolcall_action_point_fields(tmp_path):
    wrapper = make_toolcall_wrapper(
        tmp_path,
        start_group_index=0,
        max_action_points=6,
    )
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
                            "<remaining_action_points_estimation>8</remaining_action_points_estimation>"
                            "<action_points_estimation>2</action_points_estimation>"
                            "<think>Plan</think><answer>move</answer>"
                        ),
                        "llm_response": "<think>Plan</think><answer>move</answer>",
                        "actions": ["move"],
                        "reward": 0.0,
                        "token_count": 32,
                        "action_points_used": 2,
                        "info": {"success": False},
                    },
                    {
                        "state": "S2",
                        "llm_raw_response": (
                            "<budget-thinking>Budget</budget-thinking>"
                            "<remaining_action_points_estimation>4</remaining_action_points_estimation>"
                            "<action_points_estimation>3</action_points_estimation>"
                            "<think>Plan</think><answer>cook</answer>"
                        ),
                        "llm_response": "<think>Plan</think><answer>cook</answer>",
                        "actions": ["cook"],
                        "reward": 1.0,
                        "token_count": 40,
                        "action_points_used": 3,
                        "info": {"success": True},
                    },
                ],
            }
        ]
    )

    record = wrapper._estimation_records[0]
    turns = record["turns"]
    assert record["max_action_points"] == 6
    assert record["success"] is True
    assert record["final_reward"] == 1.0
    assert record["final_reward_source"] == "rollout"
    assert record["final_rollout_reward"] == 1.0
    assert turns[0]["estimate_remaining_action_points"] == 6
    assert turns[0]["estimate_remaining_action_points_to_finish"] == 6
    assert turns[0]["actual_remaining_action_points"] == 5
    assert turns[0]["actual_remaining_action_points_to_finish"] == 5
    assert turns[0]["estimate_action_points"] == 2
    assert turns[0]["actual_action_points"] == 2
    assert turns[0]["action_points_used_before_turn"] == 0
    assert turns[0]["cumulative_action_points_used"] == 2
    assert turns[0]["budget_remaining_before_turn"] == 6
    assert turns[0]["budget_remaining_after_turn"] == 4
    assert turns[0]["toolcalls_used"] == 1
    assert turns[1]["estimate_remaining_action_points"] == 4
    assert turns[1]["estimate_remaining_action_points_to_finish"] == 4
    assert turns[1]["actual_remaining_action_points"] == 3
    assert turns[1]["actual_remaining_action_points_to_finish"] == 3
    assert turns[1]["estimate_action_points"] == 3
    assert turns[1]["actual_action_points"] == 3
    assert turns[1]["action_points_used_before_turn"] == 2
    assert turns[1]["cumulative_action_points_used"] == 5
    assert turns[1]["budget_remaining_before_turn"] == 4
    assert turns[1]["budget_remaining_after_turn"] == 1
    assert turns[1]["toolcalls_used"] == 1


def test_finalize_rollout_records_all_toolcall_action_names(tmp_path):
    wrapper = make_toolcall_wrapper(
        tmp_path,
        start_group_index=0,
        max_action_points=6,
    )
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
                            "<remaining_action_points_estimation>5</remaining_action_points_estimation>"
                            "<action_points_estimation>3</action_points_estimation>"
                            "<think>Plan</think><answer>move || pick up bun || cook patty</answer>"
                        ),
                        "llm_response": (
                            "<think>Plan</think><answer>move || pick up bun || cook patty</answer>"
                        ),
                        "actions": ["move", "pick up bun", "cook patty"],
                        "reward": 1.0,
                        "token_count": 48,
                        "action_points_used": 3,
                        "info": {"success": True},
                    },
                ],
            }
        ]
    )

    turn = wrapper._estimation_records[0]["turns"][0]
    assert turn["actions"] == ["move", "pick up bun", "cook patty"]
    assert turn["action_names"] == ["move", "pick up bun", "cook patty"]
    assert turn["toolcalls_used"] == 3


def test_finalize_rollout_prefers_goal_predicate_ratio_in_json_reward(tmp_path):
    wrapper = make_toolcall_wrapper(
        tmp_path,
        start_group_index=0,
        max_action_points=6,
    )
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
                        "llm_raw_response": "<answer>move</answer>",
                        "llm_response": "<answer>move</answer>",
                        "actions": ["move"],
                        "reward": 1.0,
                        "token_count": 16,
                        "action_points_used": 1,
                        "info": {
                            "success": True,
                            "goal_predicates_satisfied": 2,
                            "goal_predicates_total": 5,
                            "goal_predicate_ratio_reward": 0.4,
                        },
                    },
                ],
            }
        ]
    )

    record = wrapper._estimation_records[0]
    turn = record["turns"][0]
    assert record["success"] is True
    assert record["final_reward"] == 0.4
    assert record["final_reward_source"] == "goal_predicate_ratio"
    assert record["final_rollout_reward"] == 1.0
    assert record["final_goal_predicate_ratio_reward"] == 0.4
    assert turn["reward"] == 0.4
    assert turn["rollout_reward"] == 1.0
    assert turn["reward_source"] == "goal_predicate_ratio"
    assert turn["goal_predicates_satisfied"] == 2
    assert turn["goal_predicates_total"] == 5
    assert turn["goal_predicate_ratio_reward"] == 0.4


def test_finalize_rollout_env_final_reward_keeps_last_goal_predicate_ratio(tmp_path):
    wrapper = make_toolcall_wrapper(
        tmp_path,
        start_group_index=0,
        max_action_points=6,
    )
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
                        "llm_raw_response": "<answer>move</answer>",
                        "llm_response": "<answer>move</answer>",
                        "actions": ["move"],
                        "reward": 1.0,
                        "token_count": 16,
                        "action_points_used": 1,
                        "info": {
                            "success": False,
                            "goal_predicates_satisfied": 2,
                            "goal_predicates_total": 4,
                            "goal_predicate_ratio_reward": 0.5,
                        },
                    },
                    {
                        "state": "S1",
                        "llm_raw_response": "<answer></answer>",
                        "llm_response": "<answer></answer>",
                        "actions": [],
                        "reward": 0.0,
                        "token_count": 4,
                        "action_points_used": 0,
                        "info": {"success": False},
                    },
                ],
            }
        ]
    )

    record = wrapper._estimation_records[0]
    turns = record["turns"]
    assert record["success"] is False
    assert record["final_reward"] == 0.5
    assert record["final_reward_source"] == "goal_predicate_ratio"
    assert record["final_rollout_reward"] == 0.0
    assert record["final_goal_predicate_ratio_reward"] == 0.5
    assert turns[0]["reward"] == 0.5
    assert turns[0]["reward_source"] == "goal_predicate_ratio"
    assert turns[1]["reward"] == 0.5
    assert turns[1]["rollout_reward"] == 0.0
    assert turns[1]["reward_source"] == "goal_predicate_ratio_carry_forward"
    assert turns[1]["goal_predicate_ratio_reward"] == 0.5
    assert turns[1]["goal_predicates_satisfied"] == 2
    assert turns[1]["goal_predicates_total"] == 4


def test_eval_compliance_log_uses_compliance_suffix(tmp_path):
    wrapper = make_compliance_wrapper(tmp_path, start_group_index=0)

    assert wrapper.get_estimation_log_path().endswith("_eval_compliance_dialogues.json")
    assert wrapper.get_eval_log_key() == "eval_compliance_json_path"


def test_eval_compliance_prompt_uses_env_specific_token_limit(tmp_path):
    wrapper = make_compliance_wrapper(tmp_path, start_group_index=0)
    messages_list = [[{"role": "user", "content": "Question"}]]

    wrapper._inject_eval_compliance_token_prompt(messages_list, env_ids=[1], group_ids=[0])

    assert "You must finish your answer in 200 tokens." in messages_list[0][0]["content"]


def test_eval_compliance_token_limit_repeats_for_original_group_copies(tmp_path):
    wrapper = make_compliance_wrapper(
        tmp_path,
        start_group_index=0,
        token_scope=[100, 200, 300],
        base_group_size=2,
    )

    assert wrapper._get_eval_compliance_token_limit_for_env(env_id=0, group_id=0) == 100
    assert wrapper._get_eval_compliance_token_limit_for_env(env_id=1, group_id=0) == 100
    assert wrapper._get_eval_compliance_token_limit_for_env(env_id=2, group_id=0) == 200
    assert wrapper._get_eval_compliance_token_limit_for_env(env_id=3, group_id=0) == 200
    assert wrapper._get_eval_compliance_token_limit_for_env(env_id=4, group_id=0) == 300
    assert wrapper._get_eval_compliance_token_limit_for_env(env_id=5, group_id=0) == 300


def test_eval_compliance_finalize_rollout_records_answered_within_limit(tmp_path):
    wrapper = make_compliance_wrapper(tmp_path, start_group_index=0)
    wrapper.begin_rollout()
    wrapper.finalize_rollout(
        [
            {
                "env_id": 0,
                "group_id": 0,
                "uid": None,
                "tag": "CoordSokoban",
                "history": [
                    {
                        "state": "S1",
                        "llm_raw_response": "<budget-thinking>Budget</budget-thinking><answer>Right</answer>",
                        "llm_response": "<answer>Right</answer>",
                        "actions": ["Right"],
                        "reward": 1.0,
                        "token_count": 80,
                        "info": {"success": True},
                    },
                ],
            },
            {
                "env_id": 1,
                "group_id": 0,
                "uid": None,
                "tag": "CoordSokoban",
                "history": [
                    {
                        "state": "S1",
                        "llm_raw_response": "<answer>Long answer</answer>",
                        "llm_response": "<answer>Long answer</answer>",
                        "actions": ["Right"],
                        "reward": 1.0,
                        "token_count": 240,
                        "info": {"success": True},
                    },
                ],
            }
        ]
    )

    record_0 = wrapper._estimation_records[0]
    turns_0 = record_0["turns"]
    assert record_0["mode"] == "compliance_token"
    assert record_0["eval_compliance_token_scope"] == [100, 200, 300, 400, 500]
    assert record_0["compliance_token_limit"] == 100
    assert turns_0[0]["compliance_token_limit"] == 100
    assert turns_0[0]["within_token_limit"] is True
    assert turns_0[0]["answered_within_token_limit"] is True

    record_1 = wrapper._estimation_records[1]
    turns_1 = record_1["turns"]
    assert record_1["compliance_token_limit"] == 200
    assert turns_1[0]["compliance_token_limit"] == 200
    assert turns_1[0]["within_token_limit"] is False
    assert turns_1[0]["answered_within_token_limit"] is False


def test_eval_compliance_requires_non_empty_scope(tmp_path):
    try:
        make_compliance_wrapper(tmp_path, start_group_index=0, token_scope=[])
    except ValueError as exc:
        assert "eval_compliance_token_scope is empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty eval_compliance_token_scope")


def test_eval_turn_compliance_log_uses_compliance_suffix(tmp_path):
    wrapper = make_turn_compliance_wrapper(tmp_path, start_group_index=0)

    assert wrapper.get_estimation_log_path().endswith("_eval_compliance_dialogues.json")
    assert wrapper.get_eval_log_key() == "eval_compliance_json_path"


def test_eval_turn_compliance_prompt_uses_env_specific_turn_limit(tmp_path):
    wrapper = make_turn_compliance_wrapper(tmp_path, start_group_index=0)
    wrapper.turn_idx = 2
    messages_list = [[{"role": "user", "content": "Question"}]]

    wrapper._inject_eval_compliance_turn_prompt(messages_list, env_ids=[0], group_ids=[0])

    prompt = messages_list[0][0]["content"]
    assert "[Turn Budget Compliance]" in prompt
    assert "Budget turn: 1." in prompt
    assert "Current turn: 3." in prompt
    assert "You have already exceeded this budget by 2 turn(s)." in prompt


def test_eval_turn_compliance_limit_repeats_for_original_group_copies(tmp_path):
    wrapper = make_turn_compliance_wrapper(
        tmp_path,
        start_group_index=0,
        turn_scope=[1, 2, 3],
        base_group_size=2,
    )

    assert wrapper._get_eval_compliance_turn_limit_for_env(env_id=0, group_id=0) == 1
    assert wrapper._get_eval_compliance_turn_limit_for_env(env_id=1, group_id=0) == 1
    assert wrapper._get_eval_compliance_turn_limit_for_env(env_id=2, group_id=0) == 2
    assert wrapper._get_eval_compliance_turn_limit_for_env(env_id=3, group_id=0) == 2
    assert wrapper._get_eval_compliance_turn_limit_for_env(env_id=4, group_id=0) == 3
    assert wrapper._get_eval_compliance_turn_limit_for_env(env_id=5, group_id=0) == 3


def test_eval_turn_compliance_finalize_rollout_records_turn_budget_fields(tmp_path):
    wrapper = make_turn_compliance_wrapper(tmp_path, start_group_index=0)
    wrapper.begin_rollout()
    wrapper.finalize_rollout(
        [
            {
                "env_id": 0,
                "group_id": 0,
                "uid": None,
                "tag": "CoordSokoban",
                "history": [
                    {
                        "state": "S1",
                        "llm_raw_response": "<answer>Right</answer>",
                        "llm_response": "<answer>Right</answer>",
                        "actions": ["Right"],
                        "reward": 1.0,
                        "token_count": 80,
                        "info": {"success": True},
                    },
                ],
            },
            {
                "env_id": 1,
                "group_id": 0,
                "uid": None,
                "tag": "CoordSokoban",
                "history": [
                    {
                        "state": "S1",
                        "llm_raw_response": "<answer>Step1</answer>",
                        "llm_response": "<answer>Step1</answer>",
                        "actions": ["Right"],
                        "reward": 0.0,
                        "token_count": 60,
                        "info": {"success": False},
                    },
                    {
                        "state": "S2",
                        "llm_raw_response": "<answer>Step2</answer>",
                        "llm_response": "<answer>Step2</answer>",
                        "actions": ["Down"],
                        "reward": 0.0,
                        "token_count": 60,
                        "info": {"success": False},
                    },
                    {
                        "state": "S3",
                        "llm_raw_response": "<answer>Finish</answer>",
                        "llm_response": "<answer>Finish</answer>",
                        "actions": ["Left"],
                        "reward": 1.0,
                        "token_count": 60,
                        "info": {"success": True},
                    },
                ],
            },
        ]
    )

    record_0 = wrapper._estimation_records[0]
    turns_0 = record_0["turns"]
    assert record_0["mode"] == "compliance_turn"
    assert record_0["eval_compliance_turn_scope"] == [1, 2, 3, 4, 5]
    assert record_0["compliance_turn_limit"] == 1
    assert record_0["total_turns"] == 1
    assert record_0["within_turn_limit"] is True
    assert record_0["turn_limit_delta"] == 0
    assert record_0["success_within_turn_limit"] is True
    assert turns_0[0]["compliance_turn_limit"] == 1
    assert turns_0[0]["current_turn"] == 1
    assert turns_0[0]["turn_budget_distance"] == 0
    assert turns_0[0]["within_turn_limit_so_far"] is True
    assert turns_0[0]["exceeded_turn_limit"] is False

    record_1 = wrapper._estimation_records[1]
    turns_1 = record_1["turns"]
    assert record_1["compliance_turn_limit"] == 2
    assert record_1["total_turns"] == 3
    assert record_1["within_turn_limit"] is False
    assert record_1["turn_limit_delta"] == 1
    assert record_1["success_within_turn_limit"] is False
    assert turns_1[2]["compliance_turn_limit"] == 2
    assert turns_1[2]["current_turn"] == 3
    assert turns_1[2]["turn_budget_distance"] == -1
    assert turns_1[2]["within_turn_limit_so_far"] is False
    assert turns_1[2]["exceeded_turn_limit"] is True
    assert "You have already exceeded this budget by 1 turn(s)." in turns_1[2]["compliance_instruction"]


def test_eval_turn_compliance_requires_non_empty_scope(tmp_path):
    try:
        make_turn_compliance_wrapper(tmp_path, start_group_index=0, turn_scope=[])
    except ValueError as exc:
        assert "neither agent_proxy.eval_compliance_turn_scope nor the mutation-based turn compliance configuration is set" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty eval_compliance_turn_scope")


def test_eval_turn_compliance_prompt_uses_mutated_budget_after_threshold(tmp_path):
    wrapper = make_turn_compliance_wrapper(
        tmp_path,
        start_group_index=0,
        turn_scope=[],
        mutation_turn=2,
        budget_change=[9, 5],
    )
    wrapper.turn_idx = 2
    messages_list = [[{"role": "user", "content": "Question"}]]

    wrapper._inject_eval_compliance_turn_prompt(messages_list, env_ids=[0], group_ids=[0])

    prompt = messages_list[0][0]["content"]
    assert "Budget turn: 5." in prompt
    assert "Current turn: 3." in prompt
    assert "Mutation turn: 2." in prompt
    assert "Budget schedule: 9 before or at the mutation turn, 5 after the mutation turn." in prompt
    assert "You are 2 turn(s) away from this budget." in prompt


def test_adaptation_turn_prompt_uses_active_budget_and_soft_guidance(tmp_path):
    wrapper = make_adaptation_turn_wrapper(tmp_path, start_group_index=0)
    wrapper.set_state(turn_idx=2, max_turn=7)
    messages_list = [[{"role": "user", "content": "Question"}]]

    wrapper._inject_eval_adaptation_turn_prompt(messages_list)

    prompt = messages_list[0][0]["content"]
    assert "[Turn Budget Adaptation]" in prompt
    assert "Suggested budget turn: 3." in prompt
    assert "Current turn: 3." in prompt
    assert "Turns used so far: 3." in prompt
    assert "Hard stop: max_turn 7." in prompt
    assert "This is guidance only, not a hard cutoff." in prompt
    assert "This is the last suggested budgeted turn." in prompt
    assert "Adaptation schedule:" not in prompt


def test_adaptation_turn_finalize_rollout_records_budget_and_accuracy_fields(tmp_path):
    wrapper = make_adaptation_turn_wrapper(tmp_path, start_group_index=0)
    wrapper.begin_rollout()
    wrapper.finalize_rollout(
        [
            {
                "env_id": 0,
                "group_id": 0,
                "uid": None,
                "tag": "CoordSokoban",
                "history": [
                    {
                        "state": "S1",
                        "llm_raw_response": (
                            "<budget-thinking>Budget</budget-thinking>"
                            "<turn_estimation>4</turn_estimation>"
                            "<token_estimation>20</token_estimation>"
                            "<think>Plan</think><answer>Step1</answer>"
                        ),
                        "llm_response": "<think>Plan</think><answer>Step1</answer>",
                        "actions": ["Right"],
                        "reward": 0.0,
                        "token_count": 21,
                        "info": {"success": False},
                    },
                    {
                        "state": "S2",
                        "llm_raw_response": (
                            "<budget-thinking>Budget</budget-thinking>"
                            "<turn_estimation>3</turn_estimation>"
                            "<token_estimation>24</token_estimation>"
                            "<think>Plan</think><answer>Step2</answer>"
                        ),
                        "llm_response": "<think>Plan</think><answer>Step2</answer>",
                        "actions": ["Down"],
                        "reward": 0.0,
                        "token_count": 24,
                        "info": {"success": False},
                    },
                    {
                        "state": "S3",
                        "llm_raw_response": (
                            "<budget-thinking>Budget</budget-thinking>"
                            "<turn_estimation>2</turn_estimation>"
                            "<token_estimation>18</token_estimation>"
                            "<think>Plan</think><answer>Step3</answer>"
                        ),
                        "llm_response": "<think>Plan</think><answer>Step3</answer>",
                        "actions": ["Left"],
                        "reward": 0.0,
                        "token_count": 19,
                        "info": {"success": False},
                    },
                    {
                        "state": "S4",
                        "llm_raw_response": (
                            "<budget-thinking>Budget</budget-thinking>"
                            "<turn_estimation>1</turn_estimation>"
                            "<token_estimation>22</token_estimation>"
                            "<think>Plan</think><answer>Finish</answer>"
                        ),
                        "llm_response": "<think>Plan</think><answer>Finish</answer>",
                        "actions": ["Up"],
                        "reward": 1.0,
                        "token_count": 22,
                        "info": {"success": True},
                    },
                ],
            },
        ]
    )

    record = wrapper._estimation_records[0]
    turns = record["turns"]
    assert record["mode"] == "adaptation_turn"
    assert record["eval_adaptation_turn_scope"] == [2, 5, 3]
    assert record["adaptation_turn_mutation_turn"] == 2
    assert record["adaptation_turn_budget_change"] == [5, 3]
    assert record["adaptation_turn_limit"] == 3
    assert record["within_adaptation_turn_limit"] is False
    assert record["adaptation_turn_limit_delta"] == 1
    assert record["success_within_adaptation_turn_limit"] is False

    assert turns[0]["adaptation_turn_limit"] == 5
    assert turns[0]["within_adaptation_turn_limit_so_far"] is True
    assert turns[0]["estimate_token_diff"] == -1
    assert turns[0]["estimate_token_abs_error"] == 1
    assert turns[0]["estimate_token_exact_match"] is False
    assert turns[0]["estimate_remaining_turn_diff"] == 0
    assert turns[0]["estimate_remaining_turn_abs_error"] == 0
    assert turns[0]["estimate_remaining_turn_exact_match"] is True
    assert turns[2]["adaptation_turn_limit"] == 3
    assert turns[2]["within_adaptation_turn_limit_so_far"] is True
    assert turns[3]["adaptation_turn_limit"] == 3
    assert turns[3]["turn_budget_distance"] == -1
    assert turns[3]["exceeded_adaptation_turn_limit"] is True
    assert "You have already exceeded the suggested budget by 1 turn(s)." in turns[3]["adaptation_instruction"]


def test_adaptation_turn_requires_non_empty_scope(tmp_path):
    try:
        make_adaptation_turn_wrapper(tmp_path, start_group_index=0, adaptation_scope=[])
    except ValueError as exc:
        assert "eval_adaptation_turn_scope is empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty eval_adaptation_turn_scope")


def test_adaptation_turn_requires_exactly_three_scope_values(tmp_path):
    try:
        make_adaptation_turn_wrapper(tmp_path, start_group_index=0, adaptation_scope=[2, 5])
    except ValueError as exc:
        assert "must contain exactly three integers" in str(exc)
    else:
        raise AssertionError("Expected ValueError for malformed eval_adaptation_turn_scope")


def test_eval_turn_compliance_finalize_rollout_records_mutation_budget_fields(tmp_path):
    wrapper = make_turn_compliance_wrapper(
        tmp_path,
        start_group_index=0,
        turn_scope=[],
        mutation_turn=2,
        budget_change=[9, 5],
    )
    wrapper.begin_rollout()
    wrapper.finalize_rollout(
        [
            {
                "env_id": 0,
                "group_id": 0,
                "uid": None,
                "tag": "CoordSokoban",
                "history": [
                    {
                        "state": "S1",
                        "llm_raw_response": "<answer>Step1</answer>",
                        "llm_response": "<answer>Step1</answer>",
                        "actions": ["Right"],
                        "reward": 0.0,
                        "token_count": 20,
                        "info": {"success": False},
                    },
                    {
                        "state": "S2",
                        "llm_raw_response": "<answer>Step2</answer>",
                        "llm_response": "<answer>Step2</answer>",
                        "actions": ["Down"],
                        "reward": 0.0,
                        "token_count": 20,
                        "info": {"success": False},
                    },
                    {
                        "state": "S3",
                        "llm_raw_response": "<answer>Finish</answer>",
                        "llm_response": "<answer>Finish</answer>",
                        "actions": ["Left"],
                        "reward": 1.0,
                        "token_count": 20,
                        "info": {"success": True},
                    },
                ],
            },
        ]
    )

    record = wrapper._estimation_records[0]
    turns = record["turns"]
    assert record["eval_compliance_turn_mutation_turn"] == 2
    assert record["eval_compliance_turn_budget_change"] == [9, 5]
    assert record["compliance_turn_limit"] == 5
    assert record["within_turn_limit"] is True
    assert turns[0]["compliance_turn_limit"] == 9
    assert turns[0]["turn_budget_distance"] == 8
    assert turns[1]["compliance_turn_limit"] == 9
    assert turns[2]["compliance_turn_limit"] == 5
    assert turns[2]["turn_budget_distance"] == 2


def test_eval_turn_compliance_rejects_scope_and_mutation_combined(tmp_path):
    try:
        make_turn_compliance_wrapper(
            tmp_path,
            start_group_index=0,
            turn_scope=[1, 2],
            mutation_turn=2,
            budget_change=[9, 5],
        )
    except ValueError as exc:
        assert "cannot be used together" in str(exc)
    else:
        raise AssertionError("Expected ValueError for combining scope and mutation turn compliance")


def test_toolcall_eval_estimation_requires_robotouille(tmp_path):
    try:
        make_toolcall_wrapper(
            tmp_path,
            start_group_index=0,
            env_type="sokoban",
        )
    except ValueError as exc:
        assert "only be enabled when all active environments are robotouille" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-robotouille toolcall estimation")


def test_toolcall_eval_estimation_requires_action_budget_enabled(tmp_path):
    try:
        make_toolcall_wrapper(
            tmp_path,
            start_group_index=0,
            enable_action_budget=False,
        )
    except ValueError as exc:
        assert "enable_action_budget=True" in str(exc)
    else:
        raise AssertionError("Expected ValueError when action budget is disabled")


def test_eval_toolcall_compliance_log_uses_compliance_suffix(tmp_path):
    wrapper = make_toolcall_compliance_wrapper(tmp_path, start_group_index=0)

    assert wrapper.get_estimation_log_path().endswith("_eval_compliance_dialogues.json")
    assert wrapper.get_eval_log_key() == "eval_compliance_json_path"


def test_eval_toolcall_compliance_prompt_uses_env_specific_limit_and_remaining_budget(tmp_path):
    wrapper = make_toolcall_compliance_wrapper(tmp_path, start_group_index=0)
    messages_list = [[{"role": "user", "content": "Question"}]]

    wrapper._inject_eval_compliance_toolcall_prompt(
        messages_list,
        env_ids=[1],
        group_ids=[0],
        action_points_used_so_far=[1],
    )

    prompt = messages_list[0][0]["content"]
    assert "[Toolcall Budget Compliance]" in prompt
    assert "within 4 action points" in prompt
    assert "Action points used so far: 1." in prompt
    assert "You can still use 3 action point(s) within this budget." in prompt


def test_mixed_toolcall_budget_prompt_uses_budget_and_usage(tmp_path):
    wrapper = make_mixed_toolcall_budget_wrapper(tmp_path, start_group_index=0)
    messages_list = [[{"role": "user", "content": "Question"}]]

    wrapper._inject_mixed_toolcall_budget_prompt(
        messages_list,
        budget_toolcalls=[5],
        action_points_used_so_far=[2],
    )

    prompt = messages_list[0][0]["content"]
    assert "[Toolcall Budget Guidance]" in prompt
    assert "finish within 5 action points total" in prompt
    assert "Action points used so far: 2." in prompt
    assert "There are 3 action point(s) left within this budget." in prompt
    assert "penalized if you finish after this budget" in prompt


def test_eval_toolcall_compliance_limit_repeats_for_original_group_copies(tmp_path):
    wrapper = make_toolcall_compliance_wrapper(
        tmp_path,
        start_group_index=0,
        toolcall_scope=[2, 4, 6],
        base_group_size=2,
    )

    assert wrapper._get_eval_compliance_toolcall_limit_for_env(env_id=0, group_id=0) == 2
    assert wrapper._get_eval_compliance_toolcall_limit_for_env(env_id=1, group_id=0) == 2
    assert wrapper._get_eval_compliance_toolcall_limit_for_env(env_id=2, group_id=0) == 4
    assert wrapper._get_eval_compliance_toolcall_limit_for_env(env_id=3, group_id=0) == 4
    assert wrapper._get_eval_compliance_toolcall_limit_for_env(env_id=4, group_id=0) == 6
    assert wrapper._get_eval_compliance_toolcall_limit_for_env(env_id=5, group_id=0) == 6


def test_eval_toolcall_compliance_finalize_rollout_records_budget_fields(tmp_path):
    wrapper = make_toolcall_compliance_wrapper(
        tmp_path,
        start_group_index=0,
        toolcall_scope=[2, 4, 6],
        max_action_points=6,
    )
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
                        "llm_raw_response": "<answer>move</answer>",
                        "llm_response": "<answer>move</answer>",
                        "actions": ["move"],
                        "reward": 1.0,
                        "token_count": 32,
                        "action_points_used": 2,
                        "info": {"success": True},
                    },
                ],
            },
            {
                "env_id": 1,
                "group_id": 0,
                "uid": None,
                "tag": "Robotouille",
                "history": [
                    {
                        "state": "S1",
                        "llm_raw_response": "<answer>step1</answer>",
                        "llm_response": "<answer>step1</answer>",
                        "actions": ["move"],
                        "reward": 0.0,
                        "token_count": 20,
                        "action_points_used": 2,
                        "info": {"success": False},
                    },
                    {
                        "state": "S2",
                        "llm_raw_response": "<answer>step2</answer>",
                        "llm_response": "<answer>step2</answer>",
                        "actions": ["cook"],
                        "reward": 1.0,
                        "token_count": 20,
                        "action_points_used": 3,
                        "info": {"success": True},
                    },
                ],
            },
        ]
    )

    record_0 = wrapper._estimation_records[0]
    turns_0 = record_0["turns"]
    assert record_0["mode"] == "compliance_toolcall"
    assert record_0["eval_compliance_toolcall_scope"] == [2, 4, 6]
    assert record_0["compliance_toolcall_limit"] == 2
    assert record_0["total_action_points_used"] == 2
    assert record_0["total_toolcalls_used"] == 1
    assert record_0["within_toolcall_limit"] is True
    assert record_0["toolcall_limit_delta"] == 0
    assert record_0["success_within_toolcall_limit"] is True
    assert turns_0[0]["compliance_toolcall_limit"] == 2
    assert turns_0[0]["action_points_used_before_turn"] == 0
    assert turns_0[0]["remaining_action_points_before_turn"] == 2
    assert turns_0[0]["cumulative_action_points_used"] == 2
    assert turns_0[0]["remaining_action_points_after_turn"] == 0
    assert turns_0[0]["within_toolcall_limit_so_far"] is True
    assert turns_0[0]["exceeded_toolcall_limit"] is False

    record_1 = wrapper._estimation_records[1]
    turns_1 = record_1["turns"]
    assert record_1["compliance_toolcall_limit"] == 4
    assert record_1["total_action_points_used"] == 5
    assert record_1["total_toolcalls_used"] == 2
    assert record_1["within_toolcall_limit"] is False
    assert record_1["toolcall_limit_delta"] == 1
    assert record_1["success_within_toolcall_limit"] is False
    assert turns_1[1]["action_points_used_before_turn"] == 2
    assert turns_1[1]["remaining_action_points_before_turn"] == 2
    assert turns_1[1]["cumulative_action_points_used"] == 5
    assert turns_1[1]["remaining_action_points_after_turn"] == -1
    assert turns_1[1]["within_toolcall_limit_so_far"] is False
    assert turns_1[1]["exceeded_toolcall_limit"] is True
    assert "Action points used so far: 2." in turns_1[1]["compliance_instruction"]
    assert "You can still use 2 action point(s) within this budget." in turns_1[1]["compliance_instruction"]


def test_toolcall_eval_finalize_rollout_prefers_env_specific_budget_toolcall(tmp_path):
    wrapper = make_toolcall_wrapper(
        tmp_path,
        start_group_index=0,
        max_action_points=10,
    )
    wrapper.begin_rollout()
    wrapper.finalize_rollout(
        [
            {
                "env_id": 0,
                "group_id": 0,
                "uid": None,
                "tag": "Robotouille",
                "budget_toolcall": 4,
                "history": [
                    {
                        "state": "S1",
                        "llm_raw_response": "<answer>move</answer>",
                        "llm_response": "<answer>move</answer>",
                        "actions": ["move"],
                        "reward": 1.0,
                        "token_count": 24,
                        "action_points_used": 2,
                        "info": {"success": True},
                    },
                ],
            },
        ]
    )

    record = wrapper._estimation_records[0]
    turn = record["turns"][0]
    assert record["budget_toolcall"] == 4
    assert record["max_action_points"] == 4
    assert turn["budget_toolcall"] == 4
    assert turn["max_action_points"] == 4
    assert turn["budget_remaining_before_turn"] == 4
    assert turn["budget_remaining_after_turn"] == 2


def test_eval_toolcall_compliance_requires_non_empty_scope(tmp_path):
    try:
        make_toolcall_compliance_wrapper(tmp_path, start_group_index=0, toolcall_scope=[])
    except ValueError as exc:
        assert "eval_compliance_toolcall_scope is empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty eval_compliance_toolcall_scope")


def test_eval_toolcall_compliance_requires_robotouille(tmp_path):
    try:
        config = OmegaConf.create(
            {
                "agent_proxy": {
                    "enable_ctx_wrapper": True,
                    "eval_compliance_toolcall": True,
                    "eval_compliance_toolcall_scope": [2],
                },
                "custom_envs": {
                    "Sokoban": {
                        "env_type": "sokoban",
                        "env_config": {},
                    }
                },
                "output": {
                    "dir": str(tmp_path),
                    "filename": "toolcall_compliance_api_eval.pkl",
                },
                "es_manager": {
                    "train": {
                        "env_configs": {"tags": ["Sokoban"]},
                        "group_size": 1,
                    },
                    "val": {
                        "start_group_index": 0,
                        "group_size": 1,
                        "env_configs": {"tags": ["Sokoban"]},
                    },
                },
            }
        )
        CtxManagerWrapper(config, DummyTokenizer())
    except ValueError as exc:
        assert "eval_compliance_toolcall can only be enabled when all active environments are robotouille" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-robotouille toolcall compliance")
