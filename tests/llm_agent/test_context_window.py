import pytest
from ragen.llm_agent.ctx_manager import ContextManager
from omegaconf import OmegaConf
from verl.protocol import DataProto

class DummyTokenizer:
    name_or_path = "qwen"  # or "llama-3" or any string your code expects

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return " ".join([msg["content"] for msg in messages])

    def __call__(self, texts, return_tensors, padding, padding_side, truncation):
        import torch
        batch_size = len(texts) if isinstance(texts, list) else 1
        class DummyOutput:
            input_ids = torch.tensor([[1, 2, 3]] * batch_size)
            attention_mask = torch.tensor([[1, 1, 1]] * batch_size)
        return DummyOutput()

    def encode(self, text):
        # Return a dummy list of token ids; must be at least length 1 for [0] indexing
        return [42, 43]

@pytest.fixture
def dummy_config():
    cfg = OmegaConf.create({
        "agent_proxy": {
            "context_window_mode": "single_turn",
            "max_context_window": 2,
            "enable_think": False,
            "use_turn_scores": False,
            "action_sep": "|",
            "max_actions_per_turn": 2,
            "reward_normalization": {
                "grouping": "batch",
                "method": "identity"
            }
        },
        "enable_response_mask": False,
        "es_manager": {
            "train": {
                "env_configs": {
                    "n_groups": [1],
                    "tags": ["sokoban"]
                },
                "group_size": 1
            }
        },
        "custom_envs": {
            "sokoban": {
                "env_type": "sokoban",
                "max_actions_per_traj": 10
            }
        },
        "actor_rollout_ref": {
            "rollout": {
                "response_length": 128
            }
        }
    })
    return cfg

def test_context_window_truncation(dummy_config):
    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=dummy_config, tokenizer=tokenizer, mode="train")
    ctx.prefix_lookup = {0: "Initial prompt"}
    ctx.env_config_lookup = {0: {"max_tokens": 128}}
    ctx.env_nums = {"": 1}  # For metrics

    env_outputs = [{
        "env_id": 0,
        "group_id": 0,
        "history": [
            {"state": "S1", "llm_response": "R1", "reward": 0.1, "actions_left": 5},
            {"state": "S2", "llm_response": "R2", "reward": 0.2, "actions_left": 4},
            {"state": "S3", "llm_response": "R3", "reward": 0.3, "actions_left": 3},
        ],
        "metrics": {},
    }]

    lm_inputs: DataProto = ctx.get_lm_inputs(env_outputs, prepare_for_update=True)
    messages = lm_inputs.non_tensor_batch["messages_list"][-1]

    # Ensure only last 2 turns are present
    assert "S1" not in str(messages)
    assert "S2" in str(messages)
    assert "S3" in str(messages)


def test_multi_eval_estimation_generation_prefix_uses_budget_thinking(dummy_config):
    cfg = OmegaConf.create(OmegaConf.to_container(dummy_config, resolve=True))
    cfg.agent_proxy["eval-estimation-single"] = False
    cfg.agent_proxy["eval-estimation-multi"] = True

    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=cfg, tokenizer=tokenizer, mode="train")

    assert ctx._get_generation_prefix() == "<budget-thinking>"


def test_single_eval_estimation_generation_prefix_uses_budget_thinking(dummy_config):
    cfg = OmegaConf.create(OmegaConf.to_container(dummy_config, resolve=True))
    cfg.agent_proxy["eval-estimation-single"] = True
    cfg.agent_proxy["eval-estimation-multi"] = False

    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=cfg, tokenizer=tokenizer, mode="train")

    assert ctx._get_generation_prefix() == "<budget-thinking>"


def test_toolcall_eval_estimation_generation_prefix_uses_budget_thinking(dummy_config):
    cfg = OmegaConf.create(
        {
            "agent_proxy": {
                "context_window_mode": "single_turn",
                "max_context_window": 2,
                "enable_think": True,
                "use_turn_scores": False,
                "action_sep": "|",
                "max_actions_per_turn": 2,
                "eval-estimation-single": False,
                "eval-estimation-multi": False,
                "eval-estimation-toolcall": True,
                "reward_normalization": {
                    "grouping": "batch",
                    "method": "identity"
                }
            },
            "enable_response_mask": False,
            "es_manager": {
                "train": {
                    "env_configs": {
                        "n_groups": [1],
                        "tags": ["Robotouille"]
                    },
                    "group_size": 1
                }
            },
            "custom_envs": {
                "Robotouille": {
                    "env_type": "robotouille",
                    "max_actions_per_traj": 10,
                    "env_config": {
                        "enable_action_budget": True,
                        "max_action_points": 6,
                    },
                }
            },
            "actor_rollout_ref": {
                "rollout": {
                    "response_length": 128
                }
            }
        }
    )

    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=cfg, tokenizer=tokenizer, mode="train")

    assert ctx._get_generation_prefix() == "<budget-thinking>"


def test_robotouille_omits_placeholder_action_lookup_from_prefix():
    cfg = OmegaConf.create(
        {
            "agent_proxy": {
                "context_window_mode": "single_turn",
                "max_context_window": 2,
                "enable_think": False,
                "use_turn_scores": False,
                "action_sep": "|",
                "max_actions_per_turn": 1,
                "reward_normalization": {
                    "grouping": "batch",
                    "method": "identity",
                },
            },
            "enable_response_mask": False,
            "es_manager": {
                "train": {
                    "env_configs": {
                        "n_groups": [1],
                        "tags": ["Robotouille"],
                    },
                    "group_size": 1,
                }
            },
            "custom_envs": {
                "Robotouille": {
                    "env_type": "robotouille",
                    "max_actions_per_traj": 10,
                }
            },
            "actor_rollout_ref": {
                "rollout": {
                    "response_length": 128,
                }
            },
        }
    )

    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=cfg, tokenizer=tokenizer, mode="train")

    assert "Your available actions are:" not in ctx.prefix_lookup[0]
    assert "Action[0]" not in ctx.prefix_lookup[0]


def test_robotouille_flattens_nested_env_config_for_prompt_metadata():
    cfg = OmegaConf.create(
        {
            "agent_proxy": {
                "context_window_mode": "single_turn",
                "max_context_window": 2,
                "enable_think": False,
                "use_turn_scores": False,
                "action_sep": "|",
                "max_actions_per_turn": 1,
                "reward_normalization": {
                    "grouping": "batch",
                    "method": "identity",
                },
            },
            "enable_response_mask": False,
            "es_manager": {
                "train": {
                    "env_configs": {
                        "n_groups": [1],
                        "tags": ["Robotouille"],
                    },
                    "group_size": 1,
                }
            },
            "custom_envs": {
                "Robotouille": {
                    "env_type": "robotouille",
                    "max_actions_per_traj": 10,
                    "env_config": {
                        "action_lookup": None,
                        "enable_action_budget": True,
                        "max_action_points": 6,
                    },
                }
            },
            "actor_rollout_ref": {
                "rollout": {
                    "response_length": 128,
                }
            },
        }
    )

    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=cfg, tokenizer=tokenizer, mode="train")

    assert "Your available actions are:" not in ctx.prefix_lookup[0]
    assert ctx.env_config_lookup[0]["enable_action_budget"] is True
    assert ctx.env_config_lookup[0]["max_action_points"] == 6


def test_default_generation_prefix_uses_answer_when_eval_estimation_disabled(dummy_config):
    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=dummy_config, tokenizer=tokenizer, mode="train")

    assert ctx._get_generation_prefix() == "<answer>"


def test_token_estimation_does_not_switch_prefix_back_to_budget_thinking(dummy_config):
    cfg = OmegaConf.create(OmegaConf.to_container(dummy_config, resolve=True))
    cfg.agent_proxy["token_estimation"] = True
    cfg.agent_proxy["enable_think"] = True

    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=cfg, tokenizer=tokenizer, mode="train")

    assert ctx._get_generation_prefix() == "<think>"


def test_single_eval_estimation_format_prompt_includes_token_estimation_once(dummy_config):
    cfg = OmegaConf.create(OmegaConf.to_container(dummy_config, resolve=True))
    cfg.agent_proxy["eval-estimation-single"] = True
    cfg.agent_proxy["eval-estimation-multi"] = False
    cfg.agent_proxy["enable_think"] = True

    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=cfg, tokenizer=tokenizer, mode="train")
    ctx.env_config_lookup = {0: {"max_tokens": 128, "env_tag": "GPQAMain", "env_type": "gpqa_main"}}

    format_prompt, _ = ctx._build_format_prompt(0)

    assert format_prompt.count("<budget-thinking>") == 1
    assert "<token_estimation>" in format_prompt


def test_multi_eval_estimation_format_prompt_includes_both_estimates_once(dummy_config):
    cfg = OmegaConf.create(OmegaConf.to_container(dummy_config, resolve=True))
    cfg.agent_proxy["eval-estimation-single"] = False
    cfg.agent_proxy["eval-estimation-multi"] = True
    cfg.agent_proxy["enable_think"] = True

    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=cfg, tokenizer=tokenizer, mode="train")
    ctx.env_config_lookup = {0: {"max_tokens": 128, "env_tag": "CoordSokoban", "env_type": "sokoban"}}

    format_prompt, _ = ctx._build_format_prompt(0)

    assert format_prompt.count("<budget-thinking>") == 1
    assert "<turn_estimation>" in format_prompt
    assert "<token_estimation>" in format_prompt


def test_toolcall_eval_estimation_format_prompt_includes_action_point_estimates(dummy_config):
    cfg = OmegaConf.create(
        {
            "agent_proxy": {
                "context_window_mode": "single_turn",
                "max_context_window": 2,
                "enable_think": True,
                "use_turn_scores": False,
                "action_sep": "|",
                "max_actions_per_turn": 2,
                "eval-estimation-single": False,
                "eval-estimation-multi": False,
                "eval-estimation-toolcall": True,
                "reward_normalization": {
                    "grouping": "batch",
                    "method": "identity"
                }
            },
            "enable_response_mask": False,
            "es_manager": {
                "train": {
                    "env_configs": {
                        "n_groups": [1],
                        "tags": ["Robotouille"]
                    },
                    "group_size": 1
                }
            },
            "custom_envs": {
                "Robotouille": {
                    "env_type": "robotouille",
                    "max_actions_per_traj": 10,
                    "env_config": {
                        "enable_action_budget": True,
                        "max_action_points": 6,
                    },
                }
            },
            "actor_rollout_ref": {
                "rollout": {
                    "response_length": 128
                }
            }
        }
    )

    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=cfg, tokenizer=tokenizer, mode="train")
    ctx.env_config_lookup = {
        0: {
            "max_tokens": 128,
            "env_tag": "Robotouille",
            "env_type": "robotouille",
            "max_action_points": 6,
        }
    }

    format_prompt, _ = ctx._build_format_prompt(0)

    assert format_prompt.count("<budget-thinking>") == 1
    assert "<remaining_action_points_estimation>" in format_prompt
    assert "<action_points_estimation>" in format_prompt


def test_eval_compliance_omits_length_prompt_from_context(dummy_config):
    cfg = OmegaConf.create(OmegaConf.to_container(dummy_config, resolve=True))
    cfg.agent_proxy["eval_compliance_token"] = True
    cfg.agent_proxy["eval_compliance_token_scope"] = [100, 200]
    cfg.agent_proxy["enable_think"] = True

    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=cfg, tokenizer=tokenizer, mode="train")
    ctx.env_config_lookup = {0: {"max_tokens": 128, "env_tag": "CoordSokoban", "env_type": "sokoban"}}

    format_prompt, length_prompt = ctx._build_format_prompt(0)
    turn_content = ctx._build_turn_state_content(
        {"state": "S1", "actions_left": 3},
        turn_number=1,
        env_id=0,
    )

    assert format_prompt == "<think> [Your thoughts] </think> <answer> [your answer] </answer>"
    assert length_prompt == ""
    assert "Max response length:" not in turn_content
    assert "<budget-thinking>" not in turn_content
    assert ctx._get_generation_prefix() == "<think>"


def test_eval_turn_compliance_omits_length_prompt_from_context(dummy_config):
    cfg = OmegaConf.create(OmegaConf.to_container(dummy_config, resolve=True))
    cfg.agent_proxy["eval_compliance_turn"] = True
    cfg.agent_proxy["eval_compliance_turn_scope"] = [1, 2]
    cfg.agent_proxy["enable_think"] = True

    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=cfg, tokenizer=tokenizer, mode="train")
    ctx.env_config_lookup = {0: {"max_tokens": 128, "env_tag": "CoordSokoban", "env_type": "sokoban"}}

    format_prompt, length_prompt = ctx._build_format_prompt(0)
    turn_content = ctx._build_turn_state_content(
        {"state": "S1", "actions_left": 3},
        turn_number=1,
        env_id=0,
    )

    assert format_prompt == "<think> [Your thoughts] </think> <answer> [your answer] </answer>"
    assert length_prompt == ""
    assert "Max response length:" not in turn_content
    assert "<budget-thinking>" not in turn_content
    assert ctx._get_generation_prefix() == "<think>"


def test_eval_toolcall_compliance_omits_length_prompt_from_context(dummy_config):
    cfg = OmegaConf.create(OmegaConf.to_container(dummy_config, resolve=True))
    cfg.agent_proxy["eval_compliance_toolcall"] = True
    cfg.agent_proxy["eval_compliance_toolcall_scope"] = [2, 4]
    cfg.agent_proxy["enable_think"] = True
    cfg.custom_envs = {
        "Robotouille": {
            "env_type": "robotouille",
            "max_actions_per_traj": 10,
            "env_config": {
                "enable_action_budget": True,
                "max_action_points": 6,
            },
        }
    }
    cfg.es_manager.train.env_configs.tags = ["Robotouille"]
    cfg.es_manager.train.env_configs.n_groups = [1]
    cfg.es_manager.train.group_size = 1

    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=cfg, tokenizer=tokenizer, mode="train")
    ctx.env_config_lookup = {
        0: {
            "max_tokens": 128,
            "env_tag": "Robotouille",
            "env_type": "robotouille",
            "max_action_points": 6,
        }
    }

    format_prompt, length_prompt = ctx._build_format_prompt(0)
    turn_content = ctx._build_turn_state_content(
        {"state": "S1", "actions_left": 3},
        turn_number=1,
        env_id=0,
    )

    assert format_prompt == "<think> [Your thoughts] </think> <answer> [your answer] </answer>"
    assert length_prompt == ""
    assert "Max response length:" not in turn_content
    assert "<budget-thinking>" not in turn_content
    assert ctx._get_generation_prefix() == "<think>"


def test_openai_reasoning_eval_estimation_keeps_explicit_reasoning_tags(dummy_config):
    cfg = OmegaConf.create(OmegaConf.to_container(dummy_config, resolve=True))
    cfg.agent_proxy["eval-estimation-single"] = True
    cfg.agent_proxy["eval-estimation-multi"] = False
    cfg.agent_proxy["enable_think"] = True
    cfg["model_config"] = {"model_name": "OpenAI-5.2-Thinking"}
    cfg["model_info"] = {
        "OpenAI-5.2-Thinking": {
            "provider_name": "openai",
            "model_name": "gpt-5.2",
        }
    }

    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=cfg, tokenizer=tokenizer, mode="train")
    ctx.env_config_lookup = {0: {"max_tokens": 128, "env_tag": "GPQAMain", "env_type": "gpqa_main"}}

    format_prompt, _ = ctx._build_format_prompt(0)

    assert format_prompt.count("<budget-thinking>") == 1
    assert "<think>" in format_prompt
    assert "<token_estimation>" in format_prompt
    assert "<answer>" in format_prompt


def test_openai_reasoning_parse_response_requires_visible_think_tags(dummy_config):
    cfg = OmegaConf.create(OmegaConf.to_container(dummy_config, resolve=True))
    cfg.agent_proxy["enable_think"] = True
    cfg["model_config"] = {"model_name": "OpenAI-5.2-Thinking"}
    cfg["model_info"] = {
        "OpenAI-5.2-Thinking": {
            "provider_name": "openai",
            "model_name": "gpt-5.2",
        }
    }

    tokenizer = DummyTokenizer()
    ctx = ContextManager(config=cfg, tokenizer=tokenizer, mode="train")

    llm_response, actions = ctx._parse_response("<think>Plan</think><answer>Right</answer>")

    assert llm_response == "<think>Plan</think><answer>Right</answer>"
    assert actions == ["Right"]
