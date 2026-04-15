import pytest
from omegaconf import OmegaConf
from ragen.llm_agent.es_manager import EnvStateManager


def make_cfg():
    return OmegaConf.create({
        'agent_proxy': {},
        'seed': {'train': 7},
        'es_manager': {
            'train': {
                'env_groups': 1,
                'group_size': 1,
                'env_configs': {'tags': ['Bandit'], 'n_groups': [1]},
            }
        },
        'custom_envs': {
            'Bandit': {
                'env_type': 'bandit',
                'max_actions_per_traj': 1,
                'env_config': None
            }
        }
    })


def test_seed_iteration():
    cfg = make_cfg()
    es = EnvStateManager(cfg, mode='train')
    es.reset()
    first_seed = es.envs[0]['status'].seed
    es.reset()
    second_seed = es.envs[0]['status'].seed
    assert first_seed == 7
    assert second_seed == 8


def test_mixed_toolcall_budget_updates_robotouille_env_config():
    cfg = OmegaConf.create(
        {
            "agent_proxy": {
                "mixed_toolcall_budget": {
                    "enabled": True,
                    "mixed_budget": True,
                    "mixed_budget_range": [4, 4],
                }
            },
            "es_manager": {
                "train": {
                    "env_groups": 1,
                    "group_size": 1,
                    "env_configs": {"tags": ["Robotouille"], "n_groups": [1]},
                }
            },
            "custom_envs": {
                "Robotouille": {
                    "env_type": "robotouille",
                    "max_actions_per_traj": 10,
                    "env_config": {
                        "env_name": "synchronous/0_cheese_sandwich",
                        "enable_action_budget": False,
                        "max_action_points": 10,
                        "action_lookup": None,
                    },
                }
            },
        }
    )

    es = EnvStateManager(cfg, mode="train")

    assert es.envs[0]["budget_toolcall"] == 4
    assert es.envs[0]["config"].enable_action_budget is True
    assert es.envs[0]["config"].max_action_points == 4


def test_token_truncation_stops_rollout_before_action_execution():
    cfg = OmegaConf.create(
        {
            "agent_proxy": {
                "truncation_mode": "token",
                "max_context_token": 10,
            },
            "seed": {"train": 7},
            "es_manager": {
                "format_penalty": 0.0,
                "train": {
                    "env_groups": 1,
                    "group_size": 1,
                    "env_configs": {"tags": ["Bandit"], "n_groups": [1]},
                },
            },
            "custom_envs": {
                "Bandit": {
                    "env_type": "bandit",
                    "max_actions_per_traj": 1,
                    "env_config": None,
                }
            },
        }
    )

    es = EnvStateManager(cfg, mode="train")
    es.reset()
    action_name = next(iter(es.envs[0]["env"].config.action_lookup.values()))

    env_outputs = es.step(
        [
            {
                "env_id": 0,
                "llm_raw_response": f"<answer>{action_name}</answer>",
                "llm_response": f"<answer>{action_name}</answer>",
                "actions": [action_name],
                "response_tokens": 3,
                "input_tokens": 8,
                "output_tokens": 5,
                "total_tokens": 13,
            }
        ]
    )

    assert env_outputs == []
    assert es.envs[0]["status"].terminated is True
    assert es.envs[0]["status"].truncated is True
    assert es.envs[0]["status"].num_actions == 0
    turn_record = es.rollout_cache[0]["history"][-2]
    assert turn_record["actions"] == []
    assert turn_record["api_total_tokens"] == 13
    assert turn_record["info"]["context_token_truncated"] is True
    assert turn_record["info"]["context_token_limit"] == 10


def test_token_truncation_ignores_previous_turn_usage():
    cfg = OmegaConf.create(
        {
            "agent_proxy": {
                "truncation_mode": "token",
                "max_context_token": 10,
            },
            "seed": {"train": 7},
            "es_manager": {
                "format_penalty": 0.0,
                "train": {
                    "env_groups": 1,
                    "group_size": 1,
                    "env_configs": {"tags": ["Bandit"], "n_groups": [1]},
                },
            },
            "custom_envs": {
                "Bandit": {
                    "env_type": "bandit",
                    "max_actions_per_traj": 2,
                    "env_config": None,
                }
            },
        }
    )

    es = EnvStateManager(cfg, mode="train")
    es.reset()
    action_name = next(iter(es.envs[0]["env"].config.action_lookup.values()))
    es.rollout_cache[0]["history"][-1].update(
        {
            "actions": [],
            "reward": 0.0,
            "info": {},
            "llm_response": "",
            "llm_raw_response": "",
            "api_total_tokens": 100,
        }
    )

    es.step(
        [
            {
                "env_id": 0,
                "llm_raw_response": f"<answer>{action_name}</answer>",
                "llm_response": f"<answer>{action_name}</answer>",
                "actions": [action_name],
                "response_tokens": 3,
                "input_tokens": 4,
                "output_tokens": 4,
                "total_tokens": 8,
            }
        ]
    )

    turn_record = es.rollout_cache[0]["history"][-2]
    assert turn_record["api_total_tokens"] == 8
    assert turn_record["actions"] != []
    assert turn_record["info"].get("context_token_truncated") is not True
    assert es.envs[0]["status"].num_actions == 1
