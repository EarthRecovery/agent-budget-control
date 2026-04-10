import pytest
from omegaconf import OmegaConf
from ragen.llm_agent.es_manager import EnvStateManager


def make_cfg():
    return OmegaConf.create({
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
