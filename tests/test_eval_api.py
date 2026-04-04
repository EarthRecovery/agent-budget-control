from omegaconf import OmegaConf

from ragen.eval_api_utils import clone_config_for_val_chunk, iter_val_rollout_chunks
from ragen.llm_agent.eval_config import expand_compliance_group_size, resolve_rollout_max_turn


def _make_config():
    return OmegaConf.create(
        {
            "es_manager": {
                "val": {
                    "env_groups": 8,
                    "group_size": 1,
                    "start_group_index": 100,
                    "rollout_chunk_size": 0,
                    "env_configs": {
                        "tags": ["EnvA", "EnvB", "EnvC"],
                        "n_groups": [2, 3, 3],
                    },
                }
            }
        }
    )


def test_iter_val_rollout_chunks_disabled_when_chunk_size_is_missing():
    assert iter_val_rollout_chunks(512, 7, None) == [(0, 7, 512)]


def test_iter_val_rollout_chunks_splits_remainder():
    assert iter_val_rollout_chunks(70, 10, 32) == [
        (0, 10, 32),
        (32, 42, 32),
        (64, 74, 6),
    ]


def test_clone_config_for_val_chunk_slices_env_groups_and_preserves_original():
    config = _make_config()

    chunk_config = clone_config_for_val_chunk(
        config,
        chunk_offset=4,
        chunk_start_group_index=104,
        chunk_env_groups=3,
    )

    assert chunk_config.es_manager.val.start_group_index == 104
    assert chunk_config.es_manager.val.env_groups == 3
    assert list(chunk_config.es_manager.val.env_configs.tags) == ["EnvB", "EnvC"]
    assert list(chunk_config.es_manager.val.env_configs.n_groups) == [1, 2]

    assert config.es_manager.val.start_group_index == 100
    assert config.es_manager.val.env_groups == 8
    assert list(config.es_manager.val.env_configs.tags) == ["EnvA", "EnvB", "EnvC"]
    assert list(config.es_manager.val.env_configs.n_groups) == [2, 3, 3]


def test_resolve_rollout_max_turn_keeps_agent_proxy_max_turn_in_compliance_mode():
    config = OmegaConf.create(
        {
            "agent_proxy": {
                "max_turn": 9,
                "eval_compliance_token": True,
                "eval_compliance_token_scope": [100, 200, 300, 400, 500],
            }
        }
    )

    assert resolve_rollout_max_turn(config) == 9


def test_expand_compliance_group_size_multiplies_val_and_train_group_size_once():
    config = OmegaConf.create(
        {
            "agent_proxy": {
                "eval_compliance_token": True,
                "eval_compliance_token_scope": [100, 200, 300, 400, 500],
            },
            "es_manager": {
                "train": {"group_size": 2},
                "val": {"group_size": 1},
            },
        }
    )

    expand_compliance_group_size(config)
    expand_compliance_group_size(config)

    assert config.es_manager.train.group_size == 10
    assert config.es_manager.val.group_size == 5
    assert config.agent_proxy.eval_compliance_group_size_expanded is True


def test_expand_turn_compliance_group_size_multiplies_val_and_train_group_size_once():
    config = OmegaConf.create(
        {
            "agent_proxy": {
                "eval_compliance_turn": True,
                "eval_compliance_turn_scope": [1, 2, 3],
            },
            "es_manager": {
                "train": {"group_size": 2},
                "val": {"group_size": 1},
            },
        }
    )

    expand_compliance_group_size(config)
    expand_compliance_group_size(config)

    assert config.es_manager.train.group_size == 6
    assert config.es_manager.val.group_size == 3
    assert config.agent_proxy.eval_compliance_group_size_expanded is True
