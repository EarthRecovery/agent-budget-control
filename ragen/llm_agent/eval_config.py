from typing import Any, List, Optional
from omegaconf import open_dict

from ragen.env import REGISTERED_ENV_CONFIGS


def agent_proxy_cfg_get(config, key: str, default=None):
    agent_cfg = getattr(config, "agent_proxy", None)
    if agent_cfg is None:
        return default
    if hasattr(agent_cfg, "get"):
        value = agent_cfg.get(key, None)
        if value is None:
            value = agent_cfg.get(key.replace("-", "_"), None)
        return default if value is None else value
    return getattr(agent_cfg, key.replace("-", "_"), default)


def env_cfg_get(env_cfg: Any, key: str, default=None):
    if env_cfg is None:
        return default
    if hasattr(env_cfg, "get"):
        value = env_cfg.get(key, None)
        return default if value is None else value
    return getattr(env_cfg, key, default)


def _normalize_eval_compliance_scope(raw_scope, config_key: str) -> List[int]:
    if raw_scope is None:
        return []
    if hasattr(raw_scope, "tolist"):
        raw_scope = raw_scope.tolist()
    if isinstance(raw_scope, (str, bytes)):
        values = [raw_scope]
    else:
        try:
            values = list(raw_scope)
        except TypeError:
            values = [raw_scope]

    scope = []
    for idx, value in enumerate(values):
        try:
            scope.append(max(0, int(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"agent_proxy.{config_key} must contain integers. "
                f"Invalid value at index {idx}: {value!r}."
            ) from exc
    return scope


def resolve_eval_estimation_mode(config) -> Optional[str]:
    single_enabled = bool(agent_proxy_cfg_get(config, "eval-estimation-single", False))
    multi_enabled = bool(agent_proxy_cfg_get(config, "eval-estimation-multi", False))
    toolcall_enabled = bool(agent_proxy_cfg_get(config, "eval-estimation-toolcall", False))
    adaptation_turn_enabled = bool(agent_proxy_cfg_get(config, "eval_adaptation_turn", False))
    enabled_modes = [
        mode_name
        for mode_name, enabled in (
            ("single", single_enabled),
            ("multi", multi_enabled),
            ("toolcall", toolcall_enabled),
            ("adaptation_turn", adaptation_turn_enabled),
        )
        if enabled
    ]
    if len(enabled_modes) > 1:
        raise ValueError(
            "agent_proxy.eval-estimation-single, "
            "agent_proxy.eval-estimation-multi, and "
            "agent_proxy.eval-estimation-toolcall, and "
            "agent_proxy.eval_adaptation_turn cannot enable more than one at the same time."
        )
    if enabled_modes:
        return enabled_modes[0]
    return None


def resolve_eval_adaptation_turn_scope(config) -> List[int]:
    raw_scope = agent_proxy_cfg_get(config, "eval_adaptation_turn_scope", [])
    scope = _normalize_eval_compliance_scope(
        raw_scope,
        config_key="eval_adaptation_turn_scope",
    )
    if scope and len(scope) != 3:
        raise ValueError(
            "agent_proxy.eval_adaptation_turn_scope must contain exactly three integers: "
            "[mutation_turn, budget_before_or_at_mutation_turn, budget_after_mutation_turn]."
        )
    return scope


def resolve_eval_adaptation_turn_config(config) -> Optional[tuple[int, int, int]]:
    enabled = bool(agent_proxy_cfg_get(config, "eval_adaptation_turn", False))
    scope = resolve_eval_adaptation_turn_scope(config)
    if not enabled:
        return None
    if not scope:
        raise ValueError(
            "agent_proxy.eval_adaptation_turn is enabled, but "
            "agent_proxy.eval_adaptation_turn_scope is empty."
        )
    mutation_turn, budget_before, budget_after = scope
    return int(mutation_turn), int(budget_before), int(budget_after)


def _iter_active_env_tags(config) -> List[str]:
    tags: List[str] = []
    es_cfg = getattr(config, "es_manager", None)
    for mode_name in ("train", "val"):
        mode_cfg = getattr(es_cfg, mode_name, None) if es_cfg is not None else None
        env_configs = getattr(mode_cfg, "env_configs", None) if mode_cfg is not None else None
        raw_tags = getattr(env_configs, "tags", None) if env_configs is not None else None
        if raw_tags is None:
            continue
        if hasattr(raw_tags, "tolist"):
            raw_tags = raw_tags.tolist()
        for tag in list(raw_tags):
            tag_str = str(tag)
            if tag_str not in tags:
                tags.append(tag_str)
    if tags:
        return tags

    custom_envs = getattr(config, "custom_envs", None)
    if custom_envs is None:
        return tags
    try:
        for tag in custom_envs.keys():
            tag_str = str(tag)
            if tag_str not in tags:
                tags.append(tag_str)
    except Exception:
        pass
    return tags


def resolve_toolcall_action_point_cap(config) -> Optional[int]:
    estimation_mode = resolve_eval_estimation_mode(config)
    compliance_mode = resolve_eval_compliance_mode(config)
    mode_key = None
    if estimation_mode == "toolcall":
        mode_key = "eval-estimation-toolcall"
    elif compliance_mode == "toolcall":
        mode_key = "eval_compliance_toolcall"

    if mode_key is None:
        return None

    if "robotouille" not in REGISTERED_ENV_CONFIGS:
        raise ValueError(
            f"agent_proxy.{mode_key} requires the robotouille environment to be installed."
        )

    active_tags = _iter_active_env_tags(config)
    if not active_tags:
        raise ValueError(
            f"agent_proxy.{mode_key} requires an active robotouille environment."
        )

    default_robotouille_cfg = REGISTERED_ENV_CONFIGS["robotouille"]()
    custom_envs = getattr(config, "custom_envs", None)
    caps: List[int] = []

    for tag in active_tags:
        custom_env = None
        if custom_envs is not None:
            if hasattr(custom_envs, "get"):
                custom_env = custom_envs.get(tag, None)
            else:
                custom_env = getattr(custom_envs, tag, None)
        env_type = str(env_cfg_get(custom_env, "env_type", "") or "").strip().lower()
        if env_type != "robotouille":
            raise ValueError(
                f"agent_proxy.{mode_key} can only be enabled when all active environments are robotouille."
            )

        raw_env_config = env_cfg_get(custom_env, "env_config", None)
        enable_action_budget = bool(
            env_cfg_get(
                raw_env_config,
                "enable_action_budget",
                getattr(default_robotouille_cfg, "enable_action_budget", False),
            )
        )
        if not enable_action_budget:
            raise ValueError(
                f"agent_proxy.{mode_key} requires custom_envs.{tag}.env_config.enable_action_budget=True."
            )

        raw_cap = env_cfg_get(
            raw_env_config,
            "max_action_points",
            getattr(default_robotouille_cfg, "max_action_points", None),
        )
        if raw_cap is None:
            raise ValueError(
                f"agent_proxy.{mode_key} requires custom_envs.{tag}.env_config.max_action_points."
            )
        try:
            cap = max(0, int(raw_cap))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"custom_envs.{tag}.env_config.max_action_points must be an integer, got {raw_cap!r}."
            ) from exc
        caps.append(cap)

    unique_caps = sorted(set(caps))
    if len(unique_caps) > 1:
        raise ValueError(
            f"agent_proxy.{mode_key} currently requires all active robotouille environments "
            "to share the same max_action_points."
        )
    return unique_caps[0]


def resolve_eval_compliance_mode(config):
    token_enabled = bool(agent_proxy_cfg_get(config, "eval_compliance_token", False))
    turn_enabled = bool(agent_proxy_cfg_get(config, "eval_compliance_turn", False))
    toolcall_enabled = bool(agent_proxy_cfg_get(config, "eval_compliance_toolcall", False))
    enabled_modes = [
        mode_name
        for mode_name, enabled in (
            ("token", token_enabled),
            ("turn", turn_enabled),
            ("toolcall", toolcall_enabled),
        )
        if enabled
    ]
    if len(enabled_modes) > 1:
        raise ValueError(
            "agent_proxy.eval_compliance_token, "
            "agent_proxy.eval_compliance_turn, and "
            "agent_proxy.eval_compliance_toolcall cannot enable more than one at the same time."
        )
    if enabled_modes:
        return enabled_modes[0]
    return None


def resolve_eval_compliance_token_scope(config) -> List[int]:
    raw_scope = agent_proxy_cfg_get(config, "eval_compliance_token_scope", [])
    return _normalize_eval_compliance_scope(
        raw_scope,
        config_key="eval_compliance_token_scope",
    )


def resolve_eval_compliance_turn_scope(config) -> List[int]:
    raw_scope = agent_proxy_cfg_get(config, "eval_compliance_turn_scope", [])
    return _normalize_eval_compliance_scope(
        raw_scope,
        config_key="eval_compliance_turn_scope",
    )


def resolve_eval_compliance_turn_mutation_turn(config) -> Optional[int]:
    raw_value = agent_proxy_cfg_get(config, "eval_compliance_turn_mutation_turn", None)
    if raw_value in (None, ""):
        return None
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "agent_proxy.eval_compliance_turn_mutation_turn must be an integer. "
            f"Got {raw_value!r}."
        ) from exc


def resolve_eval_compliance_turn_budget_change(config) -> List[int]:
    raw_value = agent_proxy_cfg_get(config, "eval_compliance_turn_budget_change", [])
    budget_change = _normalize_eval_compliance_scope(
        raw_value,
        config_key="eval_compliance_turn_budget_change",
    )
    if budget_change and len(budget_change) != 2:
        raise ValueError(
            "agent_proxy.eval_compliance_turn_budget_change must contain exactly two integers: "
            "[budget_before_or_at_mutation_turn, budget_after_mutation_turn]."
        )
    return budget_change


def resolve_eval_compliance_turn_mutation_config(config) -> Optional[tuple[int, List[int]]]:
    mutation_turn = resolve_eval_compliance_turn_mutation_turn(config)
    budget_change = resolve_eval_compliance_turn_budget_change(config)
    if mutation_turn is None and not budget_change:
        return None
    if mutation_turn is None or not budget_change:
        raise ValueError(
            "agent_proxy.eval_compliance_turn_mutation_turn and "
            "agent_proxy.eval_compliance_turn_budget_change must be set together."
        )
    return mutation_turn, budget_change


def resolve_eval_compliance_toolcall_scope(config) -> List[int]:
    raw_scope = agent_proxy_cfg_get(config, "eval_compliance_toolcall_scope", [])
    return _normalize_eval_compliance_scope(
        raw_scope,
        config_key="eval_compliance_toolcall_scope",
    )


def resolve_rollout_max_turn(config) -> int:
    return max(1, int(agent_proxy_cfg_get(config, "max_turn", 1) or 1))


def expand_compliance_group_size(config) -> None:
    compliance_mode = resolve_eval_compliance_mode(config)
    if compliance_mode is None:
        return

    if bool(agent_proxy_cfg_get(config, "eval_compliance_group_size_expanded", False)):
        return

    if compliance_mode == "turn":
        scope = resolve_eval_compliance_turn_scope(config)
        scope_key = "eval_compliance_turn_scope"
        mode_key = "eval_compliance_turn"
        mutation_cfg = resolve_eval_compliance_turn_mutation_config(config)
        if scope and mutation_cfg is not None:
            raise ValueError(
                "agent_proxy.eval_compliance_turn_scope cannot be used together with "
                "agent_proxy.eval_compliance_turn_mutation_turn / "
                "agent_proxy.eval_compliance_turn_budget_change."
            )
        if mutation_cfg is not None:
            scope = [0]
    elif compliance_mode == "toolcall":
        scope = resolve_eval_compliance_toolcall_scope(config)
        scope_key = "eval_compliance_toolcall_scope"
        mode_key = "eval_compliance_toolcall"
    else:
        scope = resolve_eval_compliance_token_scope(config)
        scope_key = "eval_compliance_token_scope"
        mode_key = "eval_compliance_token"
    if not scope:
        raise ValueError(
            f"agent_proxy.{mode_key} is enabled, but "
            f"agent_proxy.{scope_key} is empty."
        )

    factor = len(scope)
    es_cfg = getattr(config, "es_manager", None)
    if es_cfg is not None:
        for mode_name in ("train", "val"):
            mode_cfg = getattr(es_cfg, mode_name, None)
            if mode_cfg is None or getattr(mode_cfg, "group_size", None) is None:
                continue
            with open_dict(mode_cfg):
                mode_cfg.group_size = int(mode_cfg.group_size) * factor

    agent_cfg = getattr(config, "agent_proxy", None)
    if agent_cfg is not None:
        with open_dict(agent_cfg):
            agent_cfg.eval_compliance_group_size_expanded = True
