from typing import Any, Optional


def _get_value(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "get"):
        try:
            value = obj.get(key, None)
        except TypeError:
            value = None
        if value is not None:
            return value
    return getattr(obj, key, None)


def get_registered_model_info(config: Any) -> Optional[Any]:
    model_cfg = getattr(config, "model_config", None)
    model_key = _get_value(model_cfg, "model_name")
    model_registry = getattr(config, "model_info", None)
    if model_key is None or model_registry is None:
        return None
    try:
        return model_registry[model_key]
    except Exception:
        if hasattr(model_registry, "get"):
            return model_registry.get(model_key, None)
        return None


def get_registered_provider_name(config: Any) -> Optional[str]:
    model_info = get_registered_model_info(config)
    provider_name = _get_value(model_info, "provider_name")
    if provider_name is None:
        return None
    return str(provider_name).strip().lower()


def get_registered_backend_model_name(config: Any) -> Optional[str]:
    model_info = get_registered_model_info(config)
    model_name = _get_value(model_info, "model_name")
    if model_name is None:
        return None
    return str(model_name).strip()


def is_openai_reasoning_model_name(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    normalized = str(model_name).strip().lower()
    return normalized.startswith("gpt-5") or normalized.startswith("o")


def uses_openai_reasoning_model(config: Any) -> bool:
    provider_name = get_registered_provider_name(config)
    model_name = get_registered_backend_model_name(config)
    return provider_name == "openai" and is_openai_reasoning_model_name(model_name)


def should_avoid_explicit_reasoning_output(config: Any) -> bool:
    return uses_openai_reasoning_model(config)
