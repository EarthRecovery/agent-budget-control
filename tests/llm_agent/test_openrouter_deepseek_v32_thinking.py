import json
import os
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "evaluate_api_llm.yaml"


def _load_model_registry() -> Dict[str, Dict[str, Any]]:
    config = OmegaConf.load(CONFIG_PATH)
    model_info = OmegaConf.to_container(config.model_info, resolve=True)
    return model_info if isinstance(model_info, dict) else {}


def _load_model_config(model_key: str) -> Dict[str, Any]:
    registry = _load_model_registry()
    model_info = registry.get(model_key, {})
    generation_kwargs = dict(model_info.get("generation_kwargs") or {})
    extra_body = dict(generation_kwargs.get("extra_body") or {})
    return {
        "model_key": model_key,
        "provider_name": model_info.get("provider_name"),
        "model_name": model_info.get("model_name"),
        "generation_kwargs": generation_kwargs,
        "extra_body": extra_body,
    }


def _build_sokoban_messages() -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You're a helpful assistant. You are solving the Sokoban puzzle. "
                        "Push all boxes to targets. You are given the grid and zero-indexed "
                        "coordinates of the player, boxes, and targets. You can push but not "
                        "pull boxes, and cannot push a box through a wall.\n"
                        "Your available actions are:\n"
                        "Up, Down, Left, Right\n"
                        'You may output at most 3 action(s) in a single turn, separated by the '
                        'action separator " || ".'
                    ),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "user",
            "content": (
                "\nState:\n"
                "Coordinates: \n"
                "Board size: 8 rows x 8 cols (zero-indexed).\n"
                "Targets: (2, 3), (5, 1)\n"
                "Boxes: (3, 4), (4, 4)\n"
                "Player: (4, 5)\n"
                "Grid Map: \n"
                "########\n"
                "#_____##\n"
                "#__O__##\n"
                "#__#X_##\n"
                "#___XP##\n"
                "#O___###\n"
                "#_##_###\n"
                "########\n"
                "You may output at most 3 action(s) in this turn. Always output: "
                "<think> [Your thoughts, typically 2 sentences,less than 100 tokens] </think> <answer> [your answer] </answer> "
                "with no extra text. Strictly follow this format. Max response "
                "length: 800 words (tokens).\n"
            ),
        },
    ]



def _build_openrouter_request_body(model_key: str) -> Dict[str, Any]:
    model_config = _load_model_config(model_key)
    generation_kwargs = dict(model_config["generation_kwargs"])
    extra_body = dict(generation_kwargs.pop("extra_body", {}) or {})
    max_tokens = int(os.environ.get("OPENROUTER_LIVE_MAX_TOKENS", generation_kwargs.get("max_tokens", 300)))
    generation_kwargs["max_tokens"] = min(max_tokens, int(generation_kwargs.get("max_tokens", max_tokens)))
    return {
        "model": model_config["model_name"],
        "messages": _build_sokoban_messages(),
        "reasoning": {
            "effort": "low"
        },
        **generation_kwargs,
        **extra_body,
    }


def test_print_openrouter_live_output_capabilities(model_key: str):
    if model_key == "<none>":
        print("No requested live-report models found in config/evaluate_api_llm.yaml")
        return

    request_body = _build_openrouter_request_body(model_key)
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    }

    print(request_body)

    try:
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=request_body,
            timeout=float(os.environ.get("OPENROUTER_LIVE_TIMEOUT_SECONDS", "60")),
        )
        try:
            payload = response.json()
        except Exception:
            payload = {"raw_text": response.text}
    except Exception as exc:
        payload = {"request_exception": type(exc).__name__, "message": str(exc)}
        response = type("ResponseLike", (), {"status_code": None})()

    print("status_code:", response.status_code)
    print("headers:", dict(response.headers))
    print("payload:", json.dumps(payload, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_print_openrouter_live_output_capabilities("OpenRouter-Gemini-2.5-Pro")
    


# Example:
# export OPENROUTER_API_KEY=...
# export RUN_OPENROUTER_LIVE_TESTS=1
# conda run -n ragenv2 pytest -q -s /u/ylin30/agent-budget-control/tests/llm_agent/test_openrouter_deepseek_v32_thinking.py
