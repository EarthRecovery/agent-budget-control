#!/usr/bin/env python
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
from tqdm.auto import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
VERL_ROOT = os.path.join(REPO_ROOT, "verl")
if VERL_ROOT not in sys.path:
    sys.path.insert(0, VERL_ROOT)

from ragen.env.token_estimation import TokenEstimationEnv, TokenEstimationEnvConfig
from ragen.env.token_estimation.config import (
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    DEFAULT_USER_PROMPT_TEMPLATE,
)
from ragen.llm_agent.base_llm import ConcurrentLLM


def _default_api_key_env(provider: str) -> str:
    provider = provider.strip().lower()
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    if provider == "gemini":
        return "GEMINI_API_KEY"
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    if provider == "together":
        return "TOGETHER_API_KEY"
    if provider == "deepseek":
        return "DEEPSEEK_API_KEY"
    return "OPENAI_API_KEY"


def _load_optional_text(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _safe_mean(values: List[float]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _safe_rate(values: List[bool]) -> Optional[float]:
    filtered = [bool(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(1 for value in filtered if value) / len(filtered)


def _chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    if chunk_size <= 0:
        return [items]
    return [items[idx: idx + chunk_size] for idx in range(0, len(items), chunk_size)]


def _build_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    api_success_flags = [
        bool(record.get("api_result", {}).get("success", False))
        for record in records
    ]
    can_finish_correct = [
        record.get("metrics", {}).get("can_finish_correct")
        for record in records
    ]
    remaining_token_interval_coverage = [
        record.get("metrics", {}).get("remaining_token_interval_contains_actual")
        for record in records
    ]
    api_total_tokens = [
        record.get("api_result", {}).get("usage", {}).get("total_tokens")
        for record in records
        if isinstance(record.get("api_result", {}).get("usage"), dict)
    ]

    return {
        "total_samples": len(records),
        "api_success_rate": _safe_rate(api_success_flags),
        "can_finish_accuracy": _safe_rate(can_finish_correct),
        "remaining_token_interval_coverage_rate": _safe_rate(remaining_token_interval_coverage),
        "api_total_tokens_sum": sum(int(value) for value in api_total_tokens if value is not None),
    }


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline token estimation over *_eval_estimation_dialogues.json files."
    )
    parser.add_argument("--input-json", required=True, help="Path to *_eval_estimation_dialogues.json")
    parser.add_argument("--output-json", required=True, help="Path to aggregated result json")
    parser.add_argument("--temp-json", default=None, help="Path to exported temp input/output pairs json")
    parser.add_argument("--provider", default="openai", help="LLM provider: openai/anthropic/gemini/openrouter/together/deepseek")
    parser.add_argument("--model", default="gpt-5.2-2025-12-11", help="Backend model name")
    parser.add_argument("--api-key-env", default=None, help="Environment variable name for API key")
    parser.add_argument("--max-concurrency", type=int, default=8, help="Concurrent requests")
    parser.add_argument("--request-batch-size", type=int, default=64, help="Batch size passed to ConcurrentLLM")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max completion tokens per API call")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap after flattening")
    parser.add_argument("--max-turn", type=int, default=5, help="Legacy rollout turn budget retained for compatibility")
    parser.add_argument("--max-context-window-tokens", type=int, default=81920, help="Total token budget used for can-finish evaluation")
    parser.add_argument("--reasoning-effort", default=None, help="Optional reasoning_effort passed through to compatible models")
    parser.add_argument("--thinking-enabled", action="store_true", help="Enable Anthropic extended thinking")
    parser.add_argument("--thinking-budget-tokens", type=int, default=None, help="Anthropic thinking budget_tokens")
    parser.add_argument("--thinking-adaptive", action="store_true", help="Use Anthropic adaptive thinking")
    parser.add_argument("--thinking-display", default=None, help="Anthropic adaptive thinking display mode")
    parser.add_argument("--output-effort", default=None, help="Anthropic output_config.effort")
    parser.add_argument("--system-prompt-file", default=None, help="Optional txt file to override system prompt template")
    parser.add_argument("--user-prompt-file", default=None, help="Optional txt file to override user prompt template")
    parser.add_argument("--disable-source-system", action="store_true", help="Do not include original rollout system in user prompt")
    parser.add_argument("--dry-run", action="store_true", help="Only export temp json without calling the API")
    args = parser.parse_args()

    system_prompt_template = _load_optional_text(args.system_prompt_file)
    user_prompt_template = _load_optional_text(args.user_prompt_file)
    env_config = TokenEstimationEnvConfig(
        input_path=args.input_json,
        max_context_window_tokens=int(args.max_context_window_tokens),
        max_instances=args.max_samples,
        include_source_system=not bool(args.disable_source_system),
        system_prompt_template=system_prompt_template or DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        user_prompt_template=user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE,
    )
    env = TokenEstimationEnv(env_config)

    temp_json_path = args.temp_json
    if not temp_json_path:
        input_stem = os.path.splitext(os.path.basename(args.input_json))[0]
        temp_json_path = os.path.join(
            os.path.dirname(args.output_json) or ".",
            f"{input_stem}_turn_pairs.json",
        )
    _ensure_parent_dir(temp_json_path)
    env.export_temp_pairs(temp_json_path)

    if args.dry_run:
        payload = {
            "config": vars(args),
            "temp_json_path": temp_json_path,
            "summary": {
                "total_samples": len(env.samples),
            },
            "results": [],
        }
        _ensure_parent_dir(args.output_json)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return

    api_key_env = args.api_key_env or _default_api_key_env(args.provider)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise EnvironmentError(f"Missing API key env var: {api_key_env}")

    llm = ConcurrentLLM(
        provider=args.provider,
        model_name=args.model,
        api_key=api_key,
        max_concurrency=int(args.max_concurrency),
    )
    generate_kwargs: Dict[str, Any] = {
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
    }
    model_name_lower = str(args.model).lower()
    provider_lower = str(args.provider).lower()
    if provider_lower == "anthropic" and (
        model_name_lower.startswith("claude-opus-4") or model_name_lower.startswith("claude-sonnet-4")
    ):
        generate_kwargs.pop("temperature", None)
        if args.thinking_adaptive:
            thinking_kwargs: Dict[str, Any] = {"type": "adaptive"}
            if args.thinking_display:
                thinking_kwargs["display"] = str(args.thinking_display)
            generate_kwargs["thinking"] = thinking_kwargs
        elif args.thinking_enabled:
            thinking_kwargs: Dict[str, Any] = {"type": "enabled"}
            if args.thinking_budget_tokens is not None:
                thinking_kwargs["budget_tokens"] = int(args.thinking_budget_tokens)
            generate_kwargs["thinking"] = thinking_kwargs
        if args.output_effort:
            generate_kwargs["output_config"] = {"effort": str(args.output_effort)}
    if args.reasoning_effort:
        generate_kwargs["reasoning_effort"] = str(args.reasoning_effort)

    results: List[Dict[str, Any]] = []
    sample_indices = list(range(len(env.samples)))
    progress_bar = tqdm(
        total=len(sample_indices),
        desc="Token Estimation Eval",
        unit="sample",
        dynamic_ncols=True,
    )
    try:
        for index_chunk in _chunk_list(sample_indices, int(args.request_batch_size)):
            messages_list = []
            indexed_samples = []
            for sample_index in index_chunk:
                sample = env.get_sample(sample_index)
                indexed_samples.append((sample_index, sample))
                messages_list.append(env.build_api_messages(sample))

            print("[DEBUG] Messages List:", messages_list)
            batch_results, _ = llm.run_batch(
                messages_list=messages_list,
                **generate_kwargs,
            )
            for (sample_index, sample), api_result in zip(indexed_samples, batch_results):
                env.reset(index=sample_index)
                api_messages = env.build_api_messages(sample)
                _, _, _, info = env.step(api_result.get("response", ""))
                results.append(
                    {
                        "sample_id": sample.sample_id,
                        "rollout_index": sample.rollout_index,
                        "turn_idx": sample.turn_idx,
                        "source_system": sample.source_system,
                        "input_messages": api_messages,
                        "rollout_history_messages": sample.input_messages,
                        "input_text": json.dumps(api_messages, ensure_ascii=False, indent=2),
                        "estimation_user_prompt": env.build_user_prompt(sample),
                        "target_output": sample.target_output,
                        "ground_truth": {
                            "actual_tokens_used_so_far": sample.actual_tokens_used_so_far,
                            "actual_can_finish": sample.actual_can_finish,
                            "actual_remaining_total_tokens": sample.actual_remaining_total_tokens,
                            "relative_progress": sample.relative_progress,
                            "completed_turns": sample.completed_turns,
                            "total_turns": sample.total_turns,
                            "rollout_success": sample.rollout_success,
                        },
                        "prediction": info.get("prediction"),
                        "metrics": info.get("metrics"),
                        "api_result": {
                            "success": api_result.get("success"),
                            "provider": api_result.get("provider"),
                            "model": api_result.get("model"),
                            "usage": api_result.get("usage"),
                            "error": api_result.get("error"),
                            "error_type": api_result.get("error_type"),
                            "error_code": api_result.get("error_code"),
                            "status_code": api_result.get("status_code"),
                            "request_id": api_result.get("request_id"),
                            "attempts": api_result.get("attempts"),
                        },
                    }
                )
            progress_bar.update(len(index_chunk))
    finally:
        progress_bar.close()

    payload = {
        "config": {
            **vars(args),
            "api_key_env": api_key_env,
        },
        "temp_json_path": temp_json_path,
        "summary": _build_summary(results),
        "results": results,
    }
    _ensure_parent_dir(args.output_json)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
