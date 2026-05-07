#!/usr/bin/env python
import argparse
import json
import os
import random
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


def _parse_bool_flag(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


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


def _build_temp_record(env: TokenEstimationEnv, sample: Any) -> Dict[str, Any]:
    api_messages = env.build_api_messages(sample)
    record = {
        "sample_id": sample.sample_id,
        "rollout_index": sample.rollout_index,
        "turn_idx": sample.turn_idx,
        "input_messages": api_messages,
        "rollout_history_messages": sample.input_messages,
        "input_text": json.dumps(api_messages, ensure_ascii=False, indent=2),
        "estimation_user_prompt": env.build_user_prompt(sample),
        "output": sample.target_output,
        "source_system": sample.source_system,
        "completed_turns": sample.completed_turns,
        "total_turns": sample.total_turns,
        "relative_progress": sample.relative_progress,
        "completed_turn_token_usage": sample.completed_turn_token_usage,
        "completed_turn_token_usage_details": sample.completed_turn_token_usage_details,
        "actual_tokens_used_so_far": sample.actual_tokens_used_so_far,
        "actual_can_finish": sample.actual_can_finish,
        "actual_remaining_total_tokens": sample.actual_remaining_total_tokens,
    }
    if sample.completed_turn_request_token_usage is not None:
        record["completed_turn_request_token_usage"] = sample.completed_turn_request_token_usage
    if sample.completed_turn_request_token_usage_details is not None:
        record["completed_turn_request_token_usage_details"] = (
            sample.completed_turn_request_token_usage_details
        )
    return record


def _export_temp_pairs(
    env: TokenEstimationEnv,
    path: str,
    sample_indices: List[int],
) -> str:
    _ensure_parent_dir(path)
    payload = []
    for sample_index in sample_indices:
        sample = env.get_sample(int(sample_index))
        payload.append(_build_temp_record(env, sample))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def _build_rollout_length_buckets(
    samples: List[Any],
    num_bins: int,
) -> tuple[Dict[int, int], List[Dict[str, Any]]]:
    rollout_to_total_turns: Dict[int, int] = {}
    for sample in samples:
        rollout_index = int(sample.rollout_index)
        rollout_to_total_turns.setdefault(rollout_index, int(sample.total_turns))

    sorted_rollouts = sorted(
        rollout_to_total_turns.items(),
        key=lambda item: (int(item[1]), int(item[0])),
    )
    if not sorted_rollouts:
        return {}, []

    normalized_num_bins = max(1, min(int(num_bins), len(sorted_rollouts)))
    base_size, remainder = divmod(len(sorted_rollouts), normalized_num_bins)

    rollout_to_bucket: Dict[int, int] = {}
    bucket_stats: List[Dict[str, Any]] = []
    start = 0
    for bucket_id in range(normalized_num_bins):
        bucket_size = base_size + (1 if bucket_id < remainder else 0)
        if bucket_size <= 0:
            continue
        chunk = sorted_rollouts[start:start + bucket_size]
        start += bucket_size
        if not chunk:
            continue
        rollout_indices = [int(rollout_index) for rollout_index, _ in chunk]
        for rollout_index in rollout_indices:
            rollout_to_bucket[rollout_index] = int(bucket_id)
        bucket_stats.append(
            {
                "bucket_id": int(bucket_id),
                "rollout_count": len(chunk),
                "min_total_turns": int(chunk[0][1]),
                "max_total_turns": int(chunk[-1][1]),
                "rollout_indices": rollout_indices,
            }
        )
    return rollout_to_bucket, bucket_stats


def _select_fair_split_random_indices(
    samples: List[Any],
    max_samples: int,
    seed: int,
    length_bins: int,
) -> tuple[List[int], List[Dict[str, Any]]]:
    if max_samples <= 0 or not samples:
        return [], []

    rng = random.Random(int(seed))
    rollout_to_indices: Dict[int, List[int]] = {}
    for sample_index, sample in enumerate(samples):
        rollout_to_indices.setdefault(int(sample.rollout_index), []).append(int(sample_index))

    for indices in rollout_to_indices.values():
        rng.shuffle(indices)

    rollout_to_bucket, bucket_stats = _build_rollout_length_buckets(samples, num_bins=length_bins)
    bucket_to_rollouts: Dict[int, List[int]] = {}
    for rollout_index in rollout_to_indices:
        bucket_id = int(rollout_to_bucket.get(int(rollout_index), 0))
        bucket_to_rollouts.setdefault(bucket_id, []).append(int(rollout_index))
    for rollout_indices in bucket_to_rollouts.values():
        rng.shuffle(rollout_indices)

    rollout_cursors = {int(rollout_index): 0 for rollout_index in rollout_to_indices}
    bucket_pointers = {int(bucket_id): 0 for bucket_id in bucket_to_rollouts}
    selected_indices: List[int] = []
    active_bucket_ids = list(bucket_to_rollouts)

    while len(selected_indices) < int(max_samples) and active_bucket_ids:
        current_bucket_ids = list(active_bucket_ids)
        rng.shuffle(current_bucket_ids)
        progressed = False
        next_active_bucket_ids: List[int] = []

        for bucket_id in current_bucket_ids:
            rollout_indices = bucket_to_rollouts.get(int(bucket_id), [])
            if not rollout_indices:
                continue

            selected_in_bucket = False
            rollout_count = len(rollout_indices)
            for _ in range(rollout_count):
                pointer = int(bucket_pointers[int(bucket_id)] % rollout_count)
                rollout_index = int(rollout_indices[pointer])
                bucket_pointers[int(bucket_id)] = int((pointer + 1) % rollout_count)
                rollout_cursor = int(rollout_cursors[rollout_index])
                rollout_sample_indices = rollout_to_indices[rollout_index]
                if rollout_cursor >= len(rollout_sample_indices):
                    continue
                selected_indices.append(int(rollout_sample_indices[rollout_cursor]))
                rollout_cursors[rollout_index] = int(rollout_cursor + 1)
                progressed = True
                selected_in_bucket = True
                break

            bucket_has_remaining = any(
                int(rollout_cursors[int(rollout_index)]) < len(rollout_to_indices[int(rollout_index)])
                for rollout_index in rollout_indices
            )
            if bucket_has_remaining:
                next_active_bucket_ids.append(int(bucket_id))
            if len(selected_indices) >= int(max_samples):
                break
            if not selected_in_bucket and not bucket_has_remaining:
                continue

        active_bucket_ids = next_active_bucket_ids
        if not progressed:
            break

    if len(selected_indices) < int(max_samples):
        selected_set = set(int(index) for index in selected_indices)
        remaining_indices = [
            int(index)
            for index in range(len(samples))
            if int(index) not in selected_set
        ]
        rng.shuffle(remaining_indices)
        selected_indices.extend(remaining_indices[: max(0, int(max_samples) - len(selected_indices))])

    return selected_indices[: int(max_samples)], bucket_stats


def _select_sample_indices(
    samples: List[Any],
    *,
    max_samples: Optional[int],
    strategy: str,
    seed: int,
    length_bins: int,
    cache_friendly_order: bool,
) -> tuple[List[int], Dict[str, Any]]:
    total_available_samples = len(samples)
    requested_samples = total_available_samples if max_samples is None else max(0, int(max_samples))
    capped_samples = min(int(requested_samples), int(total_available_samples))
    selection_metadata: Dict[str, Any] = {
        "strategy": str(strategy),
        "seed": int(seed),
        "requested_samples": int(requested_samples),
        "total_available_samples": int(total_available_samples),
        "cache_friendly_order": bool(cache_friendly_order),
    }

    if capped_samples <= 0:
        return [], selection_metadata

    if capped_samples >= total_available_samples:
        selected_indices = list(range(total_available_samples))
    elif strategy == "first":
        selected_indices = list(range(capped_samples))
    elif strategy == "random":
        rng = random.Random(int(seed))
        selected_indices = list(range(total_available_samples))
        rng.shuffle(selected_indices)
        selected_indices = selected_indices[:capped_samples]
    elif strategy == "fair_split_random":
        selected_indices, bucket_stats = _select_fair_split_random_indices(
            samples,
            max_samples=capped_samples,
            seed=seed,
            length_bins=length_bins,
        )
        selection_metadata["length_buckets"] = bucket_stats
    else:
        raise ValueError(f"Unsupported sample selection strategy: {strategy}")

    if cache_friendly_order:
        rollout_groups: Dict[int, List[int]] = {}
        for sample_index in selected_indices:
            rollout_groups.setdefault(
                int(samples[int(sample_index)].rollout_index),
                [],
            ).append(int(sample_index))

        normalized_groups: List[tuple[tuple[int, int, int, int], List[int]]] = []
        for rollout_index, group_indices in rollout_groups.items():
            ordered_group_indices = sorted(
                group_indices,
                key=lambda sample_index: (
                    int(getattr(samples[int(sample_index)], "actual_tokens_used_so_far", 0) or 0),
                    int(samples[int(sample_index)].turn_idx),
                    int(sample_index),
                ),
            )
            first_sample = samples[int(ordered_group_indices[0])]
            first_history_tokens = int(getattr(first_sample, "actual_tokens_used_so_far", 0) or 0)
            first_turn_idx = int(first_sample.turn_idx)
            normalized_groups.append(
                (
                    (
                        first_history_tokens,
                        first_turn_idx,
                        len(ordered_group_indices),
                        int(rollout_index),
                    ),
                    ordered_group_indices,
                )
            )

        normalized_groups.sort(key=lambda item: item[0])
        selected_indices = [
            sample_index
            for _, ordered_group_indices in normalized_groups
            for sample_index in ordered_group_indices
        ]
        selection_metadata["cache_order_strategy"] = "group_by_rollout_short_history_first"

    selection_metadata["selected_samples"] = len(selected_indices)
    selection_metadata["selected_rollouts"] = len(
        {int(samples[int(sample_index)].rollout_index) for sample_index in selected_indices}
    )
    return selected_indices, selection_metadata


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
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap after flattening")
    parser.add_argument(
        "--sample-selection",
        choices=["first", "random", "fair_split_random"],
        default="first",
        help="How to choose samples when --max-samples is smaller than the flattened split-rollout count",
    )
    parser.add_argument("--sample-selection-seed", type=int, default=0, help="Random seed for sample selection")
    parser.add_argument(
        "--sample-length-bins",
        type=int,
        default=8,
        help="Number of rollout-length buckets for fair_split_random selection",
    )
    parser.add_argument(
        "--cache-friendly-order",
        action="store_true",
        help="Execute selected samples grouped by rollout and turn to maximize prompt-cache reuse",
    )
    parser.add_argument("--max-turn", type=int, default=5, help="Legacy rollout turn budget retained for compatibility")
    parser.add_argument("--max-context-window-tokens", type=int, default=81920, help="Total token budget used for can-finish evaluation")
    parser.add_argument(
        "--turn-usage-mode",
        choices=["request", "turn_excluding_history"],
        default="turn_excluding_history",
        help="How to interpret per-turn usage exposed to the estimator; default excludes replayed history",
    )
    parser.add_argument("--reasoning-effort", default=None, help="Optional reasoning_effort passed through to compatible models")
    parser.add_argument("--reasoning-enabled", type=_parse_bool_flag, default=None, help="Optional OpenRouter reasoning.enabled override")
    parser.add_argument("--thinking-enabled", action="store_true", help="Enable Anthropic extended thinking")
    parser.add_argument("--thinking-budget-tokens", type=int, default=None, help="Anthropic thinking budget_tokens")
    parser.add_argument("--thinking-adaptive", action="store_true", help="Use Anthropic adaptive thinking")
    parser.add_argument("--thinking-display", default=None, help="Anthropic adaptive thinking display mode")
    parser.add_argument("--output-effort", default=None, help="Anthropic output_config.effort")
    parser.add_argument("--cache-enabled", action="store_true", help="Attach OpenRouter cache_control={type:ephemeral}")
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
        max_instances=None,
        include_source_system=not bool(args.disable_source_system),
        system_prompt_template=system_prompt_template or DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        user_prompt_template=user_prompt_template or DEFAULT_USER_PROMPT_TEMPLATE,
        turn_usage_mode=str(args.turn_usage_mode),
    )
    env = TokenEstimationEnv(env_config)
    sample_indices, selection_metadata = _select_sample_indices(
        env.samples,
        max_samples=args.max_samples,
        strategy=str(args.sample_selection),
        seed=int(args.sample_selection_seed),
        length_bins=max(1, int(args.sample_length_bins)),
        cache_friendly_order=bool(args.cache_friendly_order),
    )

    temp_json_path = args.temp_json
    if not temp_json_path:
        input_stem = os.path.splitext(os.path.basename(args.input_json))[0]
        temp_json_path = os.path.join(
            os.path.dirname(args.output_json) or ".",
            f"{input_stem}_turn_pairs.json",
        )
    _export_temp_pairs(env, temp_json_path, sample_indices)

    if args.dry_run:
        payload = {
            "config": vars(args),
            "selection": selection_metadata,
            "temp_json_path": temp_json_path,
            "summary": {
                "total_samples": len(sample_indices),
                "total_available_samples": len(env.samples),
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
    }
    if args.temperature is not None:
        generate_kwargs["temperature"] = float(args.temperature)
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
    if provider_lower == "openrouter":
        openrouter_extra_body: Dict[str, Any] = {}
        openrouter_reasoning: Dict[str, Any] = {}
        if args.reasoning_effort:
            openrouter_reasoning["effort"] = str(args.reasoning_effort)
        if args.reasoning_enabled is not None:
            openrouter_reasoning["enabled"] = bool(args.reasoning_enabled)
        if openrouter_reasoning:
            openrouter_extra_body["reasoning"] = openrouter_reasoning
        if args.cache_enabled:
            openrouter_extra_body["cache_control"] = {"type": "ephemeral"}
        if openrouter_extra_body:
            generate_kwargs["extra_body"] = openrouter_extra_body
    elif args.reasoning_effort:
        generate_kwargs["reasoning_effort"] = str(args.reasoning_effort)

    results: List[Dict[str, Any]] = []
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

            # print("[DEBUG] Messages List:", messages_list)
            batch_results, _ = llm.run_batch(
                messages_list=messages_list,
                **generate_kwargs,
            )
            for (sample_index, sample), api_result in zip(indexed_samples, batch_results):
                env.reset(index=sample_index)
                api_messages = env.build_api_messages(sample)
                _, _, _, info = env.step(api_result.get("response", ""))
                ground_truth = {
                    "actual_tokens_used_so_far": sample.actual_tokens_used_so_far,
                    "actual_can_finish": sample.actual_can_finish,
                    "actual_remaining_total_tokens": sample.actual_remaining_total_tokens,
                    "relative_progress": sample.relative_progress,
                    "completed_turns": sample.completed_turns,
                    "total_turns": sample.total_turns,
                    "completed_turn_token_usage": sample.completed_turn_token_usage,
                    "completed_turn_token_usage_details": sample.completed_turn_token_usage_details,
                    "rollout_success": sample.rollout_success,
                }
                if sample.completed_turn_request_token_usage is not None:
                    ground_truth["completed_turn_request_token_usage"] = (
                        sample.completed_turn_request_token_usage
                    )
                if sample.completed_turn_request_token_usage_details is not None:
                    ground_truth["completed_turn_request_token_usage_details"] = (
                        sample.completed_turn_request_token_usage_details
                    )
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
                        "ground_truth": ground_truth,
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
        "selection": selection_metadata,
        "temp_json_path": temp_json_path,
        "summary": _build_summary(results),
        "results": results,
    }
    _ensure_parent_dir(args.output_json)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
