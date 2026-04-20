#!/usr/bin/env python
import argparse
import json
import os
import shutil
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


def _find_nvcc() -> Optional[str]:
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        cuda_home_nvcc = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.isfile(cuda_home_nvcc) and os.access(cuda_home_nvcc, os.X_OK):
            return cuda_home_nvcc
    return shutil.which("nvcc")


def _configure_vllm_sampler() -> None:
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") is not None:
        return
    if _find_nvcc():
        return
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"


def _render_messages_with_chat_template(tokenizer: Any, messages: List[Dict[str, Any]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        rendered_parts: List[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip().title()
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        text_parts.append(str(block.get("text", "")))
                    else:
                        text_parts.append(str(block))
                content = "".join(text_parts)
            rendered_parts.append(f"{role}:\n{content}".rstrip())
        rendered_parts.append("Assistant:\n")
        return "\n\n".join(rendered_parts)


def _count_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _trim_messages_to_fit(
    tokenizer: Any,
    messages: List[Dict[str, Any]],
    *,
    max_input_tokens: int,
) -> tuple[List[Dict[str, Any]], str, Dict[str, int]]:
    working_messages = [dict(message) for message in messages]
    rendered = _render_messages_with_chat_template(tokenizer, working_messages)
    rendered_tokens = _count_tokens(tokenizer, rendered)

    trim_info = {
        "max_input_tokens": int(max_input_tokens),
        "rendered_input_tokens": int(rendered_tokens),
        "dropped_messages": 0,
    }
    if rendered_tokens <= int(max_input_tokens):
        return working_messages, rendered, trim_info

    removable_indices = list(range(1, max(1, len(working_messages) - 1)))
    while rendered_tokens > int(max_input_tokens) and removable_indices:
        drop_index = removable_indices.pop(0)
        if drop_index >= len(working_messages) - 1:
            continue
        del working_messages[drop_index]
        removable_indices = [
            idx - 1 if idx > drop_index else idx
            for idx in removable_indices
            if idx != drop_index
        ]
        trim_info["dropped_messages"] += 1
        rendered = _render_messages_with_chat_template(tokenizer, working_messages)
        rendered_tokens = _count_tokens(tokenizer, rendered)

    trim_info["rendered_input_tokens"] = int(rendered_tokens)
    return working_messages, rendered, trim_info


def _usage_from_text(tokenizer: Any, prompt_text: str, response_text: str) -> Dict[str, Any]:
    prompt_tokens = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    completion_tokens = len(tokenizer(response_text, add_special_tokens=False)["input_ids"])
    total_tokens = prompt_tokens + completion_tokens
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "raw": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline token estimation over *_eval_estimation_dialogues.json files using a local vLLM model."
    )
    parser.add_argument("--input-json", required=True, help="Path to *_eval_estimation_dialogues.json")
    parser.add_argument("--output-json", required=True, help="Path to aggregated result json")
    parser.add_argument("--temp-json", default=None, help="Path to exported temp input/output pairs json")
    parser.add_argument("--model-path", required=True, help="Local Hugging Face model directory for vLLM")
    parser.add_argument("--model-tag", default=None, help="Optional display name recorded in the output")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap after flattening")
    parser.add_argument("--request-batch-size", type=int, default=32, help="Batch size passed to vLLM.generate")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max completion tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top-p")
    parser.add_argument("--top-k", type=int, default=-1, help="Sampling top-k")
    parser.add_argument("--max-turn", type=int, default=5, help="Legacy rollout turn budget retained for compatibility")
    parser.add_argument("--max-context-window-tokens", type=int, default=81920, help="Total token budget used for can-finish evaluation")
    parser.add_argument("--system-prompt-file", default=None, help="Optional txt file to override system prompt template")
    parser.add_argument("--user-prompt-file", default=None, help="Optional txt file to override user prompt template")
    parser.add_argument("--disable-source-system", action="store_true", help="Do not include original rollout system in user prompt")
    parser.add_argument("--dry-run", action="store_true", help="Only export temp json without loading vLLM")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor_parallel_size")
    parser.add_argument("--dtype", default="bfloat16", help="vLLM dtype")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    parser.add_argument("--max-model-len", type=int, default=32768, help="vLLM max_model_len")
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768, help="vLLM max_num_batched_tokens")
    parser.add_argument("--max-input-tokens", type=int, default=None, help="Maximum prompt tokens sent to the estimation model; defaults to max_model_len - max_tokens")
    parser.add_argument("--trust-remote-code", type=_parse_bool_flag, default=True, help="Pass trust_remote_code to vLLM/tokenizer")
    parser.add_argument("--vllm-use-v1", type=int, choices=[0, 1], default=1, help="Set VLLM_USE_V1 before importing vLLM")
    parser.add_argument("--worker-multiproc-method", choices=["fork", "spawn"], default="spawn", help="Set VLLM_WORKER_MULTIPROC_METHOD before importing vLLM")
    parser.add_argument("--enforce-eager", type=_parse_bool_flag, default=True, help="Pass enforce_eager to vLLM")
    parser.add_argument("--enable-sleep-mode", type=_parse_bool_flag, default=False, help="Pass enable_sleep_mode to vLLM")
    parser.add_argument("--enable-prefix-caching", type=_parse_bool_flag, default=True, help="Pass enable_prefix_caching to vLLM")
    parser.add_argument("--enable-chunked-prefill", type=_parse_bool_flag, default=False, help="Pass enable_chunked_prefill to vLLM")
    parser.add_argument("--disable-log-stats", type=_parse_bool_flag, default=True, help="Pass disable_log_stats to vLLM")
    parser.add_argument("--disable-custom-all-reduce", type=_parse_bool_flag, default=True, help="Pass disable_custom_all_reduce to vLLM")
    parser.add_argument("--disable-mm-preprocessor-cache", type=_parse_bool_flag, default=True, help="Pass disable_mm_preprocessor_cache to vLLM")
    parser.add_argument("--skip-tokenizer-init", type=_parse_bool_flag, default=False, help="Pass skip_tokenizer_init to vLLM")
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

    if not os.path.isfile(os.path.join(args.model_path, "config.json")):
        raise FileNotFoundError(f"MODEL_PATH does not look like a HF model directory: {args.model_path}")

    os.environ["VLLM_USE_V1"] = str(args.vllm_use_v1)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = str(args.worker_multiproc_method)
    _configure_vllm_sampler()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=bool(args.trust_remote_code),
    )
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=int(args.tensor_parallel_size),
        dtype=str(args.dtype),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        max_num_batched_tokens=int(args.max_num_batched_tokens),
        enforce_eager=bool(args.enforce_eager),
        enable_sleep_mode=bool(args.enable_sleep_mode),
        enable_prefix_caching=bool(args.enable_prefix_caching),
        enable_chunked_prefill=bool(args.enable_chunked_prefill),
        disable_log_stats=bool(args.disable_log_stats),
        disable_custom_all_reduce=bool(args.disable_custom_all_reduce),
        disable_mm_preprocessor_cache=bool(args.disable_mm_preprocessor_cache),
        skip_tokenizer_init=bool(args.skip_tokenizer_init),
        trust_remote_code=bool(args.trust_remote_code),
    )
    sampling_params = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        max_tokens=int(args.max_tokens),
    )
    model_tag = args.model_tag or os.path.basename(os.path.abspath(args.model_path.rstrip("/")))
    max_input_tokens = (
        int(args.max_input_tokens)
        if args.max_input_tokens is not None
        else max(1, int(args.max_model_len) - int(args.max_tokens))
    )

    results: List[Dict[str, Any]] = []
    sample_indices = list(range(len(env.samples)))
    progress_bar = tqdm(
        total=len(sample_indices),
        desc="Token Estimation Eval (vLLM)",
        unit="sample",
        dynamic_ncols=True,
    )
    try:
        for index_chunk in _chunk_list(sample_indices, int(args.request_batch_size)):
            indexed_samples = []
            prompt_texts: List[str] = []
            prompt_text_by_sample_index: Dict[int, str] = {}
            api_messages_by_sample_index: Dict[int, List[Dict[str, Any]]] = {}
            original_api_messages_by_sample_index: Dict[int, List[Dict[str, Any]]] = {}
            trim_info_by_sample_index: Dict[int, Dict[str, int]] = {}

            for sample_index in index_chunk:
                sample = env.get_sample(sample_index)
                api_messages = env.build_api_messages(sample)
                trimmed_messages, prompt_text, trim_info = _trim_messages_to_fit(
                    tokenizer,
                    api_messages,
                    max_input_tokens=max_input_tokens,
                )
                indexed_samples.append((sample_index, sample))
                prompt_texts.append(prompt_text)
                prompt_text_by_sample_index[sample_index] = prompt_text
                api_messages_by_sample_index[sample_index] = trimmed_messages
                original_api_messages_by_sample_index[sample_index] = api_messages
                trim_info_by_sample_index[sample_index] = trim_info

            batch_results: List[Dict[str, Any]] = []
            try:
                outputs = llm.generate(prompt_texts, sampling_params=sampling_params)
                for (sample_index, _sample), output in zip(indexed_samples, outputs):
                    response_text = output.outputs[0].text if output.outputs else ""
                    usage = _usage_from_text(
                        tokenizer,
                        prompt_text_by_sample_index[sample_index],
                        response_text,
                    )
                    batch_results.append(
                        {
                            "success": True,
                            "response": response_text,
                            "provider": "vllm-local",
                            "model": model_tag,
                            "usage": usage,
                            "error": None,
                            "error_type": None,
                            "error_code": None,
                            "status_code": None,
                            "request_id": None,
                            "attempts": [
                                {
                                    "success": True,
                                    "provider": "vllm-local",
                                    "model": model_tag,
                                    "input_tokens": usage["input_tokens"],
                                    "output_tokens": usage["output_tokens"],
                                    "total_tokens": usage["total_tokens"],
                                    "usage": usage,
                                    "request_id": None,
                                }
                            ],
                        }
                    )
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                batch_results = [
                    {
                        "success": False,
                        "response": "",
                        "provider": "vllm-local",
                        "model": model_tag,
                        "usage": None,
                        "error": error_message,
                        "error_type": None,
                        "error_code": None,
                        "status_code": None,
                        "request_id": None,
                        "attempts": [
                            {
                                "success": False,
                                "provider": "vllm-local",
                                "model": model_tag,
                                "error": error_message,
                                "exception_class": type(exc).__name__,
                                "request_id": None,
                            }
                        ],
                    }
                    for _ in indexed_samples
                ]

            for (sample_index, sample), api_result in zip(indexed_samples, batch_results):
                env.reset(index=sample_index)
                api_messages = api_messages_by_sample_index[sample_index]
                _, _, _, info = env.step(api_result.get("response", ""))
                results.append(
                    {
                        "sample_id": sample.sample_id,
                        "rollout_index": sample.rollout_index,
                        "turn_idx": sample.turn_idx,
                        "source_system": sample.source_system,
                        "input_messages": api_messages,
                        "untrimmed_input_messages": original_api_messages_by_sample_index[sample_index],
                        "rollout_history_messages": sample.input_messages,
                        "input_text": json.dumps(api_messages, ensure_ascii=False, indent=2),
                        "rendered_prompt_text": prompt_text_by_sample_index[sample_index],
                        "trim_info": trim_info_by_sample_index[sample_index],
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
        "config": vars(args),
        "temp_json_path": temp_json_path,
        "summary": _build_summary(results),
        "results": results,
    }
    _ensure_parent_dir(args.output_json)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
