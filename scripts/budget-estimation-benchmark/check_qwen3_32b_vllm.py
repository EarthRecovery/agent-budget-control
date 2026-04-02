#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time
import traceback


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Minimal bare-vLLM probe for the same runtime knobs used by "
            "the local rollout smoke script. It only tries to load the model "
            "and run one tiny generation."
        )
    )
    parser.add_argument(
        "--model-path",
        default="/projects/e32695/Qwen3-14B",
        help="Local Hugging Face model directory.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Tensor parallel size. Use 4 for 4x A100.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model dtype passed to vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.25,
        help="vLLM gpu_memory_utilization.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
        help="Small context window to minimize KV cache allocation.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=1024,
        help="Small batched token budget for a lightweight probe.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Generation length for the probe request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Sampling top-p.",
    )
    parser.add_argument(
        "--trust-remote-code",
        type=str_to_bool,
        default=False,
        help="Pass trust_remote_code to vLLM. Smoke default is False.",
    )
    parser.add_argument(
        "--vllm-use-v1",
        type=int,
        choices=[0, 1],
        default=1,
        help="Set VLLM_USE_V1 before importing vLLM. Default forces V1.",
    )
    parser.add_argument(
        "--worker-multiproc-method",
        choices=["fork", "spawn"],
        default="spawn",
        help=(
            "Set VLLM_WORKER_MULTIPROC_METHOD before importing vLLM. "
            "Smoke path uses spawn."
        ),
    )
    parser.add_argument(
        "--enforce-eager",
        type=str_to_bool,
        default=True,
        help="Pass enforce_eager to vLLM. Smoke default is True.",
    )
    parser.add_argument(
        "--enable-sleep-mode",
        type=str_to_bool,
        default=False,
        help="Pass enable_sleep_mode to vLLM. Smoke default is False.",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        type=str_to_bool,
        default=False,
        help="Pass enable_prefix_caching to vLLM. Smoke default is False.",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        type=str_to_bool,
        default=False,
        help="Pass enable_chunked_prefill to vLLM. eval.yaml default is False.",
    )
    parser.add_argument(
        "--disable-log-stats",
        type=str_to_bool,
        default=True,
        help="Pass disable_log_stats to vLLM. Smoke default is True.",
    )
    parser.add_argument(
        "--disable-custom-all-reduce",
        type=str_to_bool,
        default=True,
        help="Pass disable_custom_all_reduce to vLLM. Smoke default is True.",
    )
    parser.add_argument(
        "--disable-mm-preprocessor-cache",
        type=str_to_bool,
        default=True,
        help="Pass disable_mm_preprocessor_cache to vLLM. Smoke default is True.",
    )
    parser.add_argument(
        "--skip-tokenizer-init",
        type=str_to_bool,
        default=False,
        help="Pass skip_tokenizer_init to vLLM. Smoke default is False.",
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=None,
        help="Sampling logprobs. Smoke default is null.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Sampling top-k. Smoke default is -1.",
    )
    parser.add_argument(
        "--prompt",
        default="Move one step and answer briefly.",
        help="Prompt used for the single test request.",
    )
    return parser.parse_args()


def _find_nvcc():
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        cuda_home_nvcc = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.isfile(cuda_home_nvcc) and os.access(cuda_home_nvcc, os.X_OK):
            return cuda_home_nvcc
    return shutil.which("nvcc")


def _configure_vllm_sampler():
    preset = os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")
    if preset is not None:
        print(f"VLLM_USE_FLASHINFER_SAMPLER={preset} (preset)")
        return

    nvcc_path = _find_nvcc()
    if nvcc_path:
        print(f"nvcc={nvcc_path}")
        return

    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    print(
        "nvcc not found; setting VLLM_USE_FLASHINFER_SAMPLER=0 "
        "to avoid FlashInfer JIT compilation."
    )


def main():
    args = parse_args()
    os.environ["VLLM_USE_V1"] = str(args.vllm_use_v1)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = args.worker_multiproc_method

    print("==> Qwen3 vLLM probe")
    print(f"model_path={args.model_path}")
    print(f"VLLM_USE_V1={os.environ['VLLM_USE_V1']}")
    print(
        "VLLM_WORKER_MULTIPROC_METHOD="
        f"{os.environ['VLLM_WORKER_MULTIPROC_METHOD']}"
    )
    print(f"tensor_parallel_size={args.tensor_parallel_size}")
    print(f"dtype={args.dtype}")
    print(f"gpu_memory_utilization={args.gpu_memory_utilization}")
    print(f"max_model_len={args.max_model_len}")
    print(f"max_num_batched_tokens={args.max_num_batched_tokens}")
    print(f"max_tokens={args.max_tokens}")
    print(f"enforce_eager={args.enforce_eager}")
    print(f"enable_sleep_mode={args.enable_sleep_mode}")
    print(f"enable_prefix_caching={args.enable_prefix_caching}")
    print(f"enable_chunked_prefill={args.enable_chunked_prefill}")
    print(f"disable_log_stats={args.disable_log_stats}")
    print(f"disable_custom_all_reduce={args.disable_custom_all_reduce}")
    print(f"disable_mm_preprocessor_cache={args.disable_mm_preprocessor_cache}")
    print(f"skip_tokenizer_init={args.skip_tokenizer_init}")
    print(f"trust_remote_code={args.trust_remote_code}")
    print(f"temperature={args.temperature}")
    print(f"top_p={args.top_p}")
    print(f"top_k={args.top_k}")
    print(f"logprobs={args.logprobs}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    _configure_vllm_sampler()
    print(
        'If vLLM prints "falling back to Transformers implementation", '
        "then Qwen3 is not on a native vLLM path in this environment."
    )
    sys.stdout.flush()

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        print(f"Failed to import vLLM: {exc}", file=sys.stderr)
        raise

    load_start = time.time()
    llm_kwargs = dict(
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        enable_sleep_mode=args.enable_sleep_mode,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        disable_log_stats=args.disable_log_stats,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
        skip_tokenizer_init=args.skip_tokenizer_init,
        trust_remote_code=args.trust_remote_code,
    )
    llm = LLM(
        model=args.model_path,
        **llm_kwargs,
    )
    load_elapsed = time.time() - load_start
    print(f"Model loaded in {load_elapsed:.2f}s")
    sys.stdout.flush()

    sampling_kwargs = dict(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    if args.logprobs is not None:
        sampling_kwargs["logprobs"] = args.logprobs
    sampling_params = SamplingParams(**sampling_kwargs)

    infer_start = time.time()
    outputs = llm.generate([args.prompt], sampling_params=sampling_params)
    infer_elapsed = time.time() - infer_start

    generated = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    print(f"Inference finished in {infer_elapsed:.2f}s")
    print("Prompt:")
    print(args.prompt)
    print("Generated:")
    print(generated)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Probe failed with an exception:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
