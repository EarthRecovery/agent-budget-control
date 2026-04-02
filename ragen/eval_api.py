import os
import time

import hydra
from transformers import AutoTokenizer
from verl import DataProto

from ragen.eval_api_utils import (
    clone_config_for_val_chunk,
    iter_val_rollout_chunks,
    resolve_rollout_chunk_size,
)
from ragen.llm_agent.agent_proxy import (
    ApiCallingWrapperWg,
    LLMAgentProxy,
    _build_save_path,
    _normalize_output_cfg,
)


def _build_eval_input() -> DataProto:
    return DataProto(
        batch=None,
        non_tensor_batch=None,
        meta_info={
            "eos_token_id": 151645,
            "pad_token_id": 151643,
            "recompute_log_prob": False,
            "do_sample": False,
            "validate": True,
        },
    )


def _run_val_rollout(config, tokenizer) -> DataProto:
    actor_wg = ApiCallingWrapperWg(config, tokenizer)
    proxy = LLMAgentProxy(config, actor_wg, tokenizer)
    return proxy.rollout(_build_eval_input(), val=True)


def _print_rollout_summary(label: str, rollouts: DataProto, elapsed_seconds: float) -> None:
    print(f"{label} rollout time: {elapsed_seconds} seconds")
    rm_scores = rollouts.batch["rm_scores"]
    metrics = rollouts.meta_info["metrics"]
    avg_reward = rm_scores.sum(-1).mean().item()
    print(f"{label} rollout rewards: {avg_reward}")
    print(f"{label} metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


@hydra.main(version_base=None, config_path="../config", config_name="evaluate_api_llm")
def main(config):
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    val_cfg = config.es_manager.val
    rollout_chunks = iter_val_rollout_chunks(
        total_env_groups=val_cfg.env_groups,
        start_group_index=val_cfg.start_group_index,
        chunk_size=resolve_rollout_chunk_size(config),
    )

    start_time = time.time()
    chunk_rollouts = []
    total_chunks = len(rollout_chunks)
    for chunk_index, (chunk_offset, chunk_start_group_index, chunk_env_groups) in enumerate(
        rollout_chunks,
        start=1,
    ):
        chunk_config = clone_config_for_val_chunk(
            config,
            chunk_offset=chunk_offset,
            chunk_start_group_index=chunk_start_group_index,
            chunk_env_groups=chunk_env_groups,
        )
        chunk_start_time = time.time()
        if total_chunks > 1:
            print(
                f"Running rollout chunk {chunk_index}/{total_chunks}: "
                f"start_group_index={chunk_start_group_index}, env_groups={chunk_env_groups}"
            )
        rollouts = _run_val_rollout(chunk_config, tokenizer)
        chunk_elapsed = time.time() - chunk_start_time
        if total_chunks > 1:
            _print_rollout_summary(f"chunk {chunk_index}/{total_chunks}", rollouts, chunk_elapsed)
        chunk_rollouts.append(rollouts)

    rollouts = (
        DataProto.concat(chunk_rollouts)
        if len(chunk_rollouts) > 1
        else chunk_rollouts[0]
    )
    end_time = time.time()
    _print_rollout_summary("rollout", rollouts, end_time - start_time)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_cfg = _normalize_output_cfg(config)
    save_path = _build_save_path(config, output_cfg, timestamp)
    rollouts.save_to_disk(save_path)
    dir_path = os.path.dirname(save_path)
    print(
        f"save validation results to {save_path}. To visualize, run: "
        f"python scripts/visualize.py --rollout_path {dir_path}"
    )



if __name__ == "__main__":
    main()
