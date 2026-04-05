import json
import os
import time
from typing import Any, Dict, Optional

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl import DataProto

from ragen.llm_agent.agent_proxy import ApiCallingWrapperWg, LLMAgentProxy


def _normalize_output_cfg(config) -> Optional[Dict[str, Any]]:
    if not hasattr(config, "output"):
        return None
    return OmegaConf.to_object(config.output)


def _build_save_path(config, output_cfg: Optional[Dict[str, Any]], timestamp: str) -> str:
    if output_cfg is None:
        trainer_cfg = getattr(config, "trainer", None)
        base_dir_raw = (
            getattr(trainer_cfg, "local_log_dir", "results")
            if trainer_cfg is not None
            else "results"
        )
        exp_name = (
            getattr(trainer_cfg, "experiment_name", "eval_api")
            if trainer_cfg is not None
            else "eval_api"
        )
        base_dir = to_absolute_path(base_dir_raw)
        save_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, "val_rollouts.pkl")

    output_dir = to_absolute_path(output_cfg.get("dir", "results/eval"))
    os.makedirs(output_dir, exist_ok=True)

    filename = output_cfg.get("filename") or "val_rollouts.pkl"
    append_timestamp = output_cfg.get("append_timestamp", True)
    root, ext = os.path.splitext(filename)
    if not ext:
        ext = ".pkl"
    if append_timestamp:
        filename = f"{root}_{timestamp}{ext}"
    else:
        filename = f"{root}{ext}"
    return os.path.join(output_dir, filename)


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        try:
            return _to_jsonable(obj.tolist())
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return _to_jsonable(obj.item())
        except Exception:
            pass
    return str(obj)


def _save_rollout_json_artifacts(save_path: str, rollouts: DataProto) -> None:
    save_dir = os.path.dirname(save_path)
    json_dir = os.path.join(save_dir, "val_rollouts_json")
    os.makedirs(json_dir, exist_ok=True)

    metrics = rollouts.meta_info.get("metrics", {}) if rollouts.meta_info else {}
    metrics_path = os.path.join(json_dir, "metrics_summary.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metrics), f, ensure_ascii=False, indent=2)

    non_tensor = getattr(rollouts, "non_tensor_batch", {}) or {}
    histories = non_tensor.get("env_rollout_histories", None)
    if histories is None:
        return

    histories_data = _to_jsonable(histories)
    all_histories_path = os.path.join(json_dir, "env_rollout_histories.json")
    with open(all_histories_path, "w", encoding="utf-8") as f:
        json.dump(histories_data, f, ensure_ascii=False, indent=2)

    trajectories_dir = os.path.join(json_dir, "trajectories")
    os.makedirs(trajectories_dir, exist_ok=True)

    metrics_array = non_tensor.get("env_rollout_metrics", None)
    metrics_data = _to_jsonable(metrics_array) if metrics_array is not None else []

    if not isinstance(histories_data, list):
        histories_data = [histories_data]

    for idx, history in enumerate(histories_data):
        entry_metrics = metrics_data[idx] if isinstance(metrics_data, list) and idx < len(metrics_data) else {}
        instance_id = f"instance_{idx:04d}"

        if isinstance(history, list) and history:
            first = history[0]
            if isinstance(first, dict):
                state = str(first.get("state", ""))
                for marker in ("Instance ID:", "instance_id:"):
                    if marker in state:
                        line = next((ln for ln in state.splitlines() if marker in ln), "")
                        if line:
                            instance_id = line.split(marker, 1)[-1].strip().replace("/", "_").replace(" ", "_")
                        break

        trajectory_path = os.path.join(trajectories_dir, f"{idx:04d}_{instance_id}.json")
        with open(trajectory_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "instance_index": idx,
                    "instance_id": instance_id,
                    "history": _to_jsonable(history),
                    "metrics": _to_jsonable(entry_metrics),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


@hydra.main(version_base=None, config_path="../config", config_name="evaluate_api_llm")
def main(config):
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    actor_wg = ApiCallingWrapperWg(config, tokenizer)
    proxy = LLMAgentProxy(config, actor_wg, tokenizer)

    start_time = time.time()
    rollouts = proxy.rollout(
        DataProto(
            batch=None,
            non_tensor_batch=None,
            meta_info={
                "eos_token_id": 151645,
                "pad_token_id": 151643,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            },
        ),
        val=True,
    )
    elapsed = time.time() - start_time

    rm_scores = rollouts.batch["rm_scores"]
    metrics = rollouts.meta_info["metrics"]
    avg_reward = rm_scores.sum(-1).mean().item()

    print(f"rollout time: {elapsed} seconds")
    print(f"rollout rewards: {avg_reward}")
    print("metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_cfg = _normalize_output_cfg(config)
    save_path = _build_save_path(config, output_cfg, timestamp)
    rollouts.save_to_disk(save_path)
    _save_rollout_json_artifacts(save_path, rollouts)

    save_dir = os.path.dirname(save_path)
    print(
        f"Saved validation rollouts to {save_path}. "
        f"JSON artifacts: {os.path.join(save_dir, 'val_rollouts_json')}"
    )


if __name__ == "__main__":
    main()
