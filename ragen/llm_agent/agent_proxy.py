from .ctx_manager import ContextManager
from ragen.wrapper.ctx_manager_wrapper import CtxManagerWrapper
from .es_manager import EnvStateManager
from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
from pathlib import Path
from typing import List, Dict, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from .base_llm import ConcurrentLLM
from .eval_config import (
    expand_compliance_group_size,
    resolve_eval_estimation_mode,
    resolve_rollout_max_turn,
)
import time
from hydra.utils import to_absolute_path
import numpy as np
from omegaconf import OmegaConf, open_dict
import wandb


def _get_rollout_val_kwarg(ro_config, key: str, default=None):
    return OmegaConf.select(
        ro_config,
        f"val_kwargs.{key}",
        default=OmegaConf.select(ro_config, key, default=default),
    )


def _get_rollout_do_sample(config) -> bool:
    return bool(
        OmegaConf.select(
            config,
            "actor_rollout_ref.rollout.val_kwargs.do_sample",
            default=OmegaConf.select(
                config, "actor_rollout_ref.rollout.do_sample", default=False
            ),
        )
    )


_resolve_rollout_max_turn = resolve_rollout_max_turn


class VllmWrapperWg:  # Thi is a developing class for eval and test
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        model_name = config.actor_rollout_ref.model.path
        ro_config = config.actor_rollout_ref.rollout
        temperature = _get_rollout_val_kwarg(ro_config, "temperature", default=1.0)
        top_p = _get_rollout_val_kwarg(ro_config, "top_p", default=1.0)
        top_k = _get_rollout_val_kwarg(ro_config, "top_k", default=-1)
        logprobs = _get_rollout_val_kwarg(ro_config, "logprobs", default=None)
        log_stats_interval = getattr(ro_config, "log_stats_interval", None)
        llm_kwargs = dict(
            enable_sleep_mode=bool(getattr(ro_config, "enable_sleep_mode", True)),
            tensor_parallel_size=ro_config.tensor_model_parallel_size,
            dtype=ro_config.dtype,
            enforce_eager=ro_config.enforce_eager,
            gpu_memory_utilization=ro_config.gpu_memory_utilization,
            disable_custom_all_reduce=bool(getattr(ro_config, "disable_custom_all_reduce", True)),
            disable_mm_preprocessor_cache=bool(
                getattr(ro_config, "disable_mm_preprocessor_cache", True)
            ),
            skip_tokenizer_init=bool(getattr(ro_config, "skip_tokenizer_init", False)),
            max_model_len=ro_config.max_model_len,
            disable_log_stats=ro_config.disable_log_stats,
            max_num_batched_tokens=ro_config.max_num_batched_tokens,
            enable_chunked_prefill=ro_config.enable_chunked_prefill,
            enable_prefix_caching=bool(getattr(ro_config, "enable_prefix_caching", True)),
            trust_remote_code=bool(getattr(ro_config, "trust_remote_code", True)),
        )
        if log_stats_interval is not None:
            llm_kwargs["log_stats_interval"] = log_stats_interval
        self.llm = LLM(
            model_name,
            **llm_kwargs,
        )
        print("LLM initialized")
        sampling_kwargs = dict(
            max_tokens=ro_config.response_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        if logprobs is not None:
            sampling_kwargs["logprobs"] = logprobs
        self.sampling_params = SamplingParams(**sampling_kwargs)

    def generate_sequences(self, lm_inputs: DataProto):
        """
        Convert the input ids to text, and then generate the sequences. Finally create a dataproto.
        This aligns with the verl Worker Group interface.
        """
        # NOTE: free_cache_engine is not used in the vllm wrapper. Only used in the verl vllm.
        # cache_action = lm_inputs.meta_info.get("cache_action", None)

        if lm_inputs.meta_info.get("skip_generation", False):
            return lm_inputs

        input_ids = lm_inputs.batch["input_ids"]
        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        input_texts = [i.replace("<|endoftext|>", "") for i in input_texts]

        outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
        texts = [output.outputs[0].text for output in outputs]

        # get the entropy of the response
        entropys = []
        n_tokens = []
        for output in outputs:
            output_data = output.outputs[0]
            token_logprobs = getattr(output_data, "logprobs", None)
            token_ids = getattr(output_data, "token_ids", None) or []
            if token_logprobs is None:
                entropys.append(0.0)
                n_tokens.append(len(token_ids))
                continue

            entropy_of_the_series = []
            for logprob_in_a_token in token_logprobs:
                logprobs = np.array([i.logprob for i in logprob_in_a_token.values()])
                entropy_of_the_token = -(logprobs * np.exp(logprobs)).sum()
                entropy_of_the_series.append(entropy_of_the_token)
            entropy_of_the_series = np.array(entropy_of_the_series)
            entropys.append(entropy_of_the_series.sum())
            n_tokens.append(len(token_logprobs))
        entropys = np.array(entropys)
        n_tokens = np.array(n_tokens)

        # get the in_group_std of the response
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
            "response_texts": texts,
            "env_ids": lm_inputs.non_tensor_batch["env_ids"],
            "group_ids": lm_inputs.non_tensor_batch["group_ids"],
            "entropys": entropys,
            "n_tokens": n_tokens,
        }  # this is a bit hard-coded to bypass the __init__ check in DataProto
        lm_outputs.meta_info = lm_inputs.meta_info

        return lm_outputs


class ApiCallingWrapperWg:
    """Wrapper class for API-based LLM calls that fits into the VERL framework"""

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = OmegaConf.to_container(model_info.generation_kwargs, resolve=True)
        self.api_batch_size = max(0, int(getattr(config.model_config, "api_batch_size", 0) or 0))
        self.prompt_token_margin = int(getattr(config.model_config, "prompt_token_margin", 0) or 0)
        max_model_len = OmegaConf.select(config, "actor_rollout_ref.rollout.max_model_len", default=None)
        response_length = OmegaConf.select(config, "actor_rollout_ref.rollout.response_length", default=None)
        if response_length is not None:
            if "max_completion_tokens" in self.llm_kwargs:
                self.llm_kwargs["max_completion_tokens"] = int(response_length)
            elif "max_tokens" in self.llm_kwargs:
                self.llm_kwargs["max_tokens"] = int(response_length)
        self.prompt_token_budget = None
        if max_model_len is not None:
            self.prompt_token_budget = max(
                1,
                int(max_model_len) - int(response_length or 0) - self.prompt_token_margin,
            )

        api_key = OmegaConf.select(model_info, "api_key", default=None)
        self.llm = ConcurrentLLM(
			provider=model_info.provider_name,
            model_name=model_info.model_name,
            api_key=api_key,
            max_concurrency=config.model_config.max_concurrency
        )
        print(f"API-based LLM ({model_info.provider_name} - {model_info.model_name}) initialized")

    def _iter_api_batches(self, active_indices, active_messages_list):
        if self.api_batch_size <= 0:
            yield active_indices, active_messages_list
            return

        for start in range(0, len(active_messages_list), self.api_batch_size):
            end = start + self.api_batch_size
            yield active_indices[start:end], active_messages_list[start:end]

    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        """
        Convert the input ids to text, make API calls to generate responses,
        and create a DataProto with the results.
        """

        if lm_inputs.meta_info.get("skip_generation", False):
            return lm_inputs

        messages_list = lm_inputs.non_tensor_batch["messages_list"].tolist()
        response_errors = [None] * len(messages_list)
        texts = [""] * len(messages_list)
        api_interactions = [[] for _ in messages_list]
        api_usages = [None] * len(messages_list)
        active_indices = list(range(len(messages_list)))
        active_messages_list = messages_list

        attention_mask = None
        if getattr(lm_inputs, "batch", None) is not None:
            attention_mask = lm_inputs.batch.get("attention_mask")
        if attention_mask is not None and self.prompt_token_budget is not None:
            prompt_token_counts = attention_mask.sum(dim=-1).tolist()
            active_indices = []
            active_messages_list = []
            for idx, messages in enumerate(messages_list):
                prompt_tokens = int(prompt_token_counts[idx])
                if prompt_tokens > self.prompt_token_budget:
                    response_errors[idx] = {
                        "error": (
                            f"Prompt token estimate {prompt_tokens} exceeds local API prompt budget "
                            f"{self.prompt_token_budget} (max_model_len={self.config.actor_rollout_ref.rollout.max_model_len}, "
                            f"response_length={self.config.actor_rollout_ref.rollout.response_length}, "
                            f"prompt_token_margin={self.prompt_token_margin})."
                        ),
                        "error_type": "context_length_exceeded_local_guard",
                        "error_code": "prompt_too_long_local",
                        "status_code": None,
                        "retryable": False,
                    }
                else:
                    active_indices.append(idx)
                    active_messages_list.append(messages)

        unresolved_failed_messages = []
        for batch_indices, batch_messages_list in self._iter_api_batches(
            active_indices,
            active_messages_list,
        ):
            results, failed_messages = self.llm.run_batch(
                messages_list=batch_messages_list, **self.llm_kwargs
            ) if batch_messages_list else ([], [])
            unresolved_failed_messages.extend(failed_messages)

            for idx, result in zip(batch_indices, results):
                texts[idx] = result.get("response", "")
                api_interactions[idx] = list(result.get("attempts", []) or [])
                api_usages[idx] = result.get("usage")
                if not result.get("success", False):
                    response_errors[idx] = {
                        "error": result.get("error"),
                        "error_type": result.get("error_type"),
                        "error_code": result.get("error_code"),
                        "status_code": result.get("status_code"),
                        "retryable": result.get("retryable", False),
                    }
        if unresolved_failed_messages:
            print(
                f"[DEBUG] unresolved failed messages after retries: "
                f"{len(unresolved_failed_messages)}"
            )
        print(f"[DEBUG] texts: {texts}")
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
            "response_texts": texts,
            "env_ids": lm_inputs.non_tensor_batch["env_ids"],
            "group_ids": lm_inputs.non_tensor_batch["group_ids"],
            "response_errors": np.array(response_errors, dtype=object),
            "api_interactions": np.array(api_interactions, dtype=object),
            "api_usages": np.array(api_usages, dtype=object),
        }  # this is a bit hard-coded to bypass the __init__ check in DataProto
        lm_outputs.meta_info = lm_inputs.meta_info

        return lm_outputs


class LLMAgentProxy:
    """
    The proxy means the llm agent is trying to generate some rollout **at this time**, **at this model state**, **at this env state from the env config**
    """

    def __init__(self, config, actor_rollout_wg, tokenizer):
        expand_compliance_group_size(config)
        self.config = config
        self.train_ctx_manager = ContextManager(config, tokenizer, mode="train")
        self.train_es_manager = EnvStateManager(config, mode="train")
        self.val_ctx_manager = ContextManager(config, tokenizer, mode="val")
        self.val_es_manager = EnvStateManager(config, mode="val")
        self.ctx_wrapper = CtxManagerWrapper(config, tokenizer)
        self.actor_wg = actor_rollout_wg
        self.tokenizer = tokenizer
        self._last_padded_inputs = None

    def _agent_proxy_get(self, key: str, default=None):
        agent_cfg = getattr(self.config, "agent_proxy", None)
        if agent_cfg is None:
            return default
        if hasattr(agent_cfg, "get"):
            value = agent_cfg.get(key, None)
            if value is None:
                value = agent_cfg.get(key.replace("-", "_"), None)
            return default if value is None else value
        return getattr(agent_cfg, key.replace("-", "_"), default)

    def _get_eval_estimation_mode(self) -> Optional[str]:
        return resolve_eval_estimation_mode(self.config)

    def _get_generation_suffix(self) -> str:
        eval_estimation_mode = self._get_eval_estimation_mode()
        if eval_estimation_mode in {"single", "multi", "toolcall"}:
            return "<budget-thinking>"
        if bool(getattr(self.config.agent_proxy, "enable_think", False)):
            return "<think>"
        return "<answer>"

    def generate_sequences(self, lm_inputs: DataProto):
        # TODO: add kv cache both for the vllm wrapper here and for verl vllm.
        if isinstance(self.actor_wg, RayWorkerGroup):
            padded_lm_inputs, pad_size = pad_dataproto_to_divisor(
                lm_inputs, self.actor_wg.world_size
            )
            self._last_padded_inputs = padded_lm_inputs
            padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)
            if lm_inputs.meta_info.get("skip_generation", False):
                return lm_inputs
            lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
            lm_outputs.meta_info = lm_inputs.meta_info
            lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
        elif isinstance(self.actor_wg, VllmWrapperWg) or isinstance(
            self.actor_wg, ApiCallingWrapperWg
        ):
            lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
        else:
            raise ValueError(f"Unsupported actor worker type: {type(self.actor_wg)}")

        return lm_outputs

    def rollout(self, dataproto: DataProto, val=False):
        es_manager = self.val_es_manager if val else self.train_es_manager
        ctx_manager = self.val_ctx_manager if val else self.train_ctx_manager
        self.ctx_wrapper.begin_rollout()
        env_outputs = es_manager.reset()
        ctx_manager.reset_memory_managers()

        max_turn = _resolve_rollout_max_turn(self.config)
        multi_turn = max_turn > 1
        finalized = False
        last_inputs = None

        n_turns, n_tokens, entropys = (
            np.zeros(len(env_outputs)),
            np.zeros(len(env_outputs)),
            np.zeros(len(env_outputs)),
        )  # to calculate instance-level entropy

        for i in range(max_turn):
            if len(env_outputs) == 0:
                break
            lm_inputs: DataProto = ctx_manager.get_lm_inputs(
                env_outputs, prepare_for_update=False
            )
            lm_inputs.meta_info = (
                dataproto.meta_info
            )  # TODO: setup vllm early stop when max length is reached. make sure this can be done
            last_inputs = lm_inputs
            if multi_turn:
                if i == 0:
                    mode = "multiturn-start"
                elif i == max_turn - 1:
                    mode = "multiturn-end"
                else:
                    mode = "multiturn-middle"
            else:
                mode = "singleturn"
            lm_inputs.meta_info["mode"] = mode
            self.ctx_wrapper.set_state(turn_idx=i, mode=mode, max_turn=max_turn)
            generation_suffix = self._get_generation_suffix()
            lm_inputs = self.ctx_wrapper.intercept(
                lm_inputs,
                add_generation_prompt=True,
                generation_suffix=generation_suffix,
            )
            lm_outputs: DataProto = self.generate_sequences(lm_inputs)
            self.ctx_wrapper.log_outputs(lm_outputs)

            # calculate entropy
            if "entropys" in lm_outputs.non_tensor_batch:
                turn_entropy, env_ids = (
                    lm_outputs.non_tensor_batch["entropys"],
                    lm_outputs.non_tensor_batch["env_ids"],
                )
                n_tokens[env_ids] += lm_outputs.non_tensor_batch["n_tokens"]
                entropys[env_ids] += turn_entropy
                n_turns[env_ids] += 1

            if mode == "multiturn-end":
                finalized = True
            env_inputs: List[Dict] = ctx_manager.get_env_inputs(lm_outputs)
            env_outputs: List[Dict] = es_manager.step(env_inputs)
            if len(env_outputs) == 0:  # all finished
                if multi_turn and not finalized and last_inputs is not None:
                    last_inputs.meta_info["skip_generation"] = True
                    last_inputs.meta_info["mode"] = "multiturn-end"
                    self.generate_sequences(last_inputs)
                    finalized = True
                break

        if multi_turn and not finalized and last_inputs is not None:
            last_inputs.meta_info["skip_generation"] = True
            last_inputs.meta_info["mode"] = "multiturn-end"
            self.generate_sequences(last_inputs)
        rollout_states = es_manager.get_rollout_states()
        self.ctx_wrapper.finalize_rollout(rollout_states)
        include_collapse_data = True
        if dataproto.meta_info is not None:
            include_collapse_data = dataproto.meta_info.get("compute_collapse", True)
        rollouts = ctx_manager.formulate_rollouts(
            rollout_states, include_collapse_data=include_collapse_data
        )
        eval_log_path = self.ctx_wrapper.get_estimation_log_path()
        eval_log_key = self.ctx_wrapper.get_eval_log_key()
        if eval_log_path and eval_log_key:
            if rollouts.meta_info is None:
                rollouts.meta_info = {}
            rollouts.meta_info[eval_log_key] = eval_log_path

        # calculate instance-level entropy
        if "entropys" in rollouts.non_tensor_batch:
            safe_n_tokens = np.where(n_tokens > 0, n_tokens, 1)
            rollouts.non_tensor_batch["entropys"] = entropys / safe_n_tokens
            rollouts.non_tensor_batch["n_generated_tokens"] = n_tokens
            rollouts.non_tensor_batch["n_turns"] = n_turns

        return rollouts


def _normalize_output_cfg(config) -> Optional[Dict]:
    if not hasattr(config, "output"):
        return None
    return OmegaConf.to_object(config.output)


def _build_save_path(config, output_cfg: Optional[Dict], timestamp: str) -> str:
    if output_cfg is None:
        trainer_cfg = getattr(config, "trainer", None)
        base_dir_raw = (
            getattr(trainer_cfg, "local_log_dir", "results")
            if trainer_cfg is not None
            else "results"
        )
        exp_name = (
            getattr(trainer_cfg, "experiment_name", "eval")
            if trainer_cfg is not None
            else "eval"
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


@hydra.main(version_base=None, config_path="../../config", config_name="eval")
def main(config):
    # detect config name from python -m ragen.llm_agent.agent_proxy --config_name frozen_lake
    print("Starting evaluation process. Check config/eval.yaml for specific configs.")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    actor_wg = VllmWrapperWg(config, tokenizer)
    proxy = LLMAgentProxy(config, actor_wg, tokenizer)
    import time
    start_time = time.time()
    rollouts = proxy.rollout(
        DataProto(
            batch=None,
            non_tensor_batch=None,
            meta_info={
                "eos_token_id": 151645,
                "pad_token_id": 151643,
                "recompute_log_prob": False,
                "do_sample": _get_rollout_do_sample(config),
                "validate": True,
            }
        ),
        val=True
    )
    end_time = time.time()
    print(f"rollout time: {end_time - start_time} seconds")
    # print rollout rewards from the rm_scores
    rm_scores = rollouts.batch["rm_scores"]
    metrics = rollouts.meta_info["metrics"]
    avg_reward = rm_scores.sum(-1).mean().item()
    print(f"rollout rewards: {avg_reward}")
    print(f"metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # save to config.trainer.local_log_dir/config.trainer.experiment_name + _ + timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_cfg = _normalize_output_cfg(config)
    save_path = _build_save_path(config, output_cfg, timestamp)
    rollouts.save_to_disk(save_path)
    dir_path = os.path.dirname(save_path)
    print(
        f"save validation results to {save_path}. To visualize, run: python scripts/visualize.py --rollout_path {dir_path}"
    )


if __name__ == "__main__":
    import sys

    sys.argv.extend(
        [
            "--config-dir",
            os.path.join(os.path.dirname(__file__), "../../ragen/config"),
            "--config-dir",
            os.path.join(os.path.dirname(__file__), "../../verl/verl/trainer/config"),
        ]
    )
    main()
