import copy
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import gymnasium as gym

from ragen.env.base import BaseLanguageBasedEnv

from .config import OpenHandsSWEBenchVerifiedEnvConfig
from .utils import (
    compute_usage_delta,
    detect_finish_action,
    event_to_jsonable,
    extract_event_diagnostics,
    extract_usage_breakdown,
    extract_usage_totals,
    safe_format_template,
    summarize_events,
    to_plain_dict,
)


class OpenHandsSWEBenchVerifiedEnv(BaseLanguageBasedEnv, gym.Env):
    """OpenHands SWE-bench Verified environment adapter for RAGEN."""

    RUN_TURN_ACTION = "RUN_OPENHANDS_TURN"
    RECOVERABLE_ERROR_MARKERS = (
        "conversationrunerror",
        "remote conversation ended with error",
        "maximum iterations limit",
        "max iterations limit",
        "agent reached maximum iterations",
    )

    def __init__(self, config: Optional[OpenHandsSWEBenchVerifiedEnvConfig] = None):
        BaseLanguageBasedEnv.__init__(self)
        self.config = config if config is not None else OpenHandsSWEBenchVerifiedEnvConfig()

        self._imports: Dict[str, Any] = {}
        self._evaluator = None
        self._metadata = None
        self._instances = None

        self._workspace = None
        self._agent = None
        self._conversation = None

        self._instance = None
        self._instance_data = None
        self._instance_id = None
        self._repo_path = None

        self._turn_index = 0
        self._done = False
        self._success = False

        self._events_offset = 0
        self._usage_totals = {
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
            "cost_usd": 0.0,
        }
        self._usage_by_id: Dict[str, Dict[str, float]] = {}
        self._last_tool_status = "unknown"
        self._last_error = "none"
        self._last_state_summary = "No OpenHands activity yet."

        self._run_dir: Optional[Path] = None
        self._turn_dir: Optional[Path] = None
        self._turn_full_dir: Optional[Path] = None
        self._trajectory_path: Optional[Path] = None
        self._metrics_summary_path: Optional[Path] = None
        self._conversation_full_path: Optional[Path] = None
        self._trajectory: Dict[str, Any] = {}

        self._render_cache = "OpenHands environment not initialized yet."

    # ----------------------------- lifecycle -----------------------------

    def reset(self, seed: Optional[int] = None, mode: Optional[str] = None) -> str:
        del mode
        self.close()

        self._validate_preflight()
        self._ensure_runtime_env_defaults()
        self._lazy_imports()

        self._build_evaluator_if_needed()
        self._load_instances_if_needed()
        assert self._instances, "No SWE-bench instances available after dataset load."

        instance_index = (seed or 0) % len(self._instances)
        self._instance = self._instances[instance_index]
        self._instance_data = copy.deepcopy(self._instance.data)
        self._instance_id = self._instance.id

        self._prepare_output_paths()
        self._workspace = self._evaluator.prepare_workspace(self._instance)

        self._agent = self._build_agent()
        self._conversation = self._imports["Conversation"](
            agent=self._agent,
            workspace=self._workspace,
            callbacks=[],
            max_iteration_per_run=max(1, int(self.config.max_iterations_per_run)),
            delete_on_close=True,
        )

        self._imports["setup_acp_workspace"](self.config.agent_type, self._workspace)

        self._repo_path = f"/workspace/{self._instance_data['repo'].split('/')[-1]}/"
        self._instance_data["repo_path"] = self._repo_path

        cp_cmd = self._workspace.execute_command(
            f"mkdir -p {self._repo_path} ; cp -r /testbed/. {self._repo_path}"
        )
        if int(getattr(cp_cmd, "exit_code", 1)) != 0:
            raise RuntimeError(
                f"Failed to copy /testbed into workspace repo path: {getattr(cp_cmd, 'stderr', '')}"
            )

        reset_cmd = self._workspace.execute_command(
            f"cd {self._repo_path} ; git reset --hard"
        )
        if int(getattr(reset_cmd, "exit_code", 1)) != 0:
            raise RuntimeError(
                f"Failed to reset repo to clean state: {getattr(reset_cmd, 'stderr', '')}"
            )

        instruction = self._imports["get_instruction"](
            instance=self._instance_data,
            metadata=self._metadata,
            workspace_path=self._workspace.working_dir,
        )
        self._conversation.send_message(instruction)

        events = list(getattr(self._conversation.state, "events", []))
        self._events_offset = len(events)
        self._usage_totals = self._snapshot_usage_totals()
        self._usage_by_id = self._snapshot_usage_breakdown().get("by_usage_id", {})

        self._turn_index = 0
        self._done = False
        self._success = False
        self._last_tool_status = "ok"
        self._last_error = "none"
        self._last_state_summary = "Conversation initialized; awaiting first controlled turn."

        self._trajectory = {
            "instance_id": self._instance_id,
            "repo": self._instance_data.get("repo"),
            "base_commit": self._instance_data.get("base_commit"),
            "dataset": self.config.dataset,
            "split": self.config.split,
            "config": to_plain_dict(self.config),
            "instruction": instruction,
            "conversation_history": [],
            "per_turn_usage": [],
            "conversation_full_file": "conversation_full.json",
            "status": {
                "done": False,
                "success": False,
            },
        }

        self._render_cache = self._build_reset_state()
        self._persist_conversation_full(events)
        self._persist_trajectory()
        self._persist_metrics_summary()

        return self.render()

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self._conversation is None or self._workspace is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self._done:
            info = {
                "action_is_effective": 0.0,
                "action_is_valid": 0.0,
                "success": float(self._success),
                "openhands_action_duration_sec": 0.0,
                "openhands_new_events": 0.0,
                "openhands_tool_error": 0.0,
                "raw_reward": 0.0,
                "llm_prompt_tokens": 0.0,
                "llm_completion_tokens": 0.0,
                "llm_total_tokens": 0.0,
                "llm_cost_usd": 0.0,
            }
            return self.render(), 0.0, True, info

        raw_action = (action or "").strip()
        is_valid_action = float(
            raw_action == self.RUN_TURN_ACTION or raw_action.startswith("inject[")
        )

        turn_prompt_override = None
        if raw_action.startswith("inject[") and raw_action.endswith("]"):
            turn_prompt_override = raw_action[len("inject[") : -1].strip()

        self._turn_index += 1
        actions_left = max(0, int(self.config.max_steps) - self._turn_index)

        injected_prompt = self._compose_turn_prompt(
            actions_left=actions_left,
            per_turn_override=turn_prompt_override,
        )

        start_time = time.time()
        turn_error = None
        old_offset = self._events_offset
        usage_before = dict(self._usage_totals)
        usage_by_id_before = copy.deepcopy(self._usage_by_id)
        status_before = self._conversation_status()
        metrics_before = self._snapshot_metrics_raw()

        try:
            self._conversation.send_message(injected_prompt)
            self._conversation.run(timeout=int(self.config.conversation_timeout_sec))
        except Exception as exc:
            turn_error = f"{type(exc).__name__}: {exc}"

        duration = time.time() - start_time
        all_events = list(getattr(self._conversation.state, "events", []))
        new_events = all_events[old_offset:]
        self._events_offset = len(all_events)

        usage_after = self._snapshot_usage_totals()
        usage_breakdown_after = self._snapshot_usage_breakdown()
        usage_by_id_after = usage_breakdown_after.get("by_usage_id", {})
        usage_delta = compute_usage_delta(usage_before, usage_after)
        self._usage_totals = usage_after
        self._usage_by_id = usage_by_id_after
        metrics_after = self._snapshot_metrics_raw()

        status_after_raw = self._conversation_status()
        recoverable_turn_error = self._is_recoverable_turn_error(
            turn_error=turn_error,
            status_after=status_after_raw,
            new_events=new_events,
        )
        effective_turn_error = None if recoverable_turn_error else turn_error
        status_after = status_after_raw
        if recoverable_turn_error and str(status_after_raw).lower().strip() == "error":
            status_after = "recoverable_error"

        diagnostics = extract_event_diagnostics(new_events)
        self._last_tool_status = diagnostics.get("last_tool_status", self._last_tool_status)
        self._last_error = diagnostics.get("last_error", self._last_error)
        if effective_turn_error is not None:
            self._last_tool_status = "error"
            self._last_error = effective_turn_error
        elif self._is_recoverable_error_text(self._last_error):
            self._last_tool_status = "ok"
            self._last_error = "none"

        finished = detect_finish_action(all_events)
        reached_budget = self._turn_index >= int(self.config.max_steps)

        if effective_turn_error is not None:
            self._done = True
            self._success = False
        elif finished:
            self._done = True
            self._success = True
        elif reached_budget:
            self._done = True
            self._success = False
        else:
            self._done = False
            self._success = False

        self._last_state_summary = summarize_events(new_events)
        self._render_cache = self._build_step_state(
            new_events=new_events,
            usage_delta=usage_delta,
            duration=duration,
            status_after=status_after,
            turn_error=effective_turn_error,
        )

        usage_entry: Dict[str, Any] = {
            "turn_index": int(self._turn_index),
            "prompt_tokens": float(usage_delta["prompt_tokens"]),
            "completion_tokens": float(usage_delta["completion_tokens"]),
            "total_tokens": float(usage_delta["total_tokens"]),
            "cost_usd": float(usage_delta["cost_usd"]),
        }

        turn_payload: Dict[str, Any] = {
            "turn_index": int(self._turn_index),
            "llm_input": injected_prompt,
            "llm_output": self._extract_turn_llm_output(
                new_events=new_events,
                turn_error=effective_turn_error,
                drop_recoverable_errors=bool(recoverable_turn_error),
            ),
            "usage": dict(usage_entry),
            "status": {
                "execution_status": status_after,
                "done": bool(self._done),
                "success": bool(self._success),
                "error": effective_turn_error,
            },
        }

        full_turn_payload: Dict[str, Any] = {
            "turn_index": self._turn_index,
            "action": raw_action,
            "is_valid_action": bool(is_valid_action),
            "injected_prompt": injected_prompt,
            "timing_sec": float(duration),
            "events_offset_before": old_offset,
            "events_offset_after": self._events_offset,
            "status_before": status_before,
            "status_after": status_after,
            "usage_before": usage_before,
            "usage_after": usage_after,
            "usage_delta": usage_delta,
            "usage_by_id_before": usage_by_id_before,
            "usage_by_id_after": usage_by_id_after,
            "done": bool(self._done),
            "success": bool(self._success),
            "status_after_raw": status_after_raw,
            "recoverable_turn_error": bool(recoverable_turn_error),
            "error": effective_turn_error,
            "raw_error": turn_error,
        }
        if self.config.save_full_payload_json:
            full_turn_payload["new_events"] = [event_to_jsonable(ev) for ev in new_events]
            full_turn_payload["metrics_snapshot_before"] = metrics_before
            full_turn_payload["metrics_snapshot_after"] = metrics_after

        self._trajectory["conversation_history"].append(turn_payload)
        self._trajectory["per_turn_usage"].append(usage_entry)
        self._trajectory["status"] = {
            "done": bool(self._done),
            "success": bool(self._success),
            "turn_index": self._turn_index,
        }

        self._persist_turn_payload(turn_payload)
        self._persist_turn_full_payload(full_turn_payload)
        self._persist_conversation_full(all_events)
        self._persist_trajectory()
        self._persist_metrics_summary()

        if self._done:
            self._append_git_patch_if_available()
            self._persist_trajectory()
            self._persist_metrics_summary()

        info: Dict[str, Any] = {
            "action_is_effective": 0.0 if effective_turn_error else 1.0,
            "action_is_valid": float(is_valid_action),
            "success": float(self._success),
            "openhands_action_duration_sec": float(duration),
            "openhands_new_events": float(len(new_events)),
            "openhands_tool_error": 1.0 if effective_turn_error else 0.0,
            "raw_reward": 0.0,
            "llm_prompt_tokens": float(usage_delta["prompt_tokens"]),
            "llm_completion_tokens": float(usage_delta["completion_tokens"]),
            "llm_total_tokens": float(usage_delta["total_tokens"]),
            "llm_cost_usd": float(usage_delta["cost_usd"]),
        }

        return self.render(), 0.0, bool(self._done), info

    def render(self, mode: str = "text") -> str:
        del mode
        return self._render_cache

    def close(self):
        if self._conversation is not None:
            try:
                close_fn = getattr(self._conversation, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass

        if self._workspace is not None:
            try:
                close_fn = getattr(self._workspace, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass

        self._conversation = None
        self._workspace = None
        self._agent = None

    # ----------------------------- setup helpers -----------------------------

    def _validate_preflight(self):
        bench_root = Path(self.config.benchmarks_root).expanduser().resolve()
        if not bench_root.exists():
            raise RuntimeError(
                f"benchmarks_root does not exist: {bench_root}\n"
                "Set custom_envs.OpenHandsSWEBenchVerified.env_config.benchmarks_root=<path-to-benchmarks>."
            )

        sdk_root = bench_root / "vendor" / "software-agent-sdk"
        sdk_candidates = [
            sdk_root / "openhands-sdk",
            sdk_root / "openhands-tools",
            sdk_root / "openhands-workspace",
            sdk_root / "openhands-agent-server",
        ]
        sdk_ready = any(path.is_dir() and any(path.iterdir()) for path in sdk_candidates)
        if not sdk_ready:
            raise RuntimeError(
                "Benchmarks SDK workspace is missing or empty.\n"
                f"Expected initialized SDK under: {sdk_root}\n"
                "Run:\n"
                f"  cd {bench_root}\n"
                "  git submodule update --init --recursive\n"
                "  uv sync\n"
            )

        llm_cfg = Path(self.config.llm_config_path).expanduser().resolve()
        if not llm_cfg.exists():
            raise RuntimeError(
                f"OpenHands LLM config JSON not found: {llm_cfg}\n"
                "Create or point llm_config_path to a valid benchmarks .llm_config JSON file."
            )
        try:
            with llm_cfg.open("r", encoding="utf-8") as f:
                json.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"OpenHands LLM config is not valid JSON: {llm_cfg}\nError: {exc}"
            ) from exc

    def _ensure_runtime_env_defaults(self):
        """Set stable runtime env defaults for benchmarks/OpenHands interop."""
        bench_root = Path(self.config.benchmarks_root).expanduser().resolve()
        sdk_root = bench_root / "vendor" / "software-agent-sdk"

        # Benchmarks image tagging can include a Dockerfile-content hash when
        # IMAGE_TAG_PREFIX is unset. In mixed setups where prebuilt images are
        # tagged by SDK short SHA only, this causes a tag mismatch during
        # ensure_local_image(). Default to SDK short SHA unless the user
        # explicitly sets IMAGE_TAG_PREFIX.
        if not os.environ.get("IMAGE_TAG_PREFIX"):
            try:
                git_sha = subprocess.run(
                    ["git", "rev-parse", "--short=7", "HEAD"],
                    cwd=str(sdk_root),
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
                if git_sha:
                    os.environ["IMAGE_TAG_PREFIX"] = git_sha
            except Exception:
                pass

    def _bootstrap_python_path(self):
        bench_root = Path(self.config.benchmarks_root).expanduser().resolve()
        sdk_root = bench_root / "vendor" / "software-agent-sdk"
        bench_venv = bench_root / ".venv"

        candidates = [
            bench_root,
            sdk_root / "openhands-sdk",
            sdk_root / "openhands-tools",
            sdk_root / "openhands-workspace",
            sdk_root / "openhands-agent-server",
        ]

        # Benchmarks dependencies are installed by `uv sync` into benchmarks/.venv.
        # Add that site-packages path so RAGEN can import those packages while
        # running inside the user's active conda env.
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        venv_site_candidates = [bench_venv / "lib" / py_ver / "site-packages"]
        lib_root = bench_venv / "lib"
        if lib_root.exists():
            for path in sorted(lib_root.glob("python*/site-packages")):
                if path not in venv_site_candidates:
                    venv_site_candidates.append(path)

        candidates.extend(venv_site_candidates)

        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate.exists() and candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

    def _lazy_imports(self):
        if self._imports:
            self._pin_benchmarks_repo_root()
            return

        self._bootstrap_python_path()
        bench_root = Path(self.config.benchmarks_root).expanduser().resolve()
        prev_cwd = Path.cwd()

        try:
            # Benchmarks import-time version utilities call `git submodule status`
            # and assume current working directory is inside the benchmarks repo.
            os.chdir(str(bench_root))
            from benchmarks.swebench.run_infer import SWEBenchEvaluation, get_instruction, get_tools_for_preset
            import benchmarks.swebench.build_base_images as swebench_build_base_images
            from benchmarks.utils.acp import build_acp_agent, is_acp_agent, setup_acp_workspace
            from benchmarks.utils.litellm_proxy import build_eval_llm
            from benchmarks.utils.llm_config import load_llm_config
            from benchmarks.utils.models import EvalMetadata
            import benchmarks.utils.version as benchmarks_version
            from openhands.sdk import Agent, Conversation, Tool
            from openhands.sdk.context.condenser import LLMSummarizingCondenser
            from openhands.sdk.critic import PassCritic
            from openhands.tools.delegate import DelegateTool
        except Exception as exc:
            raise RuntimeError(
                "Failed to import OpenHands/benchmarks runtime modules.\n"
                f"benchmarks_root={bench_root}\n"
                "Run:\n"
                f"  cd {bench_root}\n"
                "  git submodule update --init --recursive\n"
                "  uv sync\n"
                f"Original import error: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            try:
                os.chdir(str(prev_cwd))
            except Exception:
                pass

        self._imports = {
            "SWEBenchEvaluation": SWEBenchEvaluation,
            "get_instruction": get_instruction,
            "get_tools_for_preset": get_tools_for_preset,
            "swebench_build_base_images": swebench_build_base_images,
            "build_acp_agent": build_acp_agent,
            "is_acp_agent": is_acp_agent,
            "setup_acp_workspace": setup_acp_workspace,
            "build_eval_llm": build_eval_llm,
            "load_llm_config": load_llm_config,
            "EvalMetadata": EvalMetadata,
            "benchmarks_version": benchmarks_version,
            "Agent": Agent,
            "Conversation": Conversation,
            "Tool": Tool,
            "LLMSummarizingCondenser": LLMSummarizingCondenser,
            "PassCritic": PassCritic,
            "DelegateTool": DelegateTool,
        }
        self._pin_benchmarks_repo_root()

    def _pin_benchmarks_repo_root(self):
        """Force benchmarks helpers to resolve repo root from configured benchmarks_root."""
        bench_root = Path(self.config.benchmarks_root).expanduser().resolve()

        build_base_images_mod = self._imports.get("swebench_build_base_images")
        if build_base_images_mod is not None:
            try:
                # `benchmarks.swebench.build_base_images._get_repo_root()` uses
                # `git rev-parse --show-toplevel` and can resolve to the wrong
                # repository when RAGEN is launched from another git workspace.
                # Pin it to configured benchmarks root for stable Dockerfile lookup.
                build_base_images_mod._get_repo_root = lambda: bench_root
            except Exception:
                pass

        version_mod = self._imports.get("benchmarks_version")
        if version_mod is not None:
            try:
                version_mod.PROJECT_ROOT = bench_root
            except Exception:
                pass

    def _build_evaluator_if_needed(self):
        if self._evaluator is not None:
            return

        bench_root = Path(self.config.benchmarks_root).expanduser().resolve()
        prompt_path = (
            Path(self.config.prompt_path).expanduser().resolve()
            if self.config.prompt_path
            else (bench_root / "benchmarks" / "swebench" / "prompts" / "default.j2")
        )
        if not prompt_path.exists():
            raise RuntimeError(
                f"SWE-bench prompt template not found: {prompt_path}. "
                "Set custom_envs.OpenHandsSWEBenchVerified.env_config.prompt_path=<path-to-j2>."
            )

        llm_config = self._imports["load_llm_config"](
            str(Path(self.config.llm_config_path).expanduser().resolve())
        )

        output_root = Path(self.config.trajectory_output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        eval_output_dir = output_root / "sdk_runtime"
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        metadata = self._imports["EvalMetadata"](
            llm=llm_config,
            dataset=self.config.dataset,
            dataset_split=self.config.split,
            max_iterations=int(self.config.max_steps),
            eval_output_dir=str(eval_output_dir),
            details={},
            prompt_path=str(prompt_path),
            env_setup_commands=list(self.config.env_setup_commands),
            eval_limit=int(self.config.max_instances),
            n_critic_runs=1,
            critic=self._imports["PassCritic"](),
            selected_instances_file=None,
            max_retries=0,
            workspace_type=str(self.config.workspace_type),
            enable_delegation=bool(self.config.enable_delegation),
            enable_condenser=bool(self.config.enable_condenser),
            condenser_max_size=int(self.config.condenser_max_size),
            condenser_keep_first=int(self.config.condenser_keep_first),
            tool_preset=str(self.config.tool_preset),
            agent_type=str(self.config.agent_type),
        )

        self._metadata = metadata
        self._evaluator = self._imports["SWEBenchEvaluation"](
            metadata=metadata,
            num_workers=1,
        )

    def _load_instances_if_needed(self):
        if self._instances is not None:
            return
        self._instances = self._evaluator.prepare_instances()

    def _build_agent(self):
        if self._imports["is_acp_agent"](self.config.agent_type):
            return self._imports["build_acp_agent"](
                self.config.agent_type,
                self._metadata.llm.model,
            )

        agent_llm = self._imports["build_eval_llm"](self._metadata.llm)
        tools = self._imports["get_tools_for_preset"](
            preset=str(self.config.tool_preset),
            enable_browser=False,
        )

        if self.config.enable_delegation:
            tools.append(self._imports["Tool"](name=self._imports["DelegateTool"].name))

        condenser = None
        if self.config.enable_condenser:
            condenser = self._imports["LLMSummarizingCondenser"](
                llm=self._imports["build_eval_llm"](self._metadata.llm, usage_id="condenser"),
                max_size=int(self.config.condenser_max_size),
                keep_first=int(self.config.condenser_keep_first),
            )

        return self._imports["Agent"](
            llm=agent_llm,
            tools=tools,
            system_prompt_kwargs={"cli_mode": True},
            condenser=condenser,
        )

    # ----------------------------- turn helpers -----------------------------

    def _compose_turn_prompt(self, actions_left: int, per_turn_override: Optional[str]) -> str:
        base_template = self.config.turn_user_prompt_template
        values = {
            "turn_index": self._turn_index,
            "max_steps": int(self.config.max_steps),
            "actions_left": actions_left,
            "instance_id": self._instance_id,
            "state_summary": self._last_state_summary,
            "last_tool_status": self._last_tool_status,
            "last_error": self._last_error,
            "answer_format": str(
                getattr(
                    self.config,
                    "answer_format",
                    "Continue with concrete actions in OpenHands and summarize if done.",
                )
            ),
        }
        base_prompt = safe_format_template(base_template, values).strip()

        # Per-turn override (inject[...]) has higher priority than global override.
        override_template = per_turn_override
        if not override_template:
            override_template = self.config.turn_user_prompt_override
        if not override_template:
            return base_prompt

        override_prompt = safe_format_template(override_template, values).strip()
        if not override_prompt:
            return base_prompt

        mode = str(getattr(self.config, "turn_user_prompt_override_mode", "replace")).strip().lower()
        if mode == "replace":
            return override_prompt
        if mode == "append":
            return f"{base_prompt}\n\n{override_prompt}".strip()
        # Default to prepend semantics for unknown modes.
        return f"{override_prompt}\n\n{base_prompt}".strip()

    def _is_recoverable_turn_error(
        self,
        turn_error: Optional[str],
        status_after: str,
        new_events: List[Any],
    ) -> bool:
        """
        Determine whether a run-level error should allow continuing next turn.

        OpenHands frequently raises a ConversationRunError when one run-cycle
        reaches the per-run iteration limit (max_iteration_per_run=1). In this
        adapter, that condition is expected and should not end the episode.
        """
        if not turn_error:
            return False

        lowered_error = turn_error.lower()
        if any(marker in lowered_error for marker in self.RECOVERABLE_ERROR_MARKERS):
            return True

        for event in new_events:
            payload_text = json.dumps(event_to_jsonable(event)).lower()
            if any(marker in payload_text for marker in self.RECOVERABLE_ERROR_MARKERS):
                return True

        status_text = str(status_after).lower().strip()
        terminal_statuses = {"finished", "completed", "stopped", "cancelled"}
        if status_text and status_text not in terminal_statuses:
            return True

        return False

    def _is_recoverable_error_text(self, text: Optional[str]) -> bool:
        if not text:
            return False
        lowered = str(text).lower()
        return any(marker in lowered for marker in self.RECOVERABLE_ERROR_MARKERS)

    def _conversation_status(self) -> str:
        if self._conversation is None:
            return "not_started"
        status = getattr(self._conversation.state, "execution_status", None)
        if status is None:
            return "unknown"
        value = getattr(status, "value", None)
        return str(value) if value is not None else str(status)

    def _snapshot_usage_totals(self) -> Dict[str, float]:
        if self._conversation is None:
            return {
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "cost_usd": 0.0,
            }

        stats = getattr(self._conversation, "conversation_stats", None)
        if stats is None:
            return {
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "cost_usd": 0.0,
            }

        metrics_obj = None
        get_combined = getattr(stats, "get_combined_metrics", None)
        if callable(get_combined):
            try:
                metrics_obj = get_combined()
            except Exception:
                metrics_obj = None

        return extract_usage_totals(metrics_obj)

    def _snapshot_usage_breakdown(self) -> Dict[str, Any]:
        if self._conversation is None:
            return {
                "totals": {
                    "prompt_tokens": 0.0,
                    "completion_tokens": 0.0,
                    "total_tokens": 0.0,
                    "cost_usd": 0.0,
                },
                "by_usage_id": {},
            }

        stats = getattr(self._conversation, "conversation_stats", None)
        if stats is None:
            return {
                "totals": {
                    "prompt_tokens": 0.0,
                    "completion_tokens": 0.0,
                    "total_tokens": 0.0,
                    "cost_usd": 0.0,
                },
                "by_usage_id": {},
            }

        metrics_obj = None
        get_combined = getattr(stats, "get_combined_metrics", None)
        if callable(get_combined):
            try:
                metrics_obj = get_combined()
            except Exception:
                metrics_obj = None

        return extract_usage_breakdown(metrics_obj)

    def _snapshot_metrics_raw(self) -> Dict[str, Any]:
        if self._conversation is None:
            return {}
        stats = getattr(self._conversation, "conversation_stats", None)
        if stats is None:
            return {}
        get_combined = getattr(stats, "get_combined_metrics", None)
        if not callable(get_combined):
            return {}
        try:
            return to_plain_dict(get_combined())
        except Exception:
            return {}

    def _build_reset_state(self) -> str:
        assert self._instance_data is not None
        problem = str(self._instance_data.get("problem_statement", ""))
        return (
            "========================================================================\n"
            "OPENHANDS SWE-BENCH VERIFIED\n"
            "========================================================================\n"
            f"Instance ID: {self._instance_id}\n"
            f"Repo: {self._instance_data.get('repo', '')}\n"
            f"Base commit: {self._instance_data.get('base_commit', '')}\n\n"
            "Problem statement:\n"
            f"{problem}\n\n"
            f"Action protocol: send `{self.RUN_TURN_ACTION}` each turn (scripted by RAGEN).\n"
            "Prompt injection is applied every turn as a user message to OpenHands."
        )

    def _build_step_state(
        self,
        new_events: List[Any],
        usage_delta: Dict[str, float],
        duration: float,
        status_after: str,
        turn_error: Optional[str],
    ) -> str:
        summary = summarize_events(new_events)
        error_text = turn_error if turn_error else "none"
        return (
            f"Step {self._turn_index}/{self.config.max_steps}\n"
            f"Execution status: {status_after}\n"
            f"Duration: {duration:.3f}s\n"
            f"New events: {len(new_events)}\n"
            f"Prompt tokens (turn): {usage_delta['prompt_tokens']}\n"
            f"Completion tokens (turn): {usage_delta['completion_tokens']}\n"
            f"Total tokens (turn): {usage_delta['total_tokens']}\n"
            f"Cost USD (turn): {usage_delta['cost_usd']}\n"
            f"Last tool status: {self._last_tool_status}\n"
            f"Last error: {error_text}\n\n"
            f"Event summary:\n{summary}"
        )

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(part.strip() for part in parts if str(part).strip())
        if isinstance(content, str):
            return content
        return ""

    @classmethod
    def _extract_message_text(cls, llm_message: Any) -> str:
        if not isinstance(llm_message, dict):
            return ""
        content = llm_message.get("content")
        text = cls._extract_text_from_content(content)
        if text:
            return text
        direct_text = llm_message.get("text")
        return str(direct_text) if direct_text else ""

    def _extract_turn_llm_output(
        self,
        new_events: List[Any],
        turn_error: Optional[str],
        drop_recoverable_errors: bool = False,
    ) -> Dict[str, Any]:
        agent_messages: List[str] = []
        agent_actions: List[Dict[str, Any]] = []
        errors: List[str] = []

        for event in new_events:
            payload = event_to_jsonable(event)
            event_type = str(payload.get("event_type", type(event).__name__))
            source = str(payload.get("source", "")).lower()

            if event_type == "MessageEvent" and source == "agent":
                msg_text = self._extract_message_text(payload.get("llm_message"))
                if msg_text:
                    agent_messages.append(msg_text)
                continue

            if event_type == "ActionEvent" and source == "agent":
                action_entry: Dict[str, Any] = {
                    "tool_name": payload.get("tool_name"),
                    "summary": payload.get("summary"),
                    "action": to_plain_dict(payload.get("action")),
                }
                agent_actions.append({k: v for k, v in action_entry.items() if v not in (None, "", [])})
                continue

            if event_type == "ConversationErrorEvent":
                code = str(payload.get("code", "")).strip()
                detail = str(payload.get("detail", "")).strip()
                if code and detail:
                    errors.append(f"{code}: {detail}")
                elif detail:
                    errors.append(detail)
                elif code:
                    errors.append(code)
                continue

            observation = payload.get("observation")
            if isinstance(observation, dict) and bool(observation.get("is_error")):
                obs_text = self._extract_text_from_content(observation.get("content"))
                tool_name = str(payload.get("tool_name", "")).strip()
                if tool_name and obs_text:
                    errors.append(f"{tool_name}: {obs_text}")
                elif obs_text:
                    errors.append(obs_text)

        if turn_error:
            errors.insert(0, turn_error)

        deduped_errors: List[str] = []
        seen_errors = set()
        for item in errors:
            key = item.strip()
            if not key or key in seen_errors:
                continue
            seen_errors.add(key)
            deduped_errors.append(key)

        if drop_recoverable_errors:
            deduped_errors = [
                e for e in deduped_errors if not self._is_recoverable_error_text(e)
            ]

        return {
            "agent_messages": agent_messages,
            "agent_actions": agent_actions,
            "errors": deduped_errors,
        }

    # ----------------------------- persistence -----------------------------

    def _prepare_output_paths(self):
        output_root = Path(self.config.trajectory_output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        safe_instance_id = str(self._instance_id).replace("/", "_")
        run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_instance_id}_{uuid4().hex[:8]}"
        self._run_dir = output_root / run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._turn_dir = self._run_dir / "turns"
        self._turn_dir.mkdir(parents=True, exist_ok=True)
        if self.config.save_full_payload_json:
            self._turn_full_dir = self._run_dir / "turns_full"
            self._turn_full_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._turn_full_dir = None

        self._trajectory_path = self._run_dir / "trajectory.json"
        self._metrics_summary_path = self._run_dir / "metrics_summary.json"
        self._conversation_full_path = self._run_dir / "conversation_full.json"

    def _persist_turn_payload(self, payload: Dict[str, Any]):
        if not self.config.save_turn_json:
            return
        if self._turn_dir is None:
            return

        turn_idx = int(payload.get("turn_index", 0))
        turn_path = self._turn_dir / f"turn_{turn_idx:04d}.json"
        self._json_dump(turn_path, payload)

    def _persist_turn_full_payload(self, payload: Dict[str, Any]):
        if not self.config.save_full_payload_json:
            return
        if self._turn_full_dir is None:
            return

        turn_idx = int(payload.get("turn_index", 0))
        turn_path = self._turn_full_dir / f"turn_{turn_idx:04d}_full.json"
        self._json_dump(turn_path, payload)

    def _persist_trajectory(self):
        if not self.config.save_trajectory_json:
            return
        if self._trajectory_path is None:
            return
        self._json_dump(self._trajectory_path, self._trajectory)

    def _persist_conversation_full(self, events: List[Any]):
        if not self.config.save_full_payload_json:
            return
        if self._conversation_full_path is None:
            return

        serialized_events = [event_to_jsonable(ev) for ev in events]

        system_prompts: List[Dict[str, Any]] = []
        user_messages: List[Dict[str, Any]] = []
        agent_messages: List[Dict[str, Any]] = []
        environment_events: List[Dict[str, Any]] = []
        agent_actions: List[Dict[str, Any]] = []
        env_observations: List[Dict[str, Any]] = []

        for event in serialized_events:
            event_type = str(event.get("event_type", ""))
            source = str(event.get("source", "")).lower()

            if event_type == "SystemPromptEvent":
                system_prompts.append(event)
            if event_type == "MessageEvent":
                llm_message = event.get("llm_message")
                role = ""
                if isinstance(llm_message, dict):
                    role = str(llm_message.get("role", "")).lower()
                if source == "user" or role == "user":
                    user_messages.append(event)
                elif source == "agent" or role in {"assistant", "agent"}:
                    agent_messages.append(event)
            if source == "environment":
                environment_events.append(event)
            if event_type == "ActionEvent" and source == "agent":
                agent_actions.append(event)
            if event_type == "ObservationEvent" and source == "environment":
                env_observations.append(event)

        conversation_id = None
        execution_status = None
        if self._conversation is not None and getattr(self._conversation, "state", None) is not None:
            conversation_id = getattr(self._conversation.state, "id", None)
            execution_status = self._conversation_status()

        payload = {
            "instance_id": self._instance_id,
            "conversation_id": conversation_id,
            "execution_status": execution_status,
            "event_count": len(serialized_events),
            "history": serialized_events,
            "messages": {
                "system": system_prompts,
                "user": user_messages,
                "agent": agent_messages,
            },
            "agent_actions": agent_actions,
            "environment_observations": env_observations,
            "environment_events": environment_events,
        }
        self._json_dump(self._conversation_full_path, payload)

    def _persist_metrics_summary(self):
        if self._metrics_summary_path is None:
            return

        usage_entries = self._trajectory.get("per_turn_usage", [])
        prompt_sum = 0.0
        completion_sum = 0.0
        total_sum = 0.0
        cost_sum = 0.0
        per_turn_usage: List[Dict[str, Any]] = []
        for idx, usage in enumerate(usage_entries):
            entry = {
                "turn_index": int(usage.get("turn_index", idx + 1)),
                "prompt_tokens": float(usage.get("prompt_tokens", 0.0) or 0.0),
                "completion_tokens": float(usage.get("completion_tokens", 0.0) or 0.0),
                "total_tokens": float(usage.get("total_tokens", 0.0) or 0.0),
                "cost_usd": float(usage.get("cost_usd", 0.0) or 0.0),
            }
            prompt_sum += entry["prompt_tokens"]
            completion_sum += entry["completion_tokens"]
            total_sum += entry["total_tokens"]
            cost_sum += entry["cost_usd"]
            per_turn_usage.append(entry)

        summary = {
            "instance_id": self._instance_id,
            "turns": len(per_turn_usage),
            "done": bool(self._done),
            "success": bool(self._success),
            "last_usage_by_id": to_plain_dict(self._usage_by_id),
            "llm_prompt_tokens_sum": prompt_sum,
            "llm_completion_tokens_sum": completion_sum,
            "llm_total_tokens_sum": total_sum,
            "llm_cost_usd_sum": cost_sum,
            "llm_prompt_tokens_avg": (prompt_sum / len(per_turn_usage)) if per_turn_usage else 0.0,
            "llm_completion_tokens_avg": (completion_sum / len(per_turn_usage)) if per_turn_usage else 0.0,
            "llm_total_tokens_avg": (total_sum / len(per_turn_usage)) if per_turn_usage else 0.0,
            "llm_cost_usd_avg": (cost_sum / len(per_turn_usage)) if per_turn_usage else 0.0,
            "per_turn_usage": per_turn_usage,
        }
        self._json_dump(self._metrics_summary_path, summary)

    def _append_git_patch_if_available(self):
        if self._workspace is None or self._repo_path is None or self._instance_data is None:
            return

        base_commit = self._instance_data.get("base_commit")
        if not base_commit:
            return

        try:
            status_cmd = self._workspace.execute_command(
                f"cd {self._repo_path} ; git --no-pager status --short"
            )
            if int(getattr(status_cmd, "exit_code", 1)) == 0:
                self._trajectory["git_status_short"] = str(getattr(status_cmd, "stdout", ""))
            else:
                self._trajectory["git_status_error"] = str(getattr(status_cmd, "stderr", ""))

            # Align with benchmarks.swebench.run_infer.py:
            # 1) git add -A
            # 2) git commit (best-effort)
            # 3) git diff <base_commit> HEAD
            add_cmd = self._workspace.execute_command(f"cd {self._repo_path} ; git add -A")
            if int(getattr(add_cmd, "exit_code", 1)) != 0:
                self._trajectory["git_add_error"] = str(getattr(add_cmd, "stderr", ""))

            commit_cmd = self._workspace.execute_command(
                (
                    f"cd {self._repo_path} && "
                    "git config --global user.email 'evaluation@openhands.dev' && "
                    "git config --global user.name 'OpenHands Evaluation' && "
                    "git commit --no-verify -m 'patch'"
                )
            )
            if int(getattr(commit_cmd, "exit_code", 1)) != 0:
                self._trajectory["git_commit_info"] = str(
                    getattr(commit_cmd, "stderr", "") or getattr(commit_cmd, "stdout", "")
                )

            diff_cmd = self._workspace.execute_command(
                f"cd {self._repo_path} ; git --no-pager diff --no-color {base_commit} HEAD"
            )
            if int(getattr(diff_cmd, "exit_code", 1)) == 0:
                patch = str(getattr(diff_cmd, "stdout", ""))
                if not patch.strip():
                    # Fallback: if no commit landed, include working tree diff.
                    fallback_diff_cmd = self._workspace.execute_command(
                        f"cd {self._repo_path} ; git --no-pager diff --no-color {base_commit}"
                    )
                    if int(getattr(fallback_diff_cmd, "exit_code", 1)) == 0:
                        patch = str(getattr(fallback_diff_cmd, "stdout", ""))
                self._trajectory["git_patch"] = patch
                if not patch.strip():
                    self._trajectory["git_patch_note"] = (
                        "No modifications detected against base commit (committed or working tree)."
                    )
            else:
                self._trajectory["git_patch_error"] = str(getattr(diff_cmd, "stderr", ""))
        except Exception as exc:
            self._trajectory["git_patch_error"] = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _json_dump(path: Path, payload: Dict[str, Any]):
        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(to_plain_dict(payload), f, ensure_ascii=False, indent=2)
        except Exception:
            # Keep environment running even if artifact persistence fails.
            traceback.print_exc()
