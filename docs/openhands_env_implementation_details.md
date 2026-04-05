# OpenHands SWE-bench Env in RAGEN: Implementation Details

## Scope
This document explains the implementation of the OpenHands + SWE-bench Verified integration inside RAGEN, including architecture, all modified files, and the concrete functions/methods added or changed.

## 1. High-level design

### 1.1 Control model
- OpenHands is the only reasoning/model execution engine.
- RAGEN remains the rollout orchestrator (turn loop, env stepping, aggregation, persistence).
- RAGEN actor is switched to a scripted backend that emits a fixed control action (`RUN_OPENHANDS_TURN`) each turn, so RAGEN does not call an outer LLM for this pipeline.

### 1.2 Turn lifecycle
Per RAGEN turn:
1. Scripted actor emits `RUN_OPENHANDS_TURN` (or optional `inject[...]`).
2. OpenHands env wrapper composes an injected user message for this turn.
3. Wrapper sends that message to OpenHands conversation.
4. Wrapper executes one OpenHands run cycle (`conversation.run(...)`).
5. Wrapper collects newly produced events, computes usage deltas, updates env info, and persists artifacts.

## 2. Files changed and what was implemented

## 2.1 `config/evaluate_api_openhands.yaml` (new)
Purpose: dedicated eval preset for OpenHands-mode runs.

Key config behavior:
- `model_config.model_name: scripted_openhands`
- `model_info.scripted_openhands.provider_name: scripted`
- `model_info.scripted_openhands.generation_kwargs.scripted_action: RUN_OPENHANDS_TURN`
- `agent_proxy.max_turn` linked to env `max_steps`
- validation env tags target `OpenHandsSWEBenchVerified`

No Python functions here; this file wires the run mode.

## 2.2 `config/envs.yaml` (modified)
Purpose: register runtime parameters for the OpenHands env.

Added/maintained block:
- `custom_envs.OpenHandsSWEBenchVerified`
- `env_type: openhands_swebench_verified`
- `max_actions_per_traj: ${.env_config.max_steps}`
- `env_instruction: ""` (outer prompt disabled for scripted mode)
- `env_config` fields for dataset/workspace/prompt injection/persistence

Prompt-specific keys:
- `turn_user_prompt_template`
- `turn_user_prompt_override`
- `turn_user_prompt_override_mode`
- `answer_format`

## 2.3 `ragen/env/__init__.py` (modified)
Purpose: env registry wiring.

Added registration:
- `REGISTERED_ENVS['openhands_swebench_verified'] = OpenHandsSWEBenchVerifiedEnv`
- `REGISTERED_ENV_CONFIGS['openhands_swebench_verified'] = OpenHandsSWEBenchVerifiedEnvConfig`

## 2.4 `ragen/env/openhands/__init__.py` (new)
Purpose: module exports.

Exports:
- `OpenHandsSWEBenchVerifiedEnv`
- `OpenHandsSWEBenchVerifiedEnvConfig`

## 2.5 `ragen/env/openhands/config.py` (new)
Purpose: config dataclass for OpenHands adapter.

Implemented class:
- `OpenHandsSWEBenchVerifiedEnvConfig`

Key fields:
- Paths: `benchmarks_root`, `llm_config_path`, `prompt_path`
- Dataset: `dataset`, `split`, `max_instances`
- Runtime: `workspace_type`, `max_steps`, `conversation_timeout_sec`, `max_iterations_per_run`
- Agent build: `tool_preset`, `agent_type`, delegation/condenser fields
- Injection: `turn_user_prompt_template`, `turn_user_prompt_override`, `turn_user_prompt_override_mode`, `answer_format`
- Persistence: `trajectory_output_dir`, `save_trajectory_json`, `save_full_payload_json`, `save_turn_json`
- Setup forwarding: `env_setup_commands`

## 2.6 `ragen/env/openhands/utils.py` (new)
Purpose: shared helpers for safe formatting, serialization, usage extraction, finish detection, and diagnostics.

Implemented functions/classes:
- `_SafeDict.__missing__`: Returns `""` for missing template placeholders instead of raising `KeyError`.
- `safe_format_template(template, values)`: Safe string formatter used for prompt templates with optional fields.
- `to_plain_dict(obj)`: Converts dataclasses/Pydantic/custom objects into JSON-safe primitives recursively.
- `event_to_jsonable(event)`: Normalizes OpenHands events into a consistent JSON-friendly dict.
- `_dig_number(data, keys)`: Safely extracts numeric values from nested dict paths.
- `_extract_single_usage(metrics)`: Parses token/cost fields from one metrics payload.
- `extract_usage_breakdown(metrics_obj)`: Returns both total usage and per-usage-id usage (for example agent/condenser).
- `extract_usage_totals(metrics_obj)`: Convenience wrapper to get only total usage.
- `compute_usage_delta(previous, current)`: Computes non-negative per-turn deltas from cumulative counters.
- `detect_finish_action(events)`: Heuristic detector for finish/terminate actions in event streams.
- `summarize_events(events, limit=5)`: Produces compact textual summaries of recent events for turn state.
- `extract_event_diagnostics(events)`: Extracts coarse `last_tool_status` and `last_error` signals from events.

## 2.7 `ragen/env/openhands/env.py` (new)
Purpose: core OpenHands SWE-bench environment adapter used by RAGEN.

Implemented class:
- `OpenHandsSWEBenchVerifiedEnv(BaseLanguageBasedEnv, gym.Env)`

Constants:
- `RUN_TURN_ACTION = "RUN_OPENHANDS_TURN"`: Canonical control action emitted by scripted actor each turn.
- `RECOVERABLE_ERROR_MARKERS = (...)`: Error substrings used to classify run-boundary exceptions as recoverable.

Main lifecycle methods:
- `__init__(config=None)`: Initializes runtime state, cached pointers, usage counters, and artifact handles.
- `reset(seed=None, mode=None)`: Builds one SWE-bench instance runtime (workspace, agent, conversation), sends initial instruction, and starts tracking.
- `step(action)`: Executes one controlled OpenHands turn, applies injection, updates usage/status, and persists turn artifacts.
- `render(mode="text")`: Returns the latest text state summary used by RAGEN.
- `close()`: Best-effort cleanup of conversation/workspace resources.

Setup/runtime helper methods:
- `_validate_preflight()`: Validates benchmarks root, vendor SDK availability, and LLM config JSON readability.
- `_ensure_runtime_env_defaults()`: Applies runtime compatibility defaults (for example image tag prefix behavior).
- `_bootstrap_python_path()`: Injects benchmarks and SDK paths (plus `.venv` site-packages) into `sys.path`.
- `_lazy_imports()`: Imports benchmark/OpenHands runtime modules only when needed, with actionable failure messages.
- `_pin_benchmarks_repo_root()`: Forces benchmark helper modules to resolve repo root from configured `benchmarks_root`.
- `_build_evaluator_if_needed()`: Constructs `EvalMetadata` and `SWEBenchEvaluation` once per env instance.
- `_load_instances_if_needed()`: Loads dataset instances lazily through evaluator.
- `_build_agent()`: Builds OpenHands agent (ACP/non-ACP), tools, optional delegation, optional condenser.

Prompt and turn logic methods:
- `_compose_turn_prompt(actions_left, per_turn_override)`: Renders base template and merges override text by mode (`replace`/`prepend`/`append`).
- `_is_recoverable_turn_error(turn_error, status_after, new_events)`: Detects whether turn errors should be treated as recoverable boundary conditions.
- `_is_recoverable_error_text(text)`: Checks a single error string against recoverable markers.
- `_conversation_status()`: Reads normalized execution status from conversation state.

Metrics snapshot methods:
- `_snapshot_usage_totals()`: Pulls cumulative prompt/completion/total/cost counters from OpenHands metrics.
- `_snapshot_usage_breakdown()`: Pulls cumulative totals plus `usage_id` breakdown for diagnostics.
- `_snapshot_metrics_raw()`: Returns full combined metrics snapshot as JSONable dict.

State rendering and extraction methods:
- `_build_reset_state()`: Builds initial environment text (instance metadata + problem statement + control protocol).
- `_build_step_state(new_events, usage_delta, duration, status_after, turn_error)`: Builds per-turn textual state with usage and event summary.
- `_extract_text_from_content(content)`: Extracts text from OpenHands mixed content payloads (list/dict/string).
- `_extract_message_text(llm_message)`: Extracts text specifically from a message object.
- `_extract_turn_llm_output(new_events, turn_error, drop_recoverable_errors=False)`: Builds compact turn output structure (messages/actions/errors).

Persistence methods:
- `_prepare_output_paths()`: Creates per-run directory and file paths (`trajectory`, `turns`, `metrics`, `conversation_full`).
- `_persist_turn_payload(payload)`: Writes compact per-turn JSON entry.
- `_persist_turn_full_payload(payload)`: Writes verbose per-turn JSON entry with raw diagnostics/events.
- `_persist_trajectory()`: Writes the evolving run-level compact trajectory file.
- `_persist_conversation_full(events)`: Writes full event history and grouped views (system/user/agent/environment).
- `_persist_metrics_summary()`: Writes per-turn and aggregated usage/cost summary for the run.
- `_append_git_patch_if_available()`: Collects git status and patch output against base commit when run ends.
- `_json_dump(path, payload)`: Shared safe JSON writer used by all persistence helpers.

### 2.7.1 Important behavior implemented in `step(...)`
- Accepts both control actions:
  - `RUN_OPENHANDS_TURN`
  - `inject[...]` (inline per-turn prompt override)
- Sends per-turn injected prompt to OpenHands via `conversation.send_message(...)`.
- Executes one run cycle via `conversation.run(...)`.
- Computes turn usage/cost as deltas from cumulative snapshots.
- Persists compact and verbose turn artifacts.
- Applies recoverable error logic for frequent run-boundary errors (`MaxIterationsReached`, conversation boundary errors) so episodes can continue next turn.

### 2.7.2 Patch collection behavior
On episode end, `_append_git_patch_if_available()` aligns to benchmark style:
1. `git add -A`
2. best-effort `git commit --no-verify -m 'patch'`
3. `git diff <base_commit> HEAD`
4. fallback `git diff <base_commit>` if needed

Patch is saved into `trajectory.json` (`git_patch`, plus diagnostic keys).

## 2.8 `ragen/llm_agent/agent_proxy.py` (modified)
Purpose: add scripted actor path for OpenHands mode.

Updated class/methods:
- `ApiCallingWrapperWg.__init__(...)`
  - supports `provider_name == "scripted"`
  - reads `generation_kwargs.scripted_action`
  - builds deterministic response text compatible with think/no-think formatting.
- `ApiCallingWrapperWg.generate_sequences(...)`
  - when scripted mode is active, emits scripted response for each input turn without API model calls.

## 2.9 `ragen/eval_api.py` (modified)
Purpose: persist eval JSON artifacts in addition to pkl.

Added helper functions:
- `_normalize_output_cfg(config)`
- `_build_save_path(config, output_cfg, timestamp)`
- `_to_jsonable(obj)`
- `_save_rollout_json_artifacts(save_path, rollouts)`

`main(config)` updated to:
- save pkl rollout,
- then persist `val_rollouts_json` outputs (summary + per-instance trajectories).

## 2.10 `tests/env/test_openhands_utils.py` (new)
Purpose: unit tests for OpenHands utility logic.

Covers:
- safe template formatting fallback behavior,
- usage extraction from cumulative fields,
- usage breakdown aggregation,
- non-negative delta computation,
- finish-action detection heuristics.

## 3. Prompt injection design details

## 3.1 Composition source
Prompt text is composed in `OpenHandsSWEBenchVerifiedEnv._compose_turn_prompt(...)`.

Base:
- `turn_user_prompt_template`

Optional override precedence:
1. per-turn `inject[...]` from action text
2. `turn_user_prompt_override` from env config

Merge mode (`turn_user_prompt_override_mode`):
- `replace`, `prepend`, `append`

Template placeholders available:
- `{turn_index}`, `{max_steps}`, `{actions_left}`, `{instance_id}`
- `{state_summary}`, `{last_tool_status}`, `{last_error}`
- `{answer_format}`

## 3.2 Injection target
Injection is sent as a **user message into OpenHands conversation**, not to RAGEN’s outer context builder in scripted mode.

## 4. Usage/cost collection details

Source:
- OpenHands `conversation.conversation_stats.get_combined_metrics()`

Extraction:
- `extract_usage_breakdown(...)` and `extract_usage_totals(...)`
- supports:
  - `accumulated_token_usage`
  - `accumulated_cost`
  - `usage_to_metrics` (per usage-id), then aggregated

Turn-level values:
- computed by `compute_usage_delta(previous, current)`
- exported to `info` keys consumed by RAGEN aggregation:
  - `llm_prompt_tokens`
  - `llm_completion_tokens`
  - `llm_total_tokens`
  - `llm_cost_usd`

## 5. Artifact model and file layout

Per-run dir under `trajectory_output_dir` (default `results/openhands_eval`):
- `trajectory.json`
  - compact conversation history by turn (`llm_input`, `llm_output`, `usage`, status)
  - per-turn usage table
  - run metadata/status
  - optional git patch diagnostics
- `metrics_summary.json`
  - summed and average usage + `per_turn_usage`
- `conversation_full.json`
  - full serialized event history
  - grouped message views (`system/user/agent`)
  - agent action events
  - environment observations/events
- `turns/turn_XXXX.json`
  - compact per-turn record
- `turns_full/turn_XXXX_full.json` (if enabled)
  - raw/verbose diagnostics including `new_events`, usage snapshots, timing and status transitions

Eval-level exports from `eval_api.py`:
- `results/eval/.../val_rollouts_json/metrics_summary.json`
- `results/eval/.../val_rollouts_json/env_rollout_histories.json`
- `results/eval/.../val_rollouts_json/trajectories/*.json`

## 6. Known operational constraints
- `max_iterations_per_run=1` intentionally enforces one OpenHands cycle per RAGEN turn.
- Boundary errors related to per-run iteration limits are treated as recoverable in compact status handling.
- If provider usage is missing upstream, turn usage can appear as zero despite successful actions.

## 7. Validation status
- Syntax checks and utility unit tests are supported in-repo (`tests/env/test_openhands_utils.py`).
- Integration runs persist both compact and full artifacts for inspection.
