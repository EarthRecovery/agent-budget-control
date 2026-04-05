# OpenHands in RAGEN: Detailed Integration Notes

## 1) High-Level Goal

Integrate OpenHands as a RAGEN environment so that:

- SWE-bench Verified runs through RAGEN eval orchestration.
- OpenHands is the only reasoning agent.
- RAGEN injects a user prompt every turn.
- Per-turn token/cost metrics are captured and persisted.
- Full per-turn payload/event artifacts are saved.

## 2) High-Level Architecture

### 2.1 Control split

- RAGEN:
  - turn scheduler (`LLMAgentProxy` / `EnvStateManager`)
  - per-turn injection template rendering
  - metric aggregation
  - rollout persistence
- OpenHands:
  - actual agent conversation + tool execution + model calls

### 2.2 Why scripted actor exists

RAGEN’s rollout loop expects an “actor output” every turn.  
For OpenHands integration, we provide a scripted actor backend that emits a control action (`RUN_OPENHANDS_TURN`) instead of calling any external LLM from RAGEN.

This keeps the existing RAGEN loop intact while ensuring OpenHands is the only model owner.

## 3) Files Added / Changed

### 3.1 New OpenHands environment module

- `ragen/env/openhands/config.py`
  - `OpenHandsSWEBenchVerifiedEnvConfig`:
    - `benchmarks_root`, `llm_config_path`, `prompt_path`
    - dataset/split/max_instances
    - `workspace_type`, `max_steps`, `conversation_timeout_sec`
    - per-turn injection template and override
    - persistence flags (`save_trajectory_json`, `save_full_payload_json`, `save_turn_json`)

- `ragen/env/openhands/utils.py`
  - safe template format helper
  - event serialization helper
  - usage extraction:
    - from `accumulated_token_usage` / `accumulated_cost`
    - from `usage_to_metrics` breakdown (agent/condenser/etc)
  - cumulative-to-delta conversion
  - finish action detection helper
  - event summary + diagnostics extraction

- `ragen/env/openhands/env.py`
  - `OpenHandsSWEBenchVerifiedEnv`
  - `reset()`:
    - preflight checks (benchmarks path, SDK availability, llm config JSON)
    - lazy imports from benchmarks/OpenHands runtime
    - evaluator setup and instance loading
    - workspace preparation and conversation initialization
    - initial instruction dispatch
    - baseline metrics snapshots and output directory setup
    - runtime compatibility defaults:
      - pins benchmarks repo-root helpers to `benchmarks_root` (stable Dockerfile lookup)
      - sets `IMAGE_TAG_PREFIX` from SDK short SHA when unset (avoids hash-tag mismatch)
  - `step(action)`:
    - accepts `RUN_OPENHANDS_TURN` (and optional `inject[...]`)
    - composes per-turn injected message
    - runs one OpenHands cycle (`conversation.run(...)`)
    - captures new events since prior offset
    - computes per-turn usage/cost delta from cumulative snapshots
    - updates `info` metrics keys consumed by RAGEN aggregation
    - persists per-turn and per-instance JSON artifacts

### 3.2 Environment registration

- `ragen/env/__init__.py`
  - registers env type: `openhands_swebench_verified`

### 3.3 Config wiring

- `config/envs.yaml`
  - added `custom_envs.OpenHandsSWEBenchVerified`
  - `max_actions_per_traj: ${.env_config.max_steps}` (single source of truth)

- `config/evaluate_api_openhands.yaml`
  - scripted backend model:
    - `provider_name: scripted`
    - `scripted_action: RUN_OPENHANDS_TURN`
  - `agent_proxy.max_turn: ${custom_envs.OpenHandsSWEBenchVerified.env_config.max_steps}`
  - val tags target OpenHands env

### 3.4 Scripted actor backend

- `ragen/llm_agent/agent_proxy.py`
  - `ApiCallingWrapperWg` supports `provider_name == "scripted"`
  - returns deterministic control output without API inference

### 3.5 Eval artifact export

- `ragen/eval_api.py`
  - persists rollout pkl
  - emits JSON artifacts:
    - `metrics_summary.json`
    - `env_rollout_histories.json`
    - per-instance trajectory JSON files

### 3.6 Tests

- `tests/env/test_openhands_utils.py`
  - coverage for:
    - template formatting
    - usage extraction
    - usage delta
    - finish detection heuristics

## 4) Turn Pipeline + Prompt Injection

### 4.1 Per-turn execution pipeline

For each RAGEN turn in OpenHands mode:

1. Scripted actor emits one control action (default `RUN_OPENHANDS_TURN`).
2. OpenHands env wrapper receives the action in `OpenHandsSWEBenchVerifiedEnv.step(...)`.
3. Env composes the per-turn user prompt in `_compose_turn_prompt(...)`.
4. Env sends the composed prompt to OpenHands with `conversation.send_message(...)`.
5. Env runs one OpenHands cycle with `conversation.run(...)`.
6. Env collects new events, computes usage/cost deltas, updates `info`, and persists JSON artifacts.

So OpenHands remains the only reasoning agent; RAGEN controls turn scheduling and injection.

### 4.2 Prompt composition semantics (current implementation)

Prompt composition always starts from the base template:

- `turn_user_prompt_template`

Then optional override text is selected with this priority:

1. per-turn `inject[...]` action payload (if used)
2. `turn_user_prompt_override`

Then merge mode decides how override and base are combined:

- `replace`: `override`
- `prepend`: `override + base`
- `append`: `base + override`

Template variables available in both base and override text:

- `turn_index`, `max_steps`, `actions_left`, `instance_id`
- `state_summary`, `last_tool_status`, `last_error`

Current default behavior:

- Base template already includes the turn-budget question and required tags:
  - `<budget-thinking> ... </budget-thinking>`
  - `<turn_estimation> ... </turn_estimation>`
- Default `turn_user_prompt_override` is empty.
- No default prompt text mentions `RAGEN`.

### 4.3 Where to edit prompt text

Main config location:

- `config/envs.yaml` under `custom_envs.OpenHandsSWEBenchVerified.env_config`:
  - `turn_user_prompt_template`
  - `turn_user_prompt_override`
  - `answer_format`
  - `turn_user_prompt_override_mode` (`replace` / `prepend` / `append`)

Low-level code default (only if you want to change repo defaults):

- `ragen/env/openhands/config.py`
  - `OpenHandsSWEBenchVerifiedEnvConfig.turn_user_prompt_*`

## 5) Token/Cost Metrics Semantics

The env records cumulative snapshots each turn and computes deltas:

- `llm_prompt_tokens`
- `llm_completion_tokens`
- `llm_total_tokens`
- `llm_cost_usd`

Metrics source:

- OpenHands `conversation.conversation_stats.get_combined_metrics()`
- Extracted from:
  - `accumulated_token_usage`
  - `accumulated_cost`
  - optionally `usage_to_metrics` per usage id (agent/condenser/etc), then summed

Important troubleshooting note:

- If OpenHands returns zero usage in `accumulated_token_usage` and empty `token_usages`,
  RAGEN will also show zero deltas for that turn.
- This is typically an upstream OpenHands/LLM usage-reporting issue (for example model/provider
  config mismatch or no usage emitted by provider), not a RAGEN delta computation bug.

## 6) Persistence Layout

Per instance run directory (`trajectory_output_dir`, default `results/openhands_eval`):

- `trajectory.json`
  - `conversation_history`: compact per-turn records
    - `llm_input`
    - `llm_output` (agent messages/actions/errors for that turn)
    - `usage` (turn-only usage)
  - `per_turn_usage`: normalized turn usage table for aggregation
  - run-level status + metadata
- `metrics_summary.json`
  - totals/averages + per-turn rows
- `turns/turn_XXXX.json`
  - compact turn-only JSON (`llm_input`, `llm_output`, `usage`, `status`)
- `turns_full/turn_XXXX_full.json` (when `save_full_payload_json=true`)
  - verbose/raw diagnostics:
    - event offsets
    - pre/post usage snapshots
    - full `new_events`
    - metrics snapshots

Patch persistence:
- `trajectory.json` now captures:
  - `git_status_short`
  - `git_patch` aligned with OpenHands benchmark flow:
    - `git add -A`
    - best-effort `git commit --no-verify -m "patch"`
    - `git diff <base_commit> HEAD`
    - fallback to `git diff <base_commit>` if commit does not land
  - `git_patch_note` if no modifications are detected

Eval-level directory (from `output.dir`/`output.filename`):

- `val_rollouts_json/metrics_summary.json`
- `val_rollouts_json/env_rollout_histories.json`
- `val_rollouts_json/trajectories/*.json`

## 7) Setup and Run Commands

Set these once for your machine:

```bash
export RAGEN_ROOT=/path/to/RAGEN
export BENCHMARKS_ROOT=/path/to/benchmarks
```

### 7.1 Required setup

```bash
# Activate your env first
conda activate ragen

# RAGEN submodules (verl imports depend on this)
cd "$RAGEN_ROOT"
git submodule update --init --recursive
pip install -e ./verl --no-deps

# Benchmarks + software-agent-sdk
cd "$BENCHMARKS_ROOT"
git submodule update --init --recursive
uv sync
```

### 7.2 Smoke run (single instance)

```bash
cd "$RAGEN_ROOT"
python -m ragen.eval_api --config-name evaluate_api_openhands \
  es_manager.val.env_groups=1 \
  es_manager.val.env_configs.n_groups='[1]' \
  custom_envs.OpenHandsSWEBenchVerified.env_config.max_instances=1 \
  custom_envs.OpenHandsSWEBenchVerified.env_config.max_steps=1
```

### 7.3 Run with injected prompt + original prompt every turn

Use `prepend` mode. If your override text has commas, escape each comma (`\,`) to avoid Hydra ambiguity:

```bash
cd "$RAGEN_ROOT"
python -m ragen.eval_api --config-name evaluate_api_openhands \
  es_manager.val.env_groups=1 \
  es_manager.val.env_configs.n_groups='[1]' \
  custom_envs.OpenHandsSWEBenchVerified.env_config.max_instances=1 \
  'custom_envs.OpenHandsSWEBenchVerified.env_config.turn_user_prompt_override=Prioritize minimal\, test-backed patches.' \
  custom_envs.OpenHandsSWEBenchVerified.env_config.turn_user_prompt_override_mode=prepend
```

The same command in a single line:

```bash
python -m ragen.eval_api --config-name evaluate_api_openhands es_manager.val.env_groups=1 es_manager.val.env_configs.n_groups='[1]' custom_envs.OpenHandsSWEBenchVerified.env_config.max_instances=1 'custom_envs.OpenHandsSWEBenchVerified.env_config.turn_user_prompt_override=Prioritize minimal\, test-backed patches.' custom_envs.OpenHandsSWEBenchVerified.env_config.turn_user_prompt_override_mode=prepend
```

Current default is already budget-injection + prepend, so if you do not override it,
each turn includes:
- an explicit budget question prompt
- budget-thinking block
- token/cost estimation block
- then the base turn template context

### 7.4 Tested one-instance command (budget injection default)

This exact command was validated to:
- start one SWE-bench instance
- execute one OpenHands turn
- persist injected prompt in trajectory JSON

```bash
cd "$RAGEN_ROOT"
OPENHANDS_SUPPRESS_BANNER=1 \
HF_HOME=/tmp/hf_cache \
HF_DATASETS_CACHE=/tmp/hf_cache/datasets \
python -m ragen.eval_api --config-name evaluate_api_openhands \
  es_manager.val.env_groups=1 \
  es_manager.val.env_configs.n_groups='[1]' \
  custom_envs.OpenHandsSWEBenchVerified.env_config.max_instances=1 \
  custom_envs.OpenHandsSWEBenchVerified.env_config.max_steps=1
```

Artifacts from the validated run:
- `results/openhands_eval/<run_id>/trajectory.json`
- `results/openhands_eval/<run_id>/turns/turn_0001.json`
- `results/openhands_eval/<run_id>/metrics_summary.json`

If your `.llm_config` still contains placeholder credentials (for example `YOUR_KEY_HERE`), the turn will end with OpenHands authentication errors and token/cost remains zero until real API credentials are configured.

### 7.5 Other useful prompt modes

```bash
# override only
custom_envs.OpenHandsSWEBenchVerified.env_config.turn_user_prompt_override_mode=replace

# original + override
custom_envs.OpenHandsSWEBenchVerified.env_config.turn_user_prompt_override_mode=append
```

### 7.6 Optional: force per-turn inject action from scripted actor

If you want the scripted actor to emit `inject[...]` directly each turn:

```bash
python -m ragen.eval_api --config-name evaluate_api_openhands \
  model_info.scripted_openhands.generation_kwargs.scripted_action='inject[Prioritize failing tests first.]'
```

### 7.7 Other useful overrides

```bash
custom_envs.OpenHandsSWEBenchVerified.env_config.max_steps=40
custom_envs.OpenHandsSWEBenchVerified.env_config.save_full_payload_json=true
custom_envs.OpenHandsSWEBenchVerified.env_config.save_turn_json=true
custom_envs.OpenHandsSWEBenchVerified.env_config.trajectory_output_dir=results/openhands_eval_custom
```

### 7.8 Troubleshooting: one-turn stop + missing injected prompt

If an episode stops after one turn with a `ConversationRunError` and message similar to
`Remote conversation ended with error`, this is now handled as a recoverable per-run boundary
condition in `ragen/env/openhands/env.py` (instead of ending the whole episode immediately).

Where to verify injection:
- `results/openhands_eval/<run_id>/trajectory.json`
  - `conversation_history[i].llm_input` (full composed per-turn prompt)
- `results/openhands_eval/<run_id>/turns/turn_XXXX.json`
  - `llm_input`
- `results/openhands_eval/<run_id>/turns_full/turn_XXXX_full.json` (when enabled)
  - full `new_events`
- In `new_events`, your injected prompt is typically the second `"source": "user"` message
  (the first user message is the SWE-bench initial instruction).

Why you may see `MaxIterationsReached: ... limit (1)`:
- `max_iterations_per_run` is intentionally set to `1` to enforce exactly one OpenHands run cycle per RAGEN turn.
- This is expected boundary behavior, not a hard failure by itself.
- The adapter treats this class of run error as recoverable and continues next turn unless another terminal condition is met.

If Hydra complains about commas in override text (`Ambiguous value ...`), either:
- escape commas: `\,`
- or avoid commas in the override text.

If you see HuggingFace cache lock or read-only errors, run with writable cache paths:

```bash
HF_HOME=/tmp/hf_cache HF_DATASETS_CACHE=/tmp/hf_cache/datasets <your command>
```

## 8) Test Commands

### 8.1 Fast syntax check

```bash
python -m py_compile \
  ragen/env/openhands/config.py \
  ragen/env/openhands/utils.py \
  ragen/env/openhands/env.py \
  ragen/llm_agent/agent_proxy.py \
  ragen/eval_api.py \
  tests/env/test_openhands_utils.py
```

### 8.2 Unit test (OpenHands utils)

```bash
cd "$RAGEN_ROOT"
python -m pytest tests/env/test_openhands_utils.py -q
```

Note: if you hit `ModuleNotFoundError: verl.utils`, initialize RAGEN submodules first (Section 7.1).

## 9) Important Clarification About `env_instruction`

`env_instruction` in `config/envs.yaml` is part of RAGEN’s generic context builder path.  
In this OpenHands integration, we run the scripted actor mode (no outer LLM inference), so `env_instruction` is effectively ignored for reasoning.

OpenHands remains the only agent receiving real per-turn injected prompts and producing actions/events.
