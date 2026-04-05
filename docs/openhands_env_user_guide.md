# OpenHands SWE-bench in RAGEN: User Guide

## 1. What this pipeline does
- Runs SWE-bench Verified through RAGEN.
- Uses OpenHands as the only reasoning/model runtime.
- Injects a turn prompt every RAGEN turn.
- Saves turn-level usage/cost and full conversation artifacts.

## 2. Prerequisites

## 2.1 Repositories
This integration requires **two repos**:
- RAGEN repo at `<RAGEN_ROOT>`
- OpenHands benchmarks repo at `<BENCHMARKS_ROOT>`

If you have not cloned them yet:
```bash
git clone <RAGEN_REPO_URL> <RAGEN_ROOT>
git clone https://github.com/All-Hands-AI/benchmarks.git <BENCHMARKS_ROOT>
```

If you use a fork/private mirror of benchmarks, replace the URL accordingly.

## 2.2 Environment
```bash
conda activate ragen
```

## 2.3 Required setup (after both repos are cloned)
```bash
# RAGEN side
cd <RAGEN_ROOT>
git submodule update --init --recursive
pip install -e ./verl --no-deps

# benchmarks/OpenHands side
cd <BENCHMARKS_ROOT>
git submodule update --init --recursive
uv sync
```

## 2.4 Configure OpenHands model/provider
Edit:
- `<BENCHMARKS_ROOT>/.llm_config/example.json`

Then make sure RAGEN points to it in:
- `<RAGEN_ROOT>/config/envs.yaml`
  - `custom_envs.OpenHandsSWEBenchVerified.env_config.llm_config_path`

## 3. Configure prompt injection in files (not CLI)
Edit only in:
- `<RAGEN_ROOT>/config/envs.yaml`
  - `turn_user_prompt_template`
  - `turn_user_prompt_override`
  - `turn_user_prompt_override_mode`
  - `answer_format`

Recommended if you do not want extra prepend text:
```yaml
turn_user_prompt_override: ""
turn_user_prompt_override_mode: prepend
```

Note:
- Old run folders may still show old override values because they store resolved runtime config from those runs.

## 4. Run commands

## 4.1 Single-instance test run
```bash
cd <RAGEN_ROOT>
python -m ragen.eval_api --config-name evaluate_api_openhands \
  es_manager.val.env_groups=1 \
  es_manager.val.env_configs.n_groups='[1]' \
  custom_envs.OpenHandsSWEBenchVerified.env_config.max_instances=1
```

## 4.2 Full configured run (all configured instances)
```bash
cd <RAGEN_ROOT>
python -m ragen.eval_api --config-name evaluate_api_openhands
```

By default, instance count comes from `envs.yaml`:
- `custom_envs.OpenHandsSWEBenchVerified.env_config.max_instances`

## 4.3 Optional: explicitly enforce empty override from CLI
Use only if you want to override file settings at runtime:
```bash
cd <RAGEN_ROOT>
python -m ragen.eval_api --config-name evaluate_api_openhands \
  es_manager.val.env_groups=1 \
  es_manager.val.env_configs.n_groups='[1]' \
  custom_envs.OpenHandsSWEBenchVerified.env_config.max_instances=1 \
  'custom_envs.OpenHandsSWEBenchVerified.env_config.turn_user_prompt_override=""'
```

## 5. Where outputs are saved

## 5.1 Per-run OpenHands artifacts
Default base directory:
- `<RAGEN_ROOT>/results/openhands_eval/`

Each run folder contains:
- `trajectory.json`
- `metrics_summary.json`
- `conversation_full.json`
- `turns/turn_XXXX.json`
- `turns_full/turn_XXXX_full.json` (if full payload enabled)

## 5.2 Eval-level rollouts
Saved under:
- `<RAGEN_ROOT>/results/eval/...`
- JSON summaries in:
  - `val_rollouts_json/metrics_summary.json`
  - `val_rollouts_json/env_rollout_histories.json`
  - `val_rollouts_json/trajectories/*.json`

## 6. Quick troubleshooting

## 6.1 `ModuleNotFoundError: verl.utils`
Cause: `verl` submodule/package not initialized in active env.
Fix:
```bash
cd <RAGEN_ROOT>
git submodule update --init --recursive
pip install -e ./verl --no-deps
```

## 6.2 `Benchmarks SDK workspace is missing or empty`
Cause: benchmarks vendor SDK submodules not initialized/synced.
Fix:
```bash
cd <BENCHMARKS_ROOT>
git submodule update --init --recursive
uv sync
```

## 6.3 Hydra “Ambiguous value …” for prompt text with commas
Cause: comma in CLI override text.
Fix options:
- Set prompt values in `envs.yaml` (recommended), or
- Escape commas in CLI values as `\,`, or
- Avoid commas.

## 6.4 Prompt text still appears after removing from files
Cause: previous run used CLI override; old run artifacts preserve that resolved config.
Fix:
- Re-run without that CLI key.
- Check new run folder’s `trajectory.json` config snapshot.

## 6.5 One-turn error mentions `MaxIterationsReached` / `Remote conversation ended with error`
This can occur because `max_iterations_per_run=1` enforces one OpenHands run cycle per RAGEN turn.
The adapter treats known boundary errors as recoverable and continues future turns.

## 6.6 Token/cost values are zero
Cause is typically upstream provider usage reporting (config/auth/provider behavior), not delta math.
Check:
- `.llm_config` model/provider/API key validity
- OpenHands provider supports usage/cost reporting for that model

## 6.7 `qwen` path appears in logs
That model path is from RAGEN tokenizer bootstrap (`actor_rollout_ref.model.path`) and does not imply OpenHands is using Qwen for reasoning.
OpenHands runtime model comes from benchmarks `.llm_config`.

## 7. Optional sanity checks

## 7.1 Syntax check
```bash
cd <RAGEN_ROOT>
python -m py_compile \
  ragen/env/openhands/config.py \
  ragen/env/openhands/utils.py \
  ragen/env/openhands/env.py \
  ragen/llm_agent/agent_proxy.py \
  ragen/eval_api.py \
  tests/env/test_openhands_utils.py
```

## 7.2 Unit tests for OpenHands utils
```bash
cd <RAGEN_ROOT>
python -m pytest tests/env/test_openhands_utils.py -q
```
