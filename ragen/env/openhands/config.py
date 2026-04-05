from dataclasses import dataclass, field


@dataclass
class OpenHandsSWEBenchVerifiedEnvConfig:
    """Configuration for OpenHandsSWEBenchVerifiedEnv."""

    # Repository/runtime paths
    benchmarks_root: str = "../benchmarks"
    llm_config_path: str = "../benchmarks/.llm_config/example.json"
    prompt_path: str = ""

    # Dataset selection
    dataset: str = "princeton-nlp/SWE-bench_Verified"
    split: str = "test"
    max_instances: int = 500

    # OpenHands runtime controls
    workspace_type: str = "docker"  # docker | remote
    max_steps: int = 20
    conversation_timeout_sec: int = 3600
    max_iterations_per_run: int = 1

    # Agent construction controls
    tool_preset: str = "default"
    agent_type: str = "default"
    enable_delegation: bool = False
    enable_condenser: bool = True
    condenser_max_size: int = 240
    condenser_keep_first: int = 2

    # Prompt injection
    turn_user_prompt_template: str = (
        "Turn {turn_index}/{max_steps}. Continue solving SWE-bench instance {instance_id}.\n"
        "Turns remaining in this run: {actions_left}.\n"
        "Current state summary:\n{state_summary}\n"
        "Last tool status: {last_tool_status}\n"
        "Last error: {last_error}\n\n"
        "Budget question: Given the current progress and remaining work, "
        "how many additional turns do you estimate are still required to finish?\n"
        "Turns completed so far: {turn_index} / {max_steps}.\n"
        "Turns remaining in this run: {actions_left}.\n"
        "<budget-thinking> [Your turn-budget reasoning for the remaining work] </budget-thinking>\n"
        "<turn_estimation> [Your estimated additional turns needed to complete the task] </turn_estimation>\n"
        "{answer_format}"
    )
    turn_user_prompt_override: str = ""
    answer_format: str = (
        "Continue with concrete actions in OpenHands. If you believe the task is solved, "
        "provide a concise completion summary."
    )
    # How override text is combined with the base template:
    # - replace: override only
    # - prepend: override + base template
    # - append: base template + override
    turn_user_prompt_override_mode: str = "prepend"

    # Output artifacts
    trajectory_output_dir: str = "results/openhands_eval"
    save_trajectory_json: bool = True
    save_full_payload_json: bool = True
    save_turn_json: bool = True

    # Forwarded setup commands for workspace preparation
    env_setup_commands: list[str] = field(
        default_factory=lambda: ["export PIP_CACHE_DIR=~/.cache/pip"]
    )
