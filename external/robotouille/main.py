import hydra
import os
import json

from omegaconf import DictConfig, OmegaConf

from robotouille.robotouille_simulator import run_robotouille

def _init_vllm_engine_from_cfg(cfg: DictConfig) -> None:
    """Initialize vLLM engine in the main process to avoid fork issues."""
    if not cfg.llm.get("use_vllm", False):
        return
    from agents.prompt_builder.prompt_llm import _get_vllm_engine
    vllm_params = {
        "dtype": cfg.llm.get("vllm_dtype", "auto"),
        "tensor_parallel_size": cfg.llm.get("vllm_tensor_parallel_size", 1),
        "gpu_memory_utilization": cfg.llm.get("vllm_gpu_memory_utilization", 0.9),
        "enforce_eager": cfg.llm.get("vllm_enforce_eager", False),
        "max_model_len": cfg.llm.get("vllm_max_model_len", cfg.llm.get("max_length", 8192)),
        "enable_prefix_caching": cfg.llm.get("vllm_enable_prefix_caching", True),
        "trust_remote_code": cfg.llm.get("vllm_trust_remote_code", True),
    }
    _get_vllm_engine(cfg.llm.llm_model, **vllm_params)

def play(cfg: DictConfig) -> None:
    """Play the game with the given configuration.
    
    Use this function for either
    1. Playing an environment as a human
    2. Running an agent in an environment

    Parameters:
        cfg (DictConfig):
            The Hydra configuration for Robotouille.
    """
    kwargs = OmegaConf.to_container(cfg.game, resolve=True)
    kwargs['llm_kwargs'] = OmegaConf.to_container(cfg.llm, resolve=True)
    environment_name = kwargs.pop('environment_name')
    agent_name = kwargs.pop('agent_name')
    run_robotouille(environment_name, agent_name, **kwargs)

def evaluate(cfg: DictConfig) -> None:
    """Play the game with the given configuration.
    
    Use this function for evaluating an agent on various environments
    and seeds.
    
    Parameters:
        cfg (DictConfig):
            The Hydra configuration for Robotouille.
    """
    log_dir_path = cfg.evaluation.log_dir_path
    os.makedirs(log_dir_path, exist_ok=True)
    results = {}
    environment_names = cfg.evaluation.environment_names
    optimal_steps = cfg.evaluation.optimal_steps
    for environment_name, max_steps in zip(environment_names, optimal_steps):
        for seed in cfg.evaluation.testing_seeds:
            log_subdir = os.path.join(log_dir_path, f"{environment_name}_{seed}")
            basefile_to_subdir_lambda = lambda file_path: os.path.join(log_subdir, os.path.basename(file_path)) if file_path is not None else None
            os.makedirs(log_subdir, exist_ok=True)
            kwargs = OmegaConf.to_container(cfg.game, resolve=True)
            kwargs['max_steps'] = max_steps if kwargs.get('max_steps') is None else kwargs['max_steps']
            kwargs['max_steps'] *= kwargs.get('max_step_multiplier', 1)
            kwargs['seed'] = seed
            kwargs['video_path'] = basefile_to_subdir_lambda(kwargs['video_path'])
            kwargs['llm_kwargs'] = OmegaConf.to_container(cfg.llm, resolve=True)
            kwargs['llm_kwargs']['log_path'] = basefile_to_subdir_lambda(kwargs['llm_kwargs']['log_path'])
            kwargs.pop('environment_name') # Unused for evaluation
            agent_name = kwargs.pop('agent_name')
            done, steps = run_robotouille(environment_name, agent_name, **kwargs)
            results[f"{environment_name}_{seed}"] = {'done': done, 'steps': steps, 'max_steps': kwargs['max_steps']}
    accuracy = sum([result['done'] for result in results.values()]) / len(results)
    average_steps = sum([result['steps'] for result in results.values()]) / len(results)
    results["accuracy"] = accuracy
    results["average_steps"] = average_steps
    results_path = os.path.join(log_dir_path, os.path.basename(cfg.evaluation.results_path))
    with open(results_path, 'w') as f:
        f.write(json.dumps(results, indent=4))

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    _init_vllm_engine_from_cfg(cfg)
    if not cfg.evaluation.evaluate:
        play(cfg)
    else:
        evaluate(cfg)

if __name__ == "__main__":
    main()
