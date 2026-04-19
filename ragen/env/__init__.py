import logging
from importlib import import_module
from typing import Dict, Type


REGISTERED_ENVS: Dict[str, Type] = {}
REGISTERED_ENV_CONFIGS: Dict[str, Type] = {}


def _register_env(
    name: str,
    *,
    config_module: str,
    config_class: str,
    env_module: str,
    env_class: str,
) -> None:
    try:
        cfg_cls = getattr(import_module(config_module, package=__name__), config_class)
        env_cls = getattr(import_module(env_module, package=__name__), env_class)
    except Exception as exc:
        logging.debug("Skipping env registration for %s: %s", name, exc)
        return
    REGISTERED_ENV_CONFIGS[name] = cfg_cls
    REGISTERED_ENVS[name] = env_cls


_register_env(
    "bandit",
    config_module=".bandit.config",
    config_class="BanditEnvConfig",
    env_module=".bandit.env",
    env_class="BanditEnv",
)
_register_env(
    "countdown",
    config_module=".countdown.config",
    config_class="CountdownEnvConfig",
    env_module=".countdown.env",
    env_class="CountdownEnv",
)
_register_env(
    "sokoban",
    config_module=".sokoban.config",
    config_class="SokobanEnvConfig",
    env_module=".sokoban.env",
    env_class="SokobanEnv",
)
_register_env(
    "frozen_lake",
    config_module=".frozen_lake.config",
    config_class="FrozenLakeEnvConfig",
    env_module=".frozen_lake.env",
    env_class="FrozenLakeEnv",
)
_register_env(
    "metamathqa",
    config_module=".metamathqa.config",
    config_class="MetaMathQAEnvConfig",
    env_module=".metamathqa.env",
    env_class="MetaMathQAEnv",
)
_register_env(
    "lean",
    config_module=".lean.config",
    config_class="LeanEnvConfig",
    env_module=".lean.env",
    env_class="LeanEnv",
)
_register_env(
    "deepcoder",
    config_module=".deepcoder.config",
    config_class="DeepCoderEnvConfig",
    env_module=".deepcoder.env",
    env_class="DeepCoderEnv",
)
_register_env(
    "gpqa_main",
    config_module=".gpqa_main.config",
    config_class="GPQAMainEnvConfig",
    env_module=".gpqa_main.env",
    env_class="GPQAMainEnv",
)
_register_env(
    "sudoku",
    config_module=".sudoku.config",
    config_class="SudokuEnvConfig",
    env_module=".sudoku.env",
    env_class="SudokuEnv",
)
_register_env(
    "game_2048",
    config_module=".game_2048.config",
    config_class="Game2048EnvConfig",
    env_module=".game_2048.env",
    env_class="Game2048Env",
)
_register_env(
    "rubikscube",
    config_module=".rubikscube.config",
    config_class="RubiksCube2x2Config",
    env_module=".rubikscube.env",
    env_class="RubiksCube2x2Env",
)
_register_env(
    "token_estimation",
    config_module=".token_estimation.config",
    config_class="TokenEstimationEnvConfig",
    env_module=".token_estimation.env",
    env_class="TokenEstimationEnv",
)
_register_env(
    "money_estimation",
    config_module=".money_estimation.config",
    config_class="MoneyEstimationEnvConfig",
    env_module=".money_estimation.env",
    env_class="MoneyEstimationEnv",
)
_register_env(
    "alfworld",
    config_module=".alfworld.config",
    config_class="AlfredEnvConfig",
    env_module=".alfworld.env",
    env_class="AlfredTXTEnv",
)
_register_env(
    "webshop",
    config_module=".webshop.config",
    config_class="WebShopEnvConfig",
    env_module=".webshop.env",
    env_class="WebShopEnv",
)
_register_env(
    "search",
    config_module=".search.config",
    config_class="SearchEnvConfig",
    env_module=".search.env",
    env_class="SearchEnv",
)
_register_env(
    "robotouille",
    config_module=".robotouille.config",
    config_class="RobotouilleEnvConfig",
    env_module=".robotouille.env",
    env_class="RobotouilleEnv",
)
