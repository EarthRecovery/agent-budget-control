"""
Sokoban grid puzzle environment for multi-turn planning with irreversible dynamics.

Original Source: gym-sokoban
License: MIT

Modifications: Added text-based observation format and custom reward shaping
for LLM agent reinforcement learning.
"""
from .env import SokobanEnv
from .config import SokobanEnvConfig

__all__ = ["SokobanEnv", "SokobanEnvConfig"]
