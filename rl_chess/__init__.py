"""
RL chess utilities package.

This module exposes key helpers so training scripts can import from
`rl_chess` without referencing submodules directly.
"""

from .chess_env import ChessEnv
from . import opponents

__all__ = ["ChessEnv", "opponents"]

