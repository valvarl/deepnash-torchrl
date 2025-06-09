from gymnasium.envs.registration import register

from stratego.core.config import StrategoConfigBase, StrategoConfig, GameMode
from stratego.core.stratego import StrategoEnv, GamePhase
from stratego.core.primitives import Player, Piece
from stratego.wrappers.cpp_env import StrategoEnvCpp, StrategoConfigCpp

register(id="stratego_gym/Stratego-v0", entry_point="stratego:StrategoEnv")
register(id="stratego_gym/StrategoCpp-v0", entry_point="stratego:StrategoEnvCpp")
