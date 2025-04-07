from gymnasium.envs.registration import register
from stratego.core.stratego import StrategoEnv

register(id="stratego_gym/Stratego-v0", entry_point="stratego:StrategoEnv")
