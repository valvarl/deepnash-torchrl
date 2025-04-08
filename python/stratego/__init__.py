from gymnasium.envs.registration import register
from stratego.core.stratego import StrategoEnv
from stratego.wrappers.cpp_env import StrategoEnvCpp

register(id="stratego_gym/Stratego-v0", entry_point="stratego:StrategoEnv")
register(id="stratego_gym/StrategoCpp-v0", entry_point="stratego:StrategoEnvCpp")
