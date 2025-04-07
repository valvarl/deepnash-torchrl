from gymnasium.envs.registration import register

register(
    id="stratego_gym/Stratego-v0", entry_point="stratego.core.stratego:StrategoEnv"
)
