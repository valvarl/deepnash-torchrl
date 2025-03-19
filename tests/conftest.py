import pytest

from stratego_gym.envs.config import (
    StrategoConfig, GameMode, 
    PLACES_TO_DEPLOY_RED_ORIGINAL, PLACES_TO_DEPLOY_BLUE_ORIGINAL, LAKES_ORIGINAL
)
from stratego_gym.envs.primitives import Piece
from stratego_gym.envs.startego import StrategoEnv

@pytest.fixture()
def env_original() -> StrategoEnv:
    config = StrategoConfig.from_game_mode(GameMode.ORIGINAL)
    env = StrategoEnv(config)
    env.reset()
    return env

PLACES_TO_DEPLOY_RED_5x5 = [((3, 0), (4, 4)),]
PLACES_TO_DEPLOY_BLUE_5x5 = [((0, 0), (1, 4)),]
LAKES_CENTRAL_5x5 = [((2, 2), (2, 2))]
LAKES_CAVE_5x5 = [((4, 1), (4, 1)), ((3, 0), (3, 0))]

@pytest.fixture()
def env_5x5():
    def _env_5x5(pieces_num: dict[Piece, int], lakes=LAKES_CENTRAL_5x5) -> StrategoEnv:
        config = StrategoConfig(
            height=5,
            width=5,
            p1_pieces_num=pieces_num,
            p1_places_to_deploy=PLACES_TO_DEPLOY_RED_5x5,
            p2_places_to_deploy=PLACES_TO_DEPLOY_BLUE_5x5,
            lakes=lakes,
        )
        env = StrategoEnv(config)
        env.reset()
        return env
    return _env_5x5

@pytest.fixture()
def env_10x10():
    def _env_10x10(pieces_num: dict[Piece, int], lakes=LAKES_ORIGINAL) -> StrategoEnv:
        config = StrategoConfig(
            height=10,
            width=10,
            p1_pieces_num=pieces_num,
            p1_places_to_deploy=PLACES_TO_DEPLOY_RED_ORIGINAL,
            p2_places_to_deploy=PLACES_TO_DEPLOY_BLUE_ORIGINAL,
            lakes=lakes,
        )
        env = StrategoEnv(config)
        env.reset()
        return env
    return _env_10x10
