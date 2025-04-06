import sys
import os

import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(project_root, "python"))

from stratego.core.config import (
    StrategoConfig, GameMode, 
    PLACES_TO_DEPLOY_RED_ORIGINAL, PLACES_TO_DEPLOY_BLUE_ORIGINAL, LAKES_ORIGINAL
)
from stratego.core.primitives import Piece
from stratego.core.startego import StrategoEnv

@pytest.fixture()
def env_original() -> StrategoEnv:
    config = StrategoConfig.from_game_mode(GameMode.ORIGINAL)
    env = StrategoEnv(config)
    env.reset()
    return env

PLACES_TO_DEPLOY_RED_5x5 = [((3, 0), (4, 4)),]
PLACES_TO_DEPLOY_BLUE_5x5 = [((0, 0), (1, 4)),]
LAKES_CENTRAL_5x5 = [((2, 2), (2, 2))]

def env_5x5():
    def _env_5x5(
        pieces_num: dict[Piece, int], 
        lakes=LAKES_CENTRAL_5x5,
        lakes_mask=None,
        p1_deploy_mask=None,
        p2_deploy_mask=None,
        render_mode=None,
    ) -> StrategoEnv:
        config = StrategoConfig(
            height=5,
            width=5,
            p1_pieces=pieces_num,
            p1_places_to_deploy=PLACES_TO_DEPLOY_RED_5x5 if p1_deploy_mask is None else None,
            p2_places_to_deploy=PLACES_TO_DEPLOY_BLUE_5x5 if p2_deploy_mask is None else None,
            lakes=lakes if lakes_mask is None else None,
            p1_deploy_mask=p1_deploy_mask,
            p2_deploy_mask=p2_deploy_mask,
            lakes_mask=lakes_mask,
        )
        env = StrategoEnv(config, render_mode=render_mode)
        env.reset()
        return env
    return _env_5x5

@pytest.fixture(name="env_5x5")
def env_5x5_fixture():
    return env_5x5()

@pytest.fixture()
def env_10x10():
    def _env_10x10(pieces_num: dict[Piece, int], lakes=LAKES_ORIGINAL) -> StrategoEnv:
        config = StrategoConfig(
            height=10,
            width=10,
            p1_pieces=pieces_num,
            p1_places_to_deploy=PLACES_TO_DEPLOY_RED_ORIGINAL,
            p2_places_to_deploy=PLACES_TO_DEPLOY_BLUE_ORIGINAL,
            lakes=lakes,
        )
        env = StrategoEnv(config)
        env.reset()
        return env
    return _env_10x10
