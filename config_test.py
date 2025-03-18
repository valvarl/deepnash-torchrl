from stratego_gym.envs.config import StrategoConfig
from stratego_gym.envs.primitives import Piece

PIECES_NUM_SCOUT = {
    Piece.SPY: 1,
}

PLACES_TO_DEPLOY_RED_SCOUT = [((4, 2), (4, 2)),]
PLACES_TO_DEPLOY_BLUE_SCOUT = [((0, 2), (0, 2)),]
LAKES_SCOUT = [((2, 2), (2, 2)), ((0, 1), (0, 1)), ((0, 3), (0, 3)), ((4, 1), (4, 1)), ((4, 3), (4, 3))]

scout_config = StrategoConfig(
    height=5,
    width=5,
    p1_pieces_num=PIECES_NUM_SCOUT,
    p1_places_to_deploy=PLACES_TO_DEPLOY_RED_SCOUT,
    p2_places_to_deploy=PLACES_TO_DEPLOY_BLUE_SCOUT,
    lakes=LAKES_SCOUT,
)

