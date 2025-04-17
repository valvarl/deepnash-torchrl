import sys

from omegaconf import DictConfig
from tensordict.nn import TensorDictModule

from .stratego import StrategoAgent, StrategoAgentConfig


class DeepNashAgent(TensorDictModule):
    def __init__(self, *args, **kwargs):
        super(DeepNashAgent).__init__(*args, **kwargs)


def from_config(cfg: DictConfig) -> DeepNashAgent:
    print(sys.modules)
    return sys.modules[cfg.class_name](cfg)
