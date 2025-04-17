import hydra
from omegaconf import DictConfig, OmegaConf

from deepnash.config import TrainingConfig, DeepNashAgentConfig
from deepnash.train.rnad import RNaDSolver


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    agent_config = cfg.agents.stratego.agent_config
    print(OmegaConf.to_yaml(agent_config))

    ac = DeepNashAgentConfig.from_config(agent_config)
    print(ac)

    trainig_config = cfg.train

    solver_config = TrainingConfig.from_config(agent_config, trainig_config)
    print(solver_config)
    solver = RNaDSolver(solver_config)


if __name__ == "__main__":
    main()
