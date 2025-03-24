import logging
import os
import time
from typing import Callable

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
import torch
from torch import nn
from torch import optim, Tensor
import wandb

from deep_nash.agent import DeepNashAgent
from learn.config import RNaDConfig
from learn.vtrace import EntropySchedule, get_loss_nerd, get_loss_v, v_trace

class RNaDSolver:
    def __init__(
        self, 
        config: RNaDConfig,
        device=torch.device("cuda"),
        directory_name=None,
        wandb=False,
        use_same_init_net_as=False,
    ):
        self.device = device
        self.config = config
        self.wandb = wandb

        if directory_name is None:
            directory_name = str(int(time.perf_counter()))
        self.directory_name = directory_name

        self.saved_keys = self.__dict__.keys()
        # only the above members are saved in and reloaded from the 'params' object

        saved_runs_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "saved_runs"
        )
        if not os.path.exists(saved_runs_dir):
            os.mkdir(saved_runs_dir)
        self.directory = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "saved_runs", directory_name
        )
        self.use_same_init_net_as = use_same_init_net_as

        self.m = 0
        self.n = 0
        self.learner_steps = 0  # saved in checkpoint
        self.actor_steps = 0 
        self.policy: DeepNashAgent = None
        self.policy_target: DeepNashAgent = None
        self.policy_prev: DeepNashAgent = None
        self.policy_prev_: DeepNashAgent = None

    def __new_net(self) -> nn.Module:
        """
        Initialize a network on self.device with self.net_params
        """
        new_net = DeepNashAgent(device=self.device)
        new_net.eval()
        return new_net

    def __initialize(self):
        """
        Creates a new directory and saves initial network weights in the first update
        """

        logging.info("Initializing R-NaD run: {}".format(self.directory_name))

        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        saved_updates = [
            int(os.path.relpath(f.path, self.directory))
            for f in os.scandir(self.directory)
            if f.is_dir()
        ]
        if not saved_updates:
            params_dict = {key: self.__dict__[key] for key in self.saved_keys}
            torch.save(params_dict, os.path.join(self.directory, "params"))

            os.mkdir(os.path.join(self.directory, "0"))
            self.policy = self.__new_net()
            if self.use_same_init_net_as:
                policy_dir = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "..",
                    "saved_runs",
                    self.use_same_init_net_as,
                    "0",
                    "0",
                )
                checkpoint = torch.load(policy_dir)
                self.policy.load_state_dict(checkpoint["policy"])
                logging.info("Loading init net from {}".format(self.use_same_init_net_as))
            self.policy.train()
            self.policy_target = self.__new_net()
            self.policy_target.load_state_dict(self.policy.state_dict())
            self.policy_prev = self.__new_net()
            self.policy_prev.load_state_dict(self.policy.state_dict())
            self.policy_prev_ = self.__new_net()
            self.policy_prev_.load_state_dict(self.policy.state_dict())

            # The machinery related to updating parameters/learner.
            self._entropy_schedule = EntropySchedule(
                sizes=self.config.entropy_schedule_size,
                repeats=self.config.entropy_schedule_repeats)
            
            # Main network optimizer with Adam and gradient clipping
            self.optimizer = optim.Adam(
                self.policy.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.adam.b1, self.config.adam.b2),
                eps=self.config.adam.eps
            )
            
            # Target network optimizer with SGD for Polyak averaging
            self.optimizer_target = optim.SGD(
                self.policy_target.parameters(),
                lr=self.config.target_network_avg
            )
            self.m = 0
            self.n = 0
            self.__save_checkpoint()
            # this initial checkpoint marks the object has having been initialized; 'saved_updates' will be non-empty now

        else:

            params_dict = torch.load(os.path.join(self.directory, "params"))
            for key, value in params_dict.items():
                if key == "directory_name":
                    params_dict[key] = self.directory_name
                    # renaming the directory will update the saved value for 'directory_name' 
                    continue
                if key == "device":
                    # resuming a run with a new device will move the nets to that device
                    continue
                if torch.is_tensor(value):
                    params_dict[key] = params_dict[key].to(self.device)
                self.__dict__[key] = value
            torch.save(params_dict, os.path.join(self.directory, "params"))
            # overwrite the saved params with the above changes

            self.m = max(saved_updates)
            last_update = os.path.join(self.directory, str(self.m))
            checkpoints = [
                int(os.path.relpath(f.path, last_update))
                for f in os.scandir(last_update)
                if not f.is_dir()
            ]
            self.n = max(checkpoints)
            self.__load_checkpoint(self.m, self.n)
            # use the latest checkpoint

        if self.wandb:
            wandb.init(
                resume=bool(saved_updates),
                project="RNaD",
                config={key: self.__dict__[key] for key in self.saved_keys},
            )
            wandb.run.name = self.directory_name

    def __load_checkpoint(self, m, n):
        """
        Updates the net weights, optimizer state, and certain stat members from those saved in the checkpoint
        """

        saved_dict = torch.load(os.path.join(self.directory, str(m), str(n)))
        self.learner_steps = saved_dict["learner_steps"]
        self.actor_steps = saved_dict["actor_steps"]
        self.policy = self.__new_net()
        self.policy.load_state_dict(saved_dict["policy"])
        self.policy_target = self.__new_net()
        self.policy_target.load_state_dict(saved_dict["policy_target"])
        self.policy_prev = self.__new_net()
        self.policy_prev.load_state_dict(saved_dict["policy_prev"])
        self.policy_prev_ = self.__new_net()
        self.policy_prev_.load_state_dict(saved_dict["policy_prev_"])
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam.b1, self.config.adam.b2),
            eps=self.config.adam.eps
        )
        self.optimizer.load_state_dict(saved_dict["optimizer"])
        self.optimizer_target = optim.SGD(
            self.policy_target.parameters(),
            lr=self.config.target_network_avg
        )
        self.optimizer_target.load_state_dict(saved_dict["optimizer_target"])
        self._entropy_schedule = saved_dict["_entropy_schedule"]

    def __save_checkpoint(self):
        saved_dict = {
            "learner_steps": self.learner_steps,
            "actor_steps": self.actor_steps,
            "policy": self.policy.state_dict(),
            "policy_target": self.policy_target.state_dict(),
            "policy_prev": self.policy_prev.state_dict(),
            "policy_prev_": self.policy_prev_.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_target": self.optimizer_target.state_dict(),
            "_entropy_schedule": self._entropy_schedule,
        }
        if not os.path.exists(os.path.join(self.directory, str(self.m))):
            os.mkdir(os.path.join(self.directory, str(self.m)))
        torch.save(saved_dict, os.path.join(self.directory, str(self.m), str(self.n)))

    def __calc_loss(self, batch: TensorDict, traj_dim=-1):
        logs = {}
        # breakpoint()
        batch = batch.transpose(0, traj_dim)

        def rollout(policy, data):
            output = policy(data)
            return output["policy"], output["value"], output["log_probs"], output["logits"]

        _, v_target, _, _ = rollout(self.policy_target, batch.clone())
        _, _, log_pi_prev, _ = rollout(self.policy_prev, batch.clone())
        _, _, log_pi_prev_, _ = rollout(self.policy_prev_, batch.clone())

        pi, v, log_pi, logit = rollout(self.policy, batch)

        policy_pprocessed = pi # TODO

        alpha, _ = self._entropy_schedule(self.learner_steps)
        log_policy_reg = log_pi - (alpha * log_pi_prev + (1 - alpha) * log_pi_prev_)

        v_target_list, has_played_list, v_trace_policy_target_list = [], [], []
        for player in [1, -1]:
            v_target_, has_played, policy_target_ = v_trace(
                v_target,
                policy_pprocessed,
                log_policy_reg,
                player,
                batch,
                lambda_=1.0,
                c=self.config.c_vtrace,
                rho=torch.inf,
                eta=self.config.eta_reward_transform
            )
            v_target_list.append(v_target_)
            has_played_list.append(has_played)
            v_trace_policy_target_list.append(policy_target_)

        # print("#################### V Debugging ######################")
        # print(v[:, 0].squeeze(-1)[batch["collector"]["mask"][:, 0]])
        # print("-------------------------------------------------------")
        # print(v_target_list[0][:, 0].squeeze(-1)[batch["collector"]["mask"][:, 0]])
        # print(v_target_list[1][:, 0].squeeze(-1)[batch["collector"]["mask"][:, 0]])
        # print("-------------------------------------------------------")
        # print(batch["game_phase"][:, 0][batch["collector"]["mask"][:, 0]])

        loss_v = get_loss_v([v] * 2, v_target_list, has_played_list)

        is_vector = torch.unsqueeze(torch.ones_like(batch["collector"]["mask"]), dim=-1)
        importance_sampling_correction = [is_vector] * 2

        legal_actions = batch["action_mask"]
        legal_actions = legal_actions.reshape(*legal_actions.shape[:-2], -1)
        loss_nerd = get_loss_nerd(
            [logit] * 2, [pi] * 2,
            v_trace_policy_target_list,
            batch["collector"]["mask"],
            batch["cur_player"],
            legal_actions,
            importance_sampling_correction,
            clip=self.config.nerd.clip,
            threshold=self.config.nerd.beta)

        logs["loss_v"] = loss_v.detach().item()
        logs["loss_nerd"] = loss_nerd.detach().item()
        logs["total loss"] = (loss_v + loss_nerd).detach().item()

        return loss_v + loss_nerd, logs
    
    def __step(self, batch):
        loss, logs = self.__calc_loss(batch)

        # breakpoint()

        # TODO: Add logging of the loss and other metrics

        self.optimizer.zero_grad()

        # print("Checking gradients BEFORE backward pass:")
        # for name, param in self.policy.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: {torch.isnan(param.grad).any()} {torch.isinf(param.grad).any()}")

        loss.backward()

        # print("Checking gradients AFTER backward pass:")
        # for name, param in self.policy.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: {torch.isnan(param.grad).any()} {torch.isinf(param.grad).any()}")

        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.clip_gradient)
        self.optimizer.step()

        self.optimizer_target.zero_grad()
        with torch.no_grad():
            for param_t, param_m in zip(self.policy_target.parameters(), self.policy.parameters()):
                # "Difference" = (target - main). This is our "pseudo-gradient".
                param_t.grad = (param_t - param_m).detach().clone()

        self.optimizer_target.step()

        _, update_target_net = self._entropy_schedule(self.learner_steps)
        if update_target_net:
            self.n = 0
            self.m += 1
            self.policy_prev_.load_state_dict(self.policy_prev.state_dict())
            self.policy_prev.load_state_dict(self.policy_target.state_dict())

        self.n += 1
        self.learner_steps += 1

        return logs
    
    def __resume(
        self,
        collector,
        max_updates: int = 10**6,
        checkpoint_mod: int = 1000,
        expl_mod: int = 1,
        log_mod: int = 20,
        evaluate_fn: Callable[[TensorDictModule], dict[str, float]] = None,
    ) -> None:
        """
        Resumes training loop. Terminates when schedule is completed or when max_updates is reached.

        max_updates:
            The max number of updates. Allows partial runs without having to use a short schedule.
        checkpoint_mod:
            Saves a checkpoint after this many steps, always starting at m=0.
        expl_mod:
            Compute NashConv of the target net after this many steps, starting at m=0.
        log_mod:
            Compute logs during learning after this many steps, starting at m=0. 
        """
        for updates, data in range(collector):
            if updates >= max_updates:
                break

            if self.m > sum(self.config.entropy_schedule_repeats):
                break

            if self.m % expl_mod == 0 and self.n == 0 and self.m != 0:
                eval_logs = evaluate_fn(self.policy)
                logging.info("Evaluating results:\n", eval_logs)
                if self.wandb:
                    wandb.log(eval_logs, step=self.learner_steps)

            if self.n == 0:
                delta_m = self._entropy_schedule.schedule[self.m]
                logging.info("m: {}, delta_m: {}".format(self.m, delta_m))

            if self.n % checkpoint_mod == 0:
                self.__save_checkpoint()
            
            logs = self.__step(data)
            if self.n % log_mod == 0 and self.wandb:
                wandb.log(logs, step=self.learner_steps)
    
    def run(
        self, 
        collector, 
        max_updates: int = 10**6, 
        checkpoint_mod: int = 1000, 
        expl_mod: int = 1, 
        log_mod: int = 20,
        evaluate_fn: Callable[[TensorDictModule], dict[str, float]] = None,
    ):
        """
        Either starts or resumes a run.
        """

        self.__initialize()
        self.__resume(
            collector=collector,
            max_updates=max_updates,
            checkpoint_mod=checkpoint_mod,
            expl_mod=expl_mod,
            log_mod=log_mod,
            evaluate_fn=evaluate_fn,
        )
        if self.wandb:
            wandb.finish()
