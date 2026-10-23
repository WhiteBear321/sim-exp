from collections import deque
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import TrainConfig


SCHEMA = "acr-masac-v2"
FIELDS = ("obs_uav", "obs_ugv", "action_uav", "action_ugv", "reward", "next_uav", "next_ugv", "done")


def transition(observation, next_observation, info, reward, done):
    return {"obs_uav": observation[0].tolist(), "obs_ugv": observation[1].tolist(),
            "action_uav": info["uav_action"], "action_ugv": info["ugv_action"], "reward": float(reward),
            "next_uav": next_observation[0].tolist(), "next_ugv": next_observation[1].tolist(),
            "done": bool(done), "event": info["event"], "target_reward": info["target_reward"]}


class Replay:
    def __init__(self, capacity=50000, seed=0):
        self.capacity = capacity
        self.online = deque(maxlen=capacity)
        self.expert = []
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.online) + len(self.expert)

    @staticmethod
    def validate(item):
        dimensions = {"obs_uav": 22, "obs_ugv": 28, "action_uav": 3, "action_ugv": 6, "next_uav": 22, "next_ugv": 28}
        for key, dimension in dimensions.items():
            value = np.asarray(item[key], dtype=float)
            if value.shape != (dimension,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key}: expected {dimension} finite values")
            if key.startswith("action") and np.any(np.abs(value) > 1.000001):
                raise ValueError(f"Invalid normalized action: {key}")
        if not np.isfinite(item["reward"]) or not isinstance(item["done"], bool):
            raise ValueError("Invalid reward or terminal indicator")

    def load_experts(self, directory):
        loaded = []
        for path in sorted(Path(directory).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            metadata = data.get("metadata", {})
            if metadata.get("schema") != SCHEMA or not metadata.get("reward_redistributed"):
                raise ValueError(f"Incompatible demonstration schema: {path.name}")
            for item in data["trajectory"]:
                self.validate(item)
                loaded.append(item)
        if len(loaded) >= self.capacity:
            raise ValueError("Demonstrations must leave capacity for online transitions")
        self.expert = loaded
        self.online = deque(self.online, maxlen=self.capacity - len(loaded))
        return len(loaded)

    def extend(self, episode):
        for item in episode:
            self.validate(item)
            self.online.append(item)

    def sample(self, batch_size, expert_ratio=None):
        if len(self) < batch_size:
            raise ValueError("Insufficient replay transitions")
        if expert_ratio is None or expert_ratio <= 0.0:
            pool = [(item, 0.0) for item in self.online] + [(item, 1.0) for item in self.expert]
            selected = self.rng.sample(pool, batch_size)
        else:
            count = min(len(self.expert), round(batch_size * expert_ratio))
            count = max(count, batch_size - len(self.online))
            selected = [(item, 1.0) for item in self.rng.sample(self.expert, count)]
            selected += [(item, 0.0) for item in self.rng.sample(list(self.online), batch_size - count)]
            self.rng.shuffle(selected)
        batch = {field: np.asarray([item[field] for item, _ in selected], dtype=np.float32) for field in FIELDS}
        batch["expert"] = np.asarray([flag for _, flag in selected], dtype=np.float32)
        return batch

    def state_dict(self):
        return {"online": list(self.online), "expert": self.expert, "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        self.expert = state["expert"]
        self.online = deque(state["online"], maxlen=self.capacity - len(self.expert))
        self.rng.setstate(state["rng"])


class Actor(nn.Module):
    def __init__(self, observations, actions, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(observations, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, actions)
        self.log_std = nn.Linear(hidden, actions)

    def forward(self, observation):
        features = self.net(observation)
        return self.mean(features), self.log_std(features).clamp(-20.0, 2.0)

    def deterministic(self, observation):
        return self(observation)[0].tanh()

    def sample(self, observation):
        mean, log_std = self(observation)
        normal = torch.distributions.Normal(mean, log_std.exp())
        latent = normal.rsample()
        correction = 2.0 * (math.log(2.0) - latent - F.softplus(-2.0 * latent))
        log_probability = (normal.log_prob(latent) - correction).sum(dim=-1, keepdim=True)
        return latent.tanh(), log_probability


class SharedCritic(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        def network():
            return nn.Sequential(nn.Linear(59, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.q1 = network()
        self.q2 = network()

    def forward(self, observation, action):
        value = torch.cat([observation, action], dim=-1)
        return self.q1(value), self.q2(value)


class MASAC:
    def __init__(self, config=None, device="cuda"):
        self.config = config or TrainConfig()
        cfg = self.config
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.actors = nn.ModuleList([Actor(22, 3, cfg.hidden_dim), Actor(28, 6, cfg.hidden_dim)]).to(self.device)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr) for actor in self.actors]
        self.critic = SharedCritic(cfg.hidden_dim).to(self.device)
        self.target = deepcopy(self.critic).requires_grad_(False)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.log_alpha = torch.tensor(math.log(cfg.alpha), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

    @torch.no_grad()
    def select_action(self, observation, phase=3, deterministic=False):
        actions = []
        for index, (actor, value) in enumerate(zip(self.actors, observation)):
            if index == 1 and phase == 1:
                actions.append(np.zeros(6, dtype=np.float32))
            else:
                tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = actor.deterministic(tensor) if deterministic else actor.sample(tensor)[0]
                actions.append(action[0].cpu().numpy())
        return tuple(actions)

    def update(self, replay, step=None, phase=None, guidance=True):
        cfg = self.config
        phase = (3 if step is None else cfg.phase(step)) if phase is None else phase
        ratio, bc_weight = cfg.guidance(step, phase) if guidance else (0.0, 0.0)
        if len(replay) < cfg.batch_size:
            return {}
        batch = replay.sample(cfg.batch_size, ratio)
        tensors = {key: torch.as_tensor(value, device=self.device) for key, value in batch.items()}
        observations = [tensors["obs_uav"], tensors["obs_ugv"]]
        next_observations = [tensors["next_uav"], tensors["next_ugv"]]
        joint_observation = torch.cat(observations, dim=-1)
        joint_next = torch.cat(next_observations, dim=-1)
        joint_action = torch.cat([tensors["action_uav"], tensors["action_ugv"]], dim=-1)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_uav, log_uav = self.actors[0].sample(next_observations[0])
            if phase == 1:
                next_ugv = torch.zeros_like(tensors["action_ugv"])
                log_ugv = torch.zeros_like(log_uav)
            else:
                next_ugv, log_ugv = self.actors[1].sample(next_observations[1])
            q1, q2 = self.target(joint_next, torch.cat([next_uav, next_ugv], dim=-1))
            target = tensors["reward"].unsqueeze(-1) + cfg.gamma * (1.0 - tensors["done"].unsqueeze(-1)) * (torch.minimum(q1, q2) - alpha * (log_uav + log_ugv))
        q1, q2 = self.critic(joint_observation, joint_action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic.requires_grad_(False)
        active = [phase != 2, phase != 1]
        logs = []
        entropy_targets = []
        stats = {"critic_loss": critic_loss.item(), "bc_weight": bc_weight, "expert_ratio": ratio}
        for index in range(2):
            if not active[index]:
                continue
            action, log_probability = self.actors[index].sample(observations[index])
            with torch.no_grad():
                other = self.actors[1 - index].sample(observations[1 - index])[0]
                if phase == 1:
                    other = torch.zeros_like(other)
            actions = [action, other] if index == 0 else [other, action]
            q1, q2 = self.critic(joint_observation, torch.cat(actions, dim=-1))
            actor_loss = (alpha * log_probability - torch.minimum(q1, q2)).mean()
            bc_loss = torch.zeros((), device=self.device)
            mask = tensors["expert"] > 0.5
            if index == 0 and phase == 1 and bc_weight > 0.0 and mask.any():
                prediction = self.actors[0].deterministic(observations[0][mask])
                bc_loss = (prediction - tensors["action_uav"][mask]).square().sum(dim=-1).mean()
                actor_loss += bc_weight * bc_loss
            self.actor_optimizers[index].zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizers[index].step()
            logs.append(log_probability.detach())
            entropy_targets.append(cfg.target_entropies[index])
            stats[f"actor_{index}_loss"] = actor_loss.item()
            if index == 0:
                stats["bc_loss"] = bc_loss.item()
        self.critic.requires_grad_(True)
        entropy_error = sum(logs) + sum(entropy_targets)
        alpha_loss = -(self.log_alpha * entropy_error).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for parameter, target_parameter in zip(self.critic.parameters(), self.target.parameters()):
                target_parameter.lerp_(parameter, cfg.tau)
        stats["alpha"] = self.log_alpha.exp().item()
        return stats

    def save(self, path, step, episode, replay, task_config):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema": SCHEMA, "config": asdict(self.config), "task_config": asdict(task_config),
                "step": step, "episode": episode, "actors": self.actors.state_dict(),
                "critic": self.critic.state_dict(), "target": self.target.state_dict(),
                "actor_optimizers": [optimizer.state_dict() for optimizer in self.actor_optimizers],
                "critic_optimizer": self.critic_optimizer.state_dict(), "log_alpha": self.log_alpha.detach(),
                "alpha_optimizer": self.alpha_optimizer.state_dict(), "replay": replay.state_dict(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        temporary = path.with_suffix(".tmp")
        torch.save(data, temporary)
        temporary.replace(path)

    def restore(self, data, replay):
        self.actors.load_state_dict(data["actors"])
        self.critic.load_state_dict(data["critic"])
        self.target.load_state_dict(data["target"])
        for optimizer, state in zip(self.actor_optimizers, data["actor_optimizers"]):
            optimizer.load_state_dict(state)
        self.critic_optimizer.load_state_dict(data["critic_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(data["log_alpha"])
        self.alpha_optimizer.load_state_dict(data["alpha_optimizer"])
        replay.load_state_dict(data["replay"])
        torch.set_rng_state(data["torch_rng"].cpu())
        if torch.cuda.is_available() and data["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in data["cuda_rng"]])
