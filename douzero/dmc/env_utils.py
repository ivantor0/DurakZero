"""Environment utilities for Durak training."""

import torch


def _format_observation(obs, device):
    if not device == "cpu":
        device = "cuda:" + str(device)
    device = torch.device(device)

    position = obs["position"]
    state = torch.from_numpy(obs["state"]).to(device)
    action_embeddings = torch.from_numpy(obs["action_embeddings"]).to(device)
    legal_actions = obs["legal_actions"].tolist()
    return position, {
        "state": state,
        "action_embeddings": action_embeddings,
        "legal_actions": legal_actions,
    }


class Environment:
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self):
        obs = self.env.reset()
        position, obs = _format_observation(obs, self.device)
        self.episode_return = torch.zeros(1, 1)
        initial_done = torch.ones(1, 1, dtype=torch.bool)
        return position, obs, {
            "done": initial_done,
            "episode_return": self.episode_return,
        }

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.episode_return += reward
        episode_return = self.episode_return

        if done:
            obs = self.env.reset()
            self.episode_return = torch.zeros(1, 1)

        position, obs = _format_observation(obs, self.device)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        return position, obs, {
            "done": done,
            "episode_return": episode_return,
        }

    def close(self):
        self.env.close()
