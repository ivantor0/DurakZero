"""Evaluation utilities for DurakZero models."""

from types import SimpleNamespace
from typing import Optional

import torch

from douzero.dmc.models import Model
from douzero.env import Env


def load_model(checkpoint_path: str, device: str | int | torch.device = 'cpu') -> Model:
    if isinstance(device, torch.device):
        if device.type == 'cpu':
            model_device: str | int = 'cpu'
            map_location = 'cpu'
        else:
            index = device.index or 0
            model_device = index
            map_location = f'cuda:{index}'
    elif isinstance(device, int):
        model_device = device
        map_location = f'cuda:{device}'
    else:
        dev_lower = device.lower()
        if dev_lower.startswith('cpu'):
            model_device = 'cpu'
            map_location = 'cpu'
        elif dev_lower.startswith('cuda'):
            if ':' in dev_lower:
                index_str = dev_lower.split(':', 1)[1] or '0'
            else:
                index_str = '0'
            model_device = int(index_str)
            map_location = f'cuda:{index_str}'
        elif dev_lower.isdigit():
            model_device = int(dev_lower)
            map_location = f'cuda:{dev_lower}'
        else:
            model_device = device
            map_location = 'cpu'
    model = Model(device=model_device)
    state = torch.load(checkpoint_path, map_location=map_location)
    model_state_dict = state.get('model_state_dict', {})
    for position in ['player_0', 'player_1']:
        if position in model_state_dict:
            model.get_model(position).load_state_dict(model_state_dict[position])
    model.eval()
    return model


def evaluate(model_path: str, num_games: int = 100, seed: Optional[int] = None) -> dict:
    device = 'cpu'
    model = load_model(model_path, device=device)
    env = Env(seed=seed)
    wins = {0: 0, 1: 0}
    flags = SimpleNamespace(exp_epsilon=0.0)

    for _ in range(num_games):
        obs = env.reset()
        done = False
        while not done:
            position = obs['position']
            state = torch.from_numpy(obs['state']).to(device)
            action_embeddings = torch.from_numpy(obs['action_embeddings']).to(device)
            with torch.no_grad():
                agent_output = model.act(position, state, action_embeddings, flags=flags)
            action_index = agent_output['action_index']
            action_id = int(obs['legal_actions'][action_index])
            obs, reward, done, info = env.step(action_id)
            if done:
                winner = info.get('winner', 0)
                if winner is None:
                    winner = 0
                wins[winner] += 1
            else:
                assert obs is not None

    env.close()
    total_games = wins[0] + wins[1]
    return {
        'player_0_wins': wins[0],
        'player_1_wins': wins[1],
        'player_0_win_rate': wins[0] / total_games if total_games else 0.0,
        'player_1_win_rate': wins[1] / total_games if total_games else 0.0,
    }
