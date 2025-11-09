"""Models for Durak Deep Monte Carlo training."""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from douzero.env.env import ACTION_VECTOR_LENGTH, STATE_VECTOR_LENGTH


class DurakQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = STATE_VECTOR_LENGTH + ACTION_VECTOR_LENGTH
        hidden = 256
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Model:
    def __init__(self, device=0):
        if not device == "cpu":
            device = "cuda:" + str(device)
        device = torch.device(device)
        self.models = {
            "player_0": DurakQNetwork().to(device),
            "player_1": DurakQNetwork().to(device),
        }

    def act(self, position, state, action_embeddings, flags=None):
        model = self.models[position]
        num_actions = action_embeddings.shape[0]
        state_expanded = state.unsqueeze(0).expand(num_actions, -1)
        inputs = torch.cat([state_expanded, action_embeddings], dim=-1)
        values = model(inputs).squeeze(-1)
        if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
            action_index = torch.randint(num_actions, (1,), device=values.device)[0]
        else:
            action_index = torch.argmax(values)
        return {
            "action_index": int(action_index.item()),
            "values": values.detach(),
        }

    def evaluate(self, position, states, actions):
        model = self.models[position]
        inputs = torch.cat([states, actions], dim=-1)
        return model(inputs)

    def share_memory(self):
        for model in self.models.values():
            model.share_memory()

    def eval(self):
        for model in self.models.values():
            model.eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
