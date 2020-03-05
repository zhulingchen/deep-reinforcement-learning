import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, use_dueling=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            use_dueling (bool): whether to use the dueling network architecture
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_dueling = use_dueling
        self.feature_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        if use_dueling:
            # state value function V(s)
            self.v_net = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)  # linear activation
            )
            # advantage value function A(s, a)
            self.a_net = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_size)  # linear activation
            )
        else:
            # state-action value function Q(s, a)
            self.q_net = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_size)  # linear activation
            )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        features = self.feature_net(state)
        if self.use_dueling:
            v_val = self.v_net(features)
            a_val = self.a_net(features)
            return v_val + (a_val - a_val.mean())
        else:
            return self.q_net(features)