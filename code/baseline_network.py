import numpy as np
import torch
import torch.nn as nn
from network_utils import build_mlp, device, np2torch
from collections import OrderedDict


class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network
    """
    def __init__(self, env, config):

        super().__init__()
        self.config = config
        self.env = env
        self.baseline = None
        self.lr = self.config.learning_rate
        self.observation_dim = self.env.observation_space.shape[0]
        self.network = build_mlp(self.observation_dim, 1, self.config.n_layers, self.config.layer_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = self.lr)

    def save_weights(self):
        weights = OrderedDict()
        for name, param in self.network.named_parameters():
            weights[name] = param.detach().numpy()
        return weights

    def calculate_deltas(self, weights_1, weights_2):
        deltas = OrderedDict()
        for name in weights_1:
            delta = np.linalg.norm(weights_1[name] - weights_2[name])
            deltas[name] = delta
        return deltas

    def print_weights(self):
        return
        # for param in self.network.parameters():
        #     print(param)
    def print_params(self):
        for name, param in self.network.named_parameters():
            print(name)
            print(param.requires_grad)

    def forward(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]
        """
        output = torch.squeeze(self.network(observations))
        assert output.ndim == 1
        return output

    def calculate_advantage(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]

        """
        observations = np2torch(observations)
        res = self.forward(observations)
        res = res.detach().numpy()
        advantages = returns - res
        assert(isinstance(advantages, np.ndarray))
        return advantages

    def update_baseline(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size], containing all discounted
                future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        """
        returns = np2torch(returns)
        observations = np2torch(observations)
        self.optimizer.zero_grad()
        preds = self.forward(observations)
        loss = ((returns - preds)**2).mean()
        loss.backward()
        #print("Loss: ", loss)
        self.optimizer.step()
