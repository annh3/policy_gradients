"""
Baseline class for DDPG, can probably implement
Dueling DDPG on top of it later.
"""

import numpy as np
import torch
import gym
import os
from general import get_logger, Progbar, export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, device, np2torch
from policy import CategoricalPolicy, GaussianPolicy
import pdb

class TabularPolicyGradient(object):
	"""
	Class for implementing hybrid Q-learning
	and policy gradient methods
	"""
	def __init__(self, env, config, seed, logger=None):
		"""
		Initialize Tabular Policy Gradient Class

		Args:
			env: an OpenAI Gym environment
			config: class with hyperparameters
			logger: logger instance from the logging module
		"""

		# directory for training outputs
		if not os.path.exists(config.output_path):
			os.makedirs(config.output_path)

		# store hyperparameters
		self.config = config
		self.seed = seed

		self.logger = logger
		if logger is None:
			self.logger = get_logger(config.log_path)
		self.env = env
		self.env.seed(self.seed)

		# discrete vs continuous action space
		self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
		self.observation_dim = self.env.observation_space.shape[0]
		self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
		self.lr = self.config.learning_rate

		self.init_policy()

	def init_policy(self):
		self.network = build_mlp(self.observation_dim, self.action_dim, self.config.n_layers, self.config.layer_size)
        if self.discrete:
            self.policy = CategoricalPolicy(self.network)
        else:
            self.policy = GaussianPolicy(self.network, self.action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def train(self):
    	"""
    	Performs training
    	"""
    	