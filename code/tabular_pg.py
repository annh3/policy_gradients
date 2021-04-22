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
from utils import ReplayBuffer
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

		# only continuous action space
		self.observation_dim = self.env.observation_space.shape[0]
		self.action_dim = self.env.action_space.shape[0]
		self.lr = self.config.learning_rate

		self.init_policy()

	"""
	Trying to figure out best way to update target networks

	policy_net = DQN()
	target_net = DQN()
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()

	self.update_target_net_op = list(
    map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))
	"""

	def init_policy_networks(self):
		"""
		Initialize DETERMINISTIC policy

		Initialize target policy
		"""
		self.policy_network = build_mlp(input_size,output_size,n_layers,size)
		self.policy = ContinuousPolicy(self.policy_network)

		# we never train this one
		self.target_policy_network = build_mlp(input_size,output_size,n_layers,size)
		self.target_policy = ContinuousPolicy(self.target_policy_network)

		self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)

	def update_target_policy(self):
		self.target_policy.update_network(self.policy_network.state_dict())

	def init_q_networks(self):
		"""
		Initialize q network

		* Q(s, u(s))
		* the Q network takes in an observation and also an action

		Initialize target q network
		"""
		self.q_network = build_mlp(input_size+self.action_dim, 1, n_layers, size)
		self.target_q_network = build_mlp(input_size+self.action_dim, 1, n_layers, size)

		self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.q_lr)

	def update_target_q(self):
		self.target_q_network.load_state_dict(self.q_network.state_dict())

    def train(self):
    	"""
    	Performs training

    	Our DDPG implementation uses a trick to improve exploration at the start 
    	f training. For a fixed number of steps at the beginning (set with the 
    	start_steps keyword argument), the agent takes actions which are sampled 
    	from a uniform random distribution over valid actions. After that, it 
    	returns to normal DDPG exploration.
    	"""
    	for t in range(self.config.total_env_interacts):
    		"""
    		# observe state
    		# (unless t < start_steps) select a by perturbing deterministic policy with Gaussian noise and clipping
    		# observe next state, reward, and potential done signal
    		# store (s,a,r,s',d) in replay buffer
    		"""


    		"""
    		if t % self.config.update_every == 0, then perform update
    		"""

    		"""
    		for k in range(self.config.num_update_steps)
    			* sample batch size of transitions from replay buffer
    			* compute targets
    			* update Q-function by one step of gradient descent using /
    				MSE loss: grad { 1/|B| sum Q(s,a) - y(r,s',d)^2 }
				* update policy by one step of gradient ascent using
					grad {1/|B| sum_s Q(s,u(s))}
				* update target networks with polak averaging
					do you have to save and reload weights?
    		"""





