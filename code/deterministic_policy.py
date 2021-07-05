#annhe

import torch
import torch.nn as nn
import torch.distributions as ptd

from network_utils import np2torch, device

class ContinuousPolicy(nn.Module):
	def __init__(self, network):
		nn.Module.__init__(self)
		self.network = network

	def act(self, state, network):
		"""
		Returns Q values for all actions
		"""
		raise NotImplementedError

	def update_network(self, state_dictionary):
		self.network.load_state_dict(state_dictionary)


