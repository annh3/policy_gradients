#annhe

import torch
import torch.nn as nn
import torch.distributions as ptd

from network_utils import np2torch, device

class ContinuousPolicy(nn.Module):
	def __init__(self, network):
		nn.Module.__init__(self)
		self.network = network

	def act(self, state):
		"""
		Returns Q values for all actions
		"""
		action = self.network(state)
		return action # note: this may or may not be of the correct form

	"""
	This may or may not be used
	"""
	def update_network(self, state_dictionary):
		self.network.load_state_dict(state_dictionary)


