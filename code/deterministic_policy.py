import torch
import torch.nn as nn
import torch.distributions as ptd
from policy import *

from network_utils import np2torch, device
import pdb

class ContinuousPolicy(BasePolicy):
	def __init__(self, network):
		nn.Module.__init__(self)
		self.network = network

	def act(self, observations):
		"""
		Returns Q values for all actions
		"""
		observations = np2torch(observations)
		actions = self.network(observations)
		#pdb.set_trace()
		actions = actions.detach().numpy()
		return actions # note: this may or may not be of the correct form

	"""
	This may or may not be used
	"""
	def update_network(self, state_dictionary):
		self.network.load_state_dict(state_dictionary)


