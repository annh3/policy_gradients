import numpy as np 
import random

class ReplayBuffer(object):
	raise NotImplementedError
	self.num_in_buffer = 0

	def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer
