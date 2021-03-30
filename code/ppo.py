import numpy as np
import torch
import gym
import os
from general import get_logger, Progbar, export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, device, np2torch
from policy import CategoricalPolicy, GaussianPolicy
from policy_gradient import PolicyGradient
import pdb

class PPO(PolicyGradient):
    """
    Class for implementing PPO algorithm on
    """

    # def __init__(self, env, config, seed, logger=None):
    def __init__(self, env, config, seed, logger=None):
        self.alg_name = "ppo"

        PolicyGradient.__init__(self, env, config, seed, logger)

    # override inherited method
    def update_policy(self, observations, actions, advantages):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size]

        Perform one update on the policy using the provided data.
        To compute the loss, you will need the log probabilities of the actions
        given the observations. Note that the policy's action_distribution
        method returns an instance of a subclass of
        torch.distributions.Distribution, and that object can be used to
        compute log probabilities.
        See https://pytorch.org/docs/stable/distributions.html#distribution

        Note:
        PyTorch optimizers will try to minimize the loss you compute, but you
        want to maximize the policy's performance.
        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        #######################################################
        #########   YOUR CODE HERE - 5-7 lines.    ############
        self.optimizer.zero_grad()
        res = self.policy.action_distribution(observations).log_prob(actions) 
        #res = self.policy.action_distribution(observations)
        #pdb.set_trace()
        loss = -(res * advantages).mean()
        loss.backward()
        self.optimizer.step()

        #######################################################
        #########          END YOUR CODE.          ############

