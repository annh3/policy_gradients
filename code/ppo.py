import numpy as np
import torch
import gym
import os
from general import get_logger, Progbar, export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, device, np2torch, print_network_grads
from policy import CategoricalPolicy, GaussianPolicy
from policy_gradient import PolicyGradient
import pdb

class PPO(PolicyGradient):
    """
    Class for implementing PPO algorithm on
    """

    # def __init__(self, env, config, seed, logger=None):
    def __init__(self, env, config, seed, epsilon_clip=0.2, logger=None):
        self.alg_name = "ppo"
        self.epsilon_clip = epsilon_clip

        PolicyGradient.__init__(self, env, config, seed, logger)

    # override inherited method to save previous log probabilities of actions taken
    def sample_path(self, env, num_episodes = None):
        """
        Sample paths (trajectories) from the environment.

        Args:
            num_episodes: the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            env: open AI Gym envinronment

        Returns:
            paths: a list of paths. Each path in paths is a dictionary with
                path["observation"] a numpy array of ordered observations in the path
                path["actions"] a numpy array of the corresponding actions in the path
                path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards: the sum of all rewards encountered during this "path"
        """
        episode = 0
        episode_rewards = []
        paths = []
        t = 0

        while (num_episodes or t < self.config.batch_size):
            state = env.reset()
            states, actions, rewards, prev_logprobs = [], [], [], []
            episode_reward = 0

            for step in range(self.config.max_ep_len):
                states.append(state)
                #pdb.set_trace()
                action = self.policy.act(states[-1][None])[0]
                state_torch = states[-1][None]
                #pdb.set_trace()
                state_torch = np2torch(state_torch)
                action_torch = np2torch(np.asarray(action))
                log_prob = self.policy.action_distribution(state_torch).log_prob(action_torch) 
                log_prob = log_prob.detach().numpy()
                state, reward, done, info = env.step(action)
                actions.append(action)
                rewards.append(reward)
                prev_logprobs.append(log_prob)
                episode_reward += reward
                t += 1
                if (done or step == self.config.max_ep_len-1):
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    break

            path = {"observation" : np.array(states),
                    "reward" : np.array(rewards),
                    "action" : np.array(actions),
                    "prev_logprob": np.array(prev_logprobs)}
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards

    # override inherited method
    def update_policy(self, observations, actions, advantages, prev_logprobs):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size]

        Perform one update on the policy using the provided data..
        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        prev_logprobs = np2torch(prev_logprobs)
        prev_logprobs = torch.squeeze(prev_logprobs)
        prev_logprobs = torch.max(prev_logprobs, torch.tensor(1e-5))
        self.optimizer.zero_grad()
        res = self.policy.action_distribution(observations).log_prob(actions) 
        ratio = torch.div(res,prev_logprobs) 
        nans = torch.isnan(ratio)
        nans_idx = (nans == True).nonzero(as_tuple=True)[0]
        clipped_ratio = torch.clamp(ratio, 1-self.epsilon_clip, 1+self.epsilon_clip)
        loss = -(torch.min(ratio,clipped_ratio) * advantages).mean() 
        loss.backward()
        print_network_grads(self.policy.network)
        self.optimizer.step()

    # override inherited method
    def train(self):
        """
        Performs training

        You do not have to change or use anything here, but take a look
        to see how all the code you've written fits together!
        """
        last_record = 0

        self.init_averages()
        all_total_rewards = []         # the returns of all episodes samples for training purposes
        averaged_total_rewards = []    # the returns for each iteration

        for t in range(self.config.num_batches):

            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.env)
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            prev_logprobs = np.concatenate([path["prev_logprob"] for path in paths])
            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)

            # advantage will depend on the baseline implementation
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            if self.config.use_baseline:
                self.baseline_network.update_baseline(returns, observations)
            self.update_policy(observations, actions, advantages, prev_logprobs) # PASS IN PREV LOG PROB

            # logging
            if (t % self.config.summary_freq == 0):
                self.update_averages(total_rewards, all_total_rewards)
                self.record_summary(t)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            averaged_total_rewards.append(avg_reward)
            self.logger.info(msg)

            if  self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        self.logger.info("- Training done.")
        np.save(self.config.scores_output, averaged_total_rewards)
        export_plot(averaged_total_rewards, "Score", self.config.env_name, self.config.plot_output)

