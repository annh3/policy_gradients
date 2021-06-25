import numpy as np
import torch
import gym
import os
from general import get_logger, Progbar, export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, device, np2torch, print_network_grads
from policy import CategoricalPolicy, GaussianPolicy
from policy_gradient import PolicyGradient
from utils import conjugate_gradient, line_search
import pdb

class TRPO(PolicyGradient):
    """
    Class for implementing PPO algorithm on
    """

    # def __init__(self, env, config, seed, logger=None):
    def __init__(self, env, config, seed, epsilon_clip=0.2, kl_limit=1, backtrack_alpha=1, max_backtrack=20, logger=None):
        self.alg_name = "ppo"
        self.epsilon_clip = epsilon_clip

        """
        Adding hyper-parameters for TRPO

        kl_limit = constant for back-tracking line search
        backtrack_alpha = the constant we exponentiate in back-tracking line search
        max_backtrack = the maximum number of backtracking steps to search on
        """
        self.kl_limit = kl_limit
        self.backtrack_alpha = backtrack_alpha
        self.max_backtrack = max_backtrack


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

        Perform one update on the policy using the provided data.

        References:

        [1] https://spinningup.openai.com/en/latest/algorithms/trpo.html#pseudocode
        [2] https://wiseodd.github.io/techblog/2018/03/11/fisher-information/
        [3] https://github.com/ikostrikov/pytorch-trpo/blob/master/trpo.py#L56

        We've passed in the advantages. We need to:

        1. Estimate the policy gradient g_k
        2. Compute the Hessian H of the sample average KL-divergence
        ###### What is the KL divergence OF?
        ###### Modify code to store old proba as well as old log pob?
        ###### Read reference [2] and look at [3] to see how they're computing
        ###### Sample Fisher information matrix (aka Hessian of KL divergence)
        3. Compute x_k using H and g_k via conjugate gradient descent
        4. Update the policy parameters with backtracking line search
        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        prev_logprobs = np2torch(prev_logprobs)
        prev_logprobs = torch.squeeze(prev_logprobs)
        prev_logprobs = torch.max(prev_logprobs, torch.tensor(1e-5))
        self.optimizer.zero_grad()
        res = self.policy.action_distribution(observations).log_prob(actions) 

        # for 1. let's look at VPG implementation
        pg = torch.mean(res*advantages)
        grads = torch.autograd.grad(pg, self.policy.parameters())
        pg_grad = torch.cat([grad.view(-1) for grad in grads]).data

        ##### To-Do: set up a symbolic function for computing the Hessian of KL divergence
        ##### Figure out what the derivative is with respect to - can look at ikostrikov's implementation
        ##### Or read the paper
        ##### My hypothesis/assumption: it's with respect to the current log probs, which we can
        ##### Access with self.policy.action_distribution(observations).log_prob(actions)
        ##### The question is: What parameter(s) is the Hessian with respect to?
        ##### Then we plug this symbolic function through the conjugate gradient algorithm

        # KL-divergence - we use estimator k_3
        # r = p / q where we're sampling from q - I think q is current proba
        # To-Do: just use torch exponentiation on this
        r = torch.div(prev_probs, self.policy.action_distribution(observations))
        # To-Do: store prev_probs, check that you are indeed getting the probabilities
        kl = (r-1) - torch.log(r)
        # take the mean?

        #### Both implementations do something like this
        # params = stochpol.trainable_variables
        # pg = flatgrad(surr, params)

        def H_sample(v):
            grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

            kl_v = (flat_grad_kl * Variable(v)).sum()
            grads = torch.autograd.grad(kl_v, self.policy.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

            return flat_grad_grad_kl + v * damping

        stepdir = conjugate_gradients(Fvp, pg_grad, 10)
        ## To-do: compute loss_grad above (Done)

        ## To-do: figure out why they're doing (in both implementations) all the steps after calling cg


        ##### Then we perform line search

        ##### Then we can call loss.backward() and optimizer.step() 


        # ratio = torch.div(res,prev_logprobs) 
        # nans = torch.isnan(ratio)
        # nans_idx = (nans == True).nonzero(as_tuple=True)[0]
        # clipped_ratio = torch.clamp(ratio, 1-self.epsilon_clip, 1+self.epsilon_clip)
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

