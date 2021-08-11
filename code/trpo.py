import numpy as np
import torch
import gym
import os
from general import get_logger, Progbar, export_plot
from baseline_network import BaselineNetwork
from network_utils import build_mlp, device, np2torch, print_network_grads
from policy import CategoricalPolicy, GaussianPolicy
from policy_gradient import PolicyGradient
from utils import conjugate_gradient, linesearch, normal_log_density, set_flat_params_to, get_flat_params_from
import pdb

class TRPO(PolicyGradient):
    """
    Class for implementing TRPO
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

        print("Initializing TRPO")
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
    def update_policy(self, observations, actions, advantages, prev_logprobs, max_kl):
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
        """
        We may not need this
        """
        prev_logprobs = np2torch(prev_logprobs)
        prev_logprobs = torch.squeeze(prev_logprobs)
        prev_logprobs = torch.max(prev_logprobs, torch.tensor(1e-5))
        """
        End we may not need this
        """
        self.optimizer.zero_grad()
        cur_logprobs = self.policy.action_distribution(observations).log_prob(actions) 
        fixed_cur_logprobs = cur_logprobs.detach()
        # if this doesn't work, follow the approach in the tutorial
        action_loss = Variable(advantages) * torch.exp(cur_logprobs - prev_logprobs)

        grad1 = torch.autograd.grad(action_loss, self.policy.network.parameters(), retain_graph=True)

        # grad_flat will be a vector K*1, where K is the total number of parameters in the policy net

        grad1_flat = torch._utils._flatten_dense_tensors(grad1)

        # Now, compute derivative using policy gradient formula another way
        _grad2 = torch.autograd.grad(cur_logprobs, self.policy.network.parameters())
        _grad2_flat = torch._utils._flatten_dense_tensors(_grad2)
        grad2 = advantages*_grad2_flat
        # assert(grad2 == grad1)


        """
        Ok here we need some work!!!
        
        We've computed the action loss

        Now we need to find (1) the right step direction, and 
        (2) the right step size

        """
        def get_kl():
            action_prob1 = self.network.forward(observations)
            # calling .data detaches action_prob0 from the graph
            action_prob0 = Variable(action_prob1.data)
            kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
            return kl.sum(1, keepdim=True)

        """
        Recall: Hs = g
        s = H^{-1}g

        ^ we can calculate this Fisher-vector product directly

        Instead of ever caculating the Hessian
        """

        # Note that v is g, our action_loss
        def Fvp_direct(v):
            damping = 1e-2
            kl_sum = get_kl()
            # compute the first derivative or Kl wrt the network parameters
            # flatten into a vector
            grads = torch.autograd.grad(kl_sum, self.network.parameters(), create_graph=True)
            grads_flat = torch.cat([grad.view(-1) for grad in grads])
            # compute the dot prodct with the input vector
            grads_v = torch.sum(grads_flat * v)
            # now compute the derivative again
            grads_grads_v = torch.autograd.grad(grads_v, self.network.parameters(), create_grad=False)
            flat_grad_grad_v = torch.cat([grad.contiguous().view(-1) for grad in grads_grads_v]).data
            return flat_grad_grad_v + v * damping

        def get_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    action_means, action_log_stds, action_stds = self.policy.network(Variable(observations))
            else:
                action_means, action_log_stds, action_stds = self.policy.network(Variable(observations))
                
            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()

        stepdir = conjugate_gradient(Fvp_direct, -action_loss, 10)

        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

        lm = torch.sqrt(shs / max_kl)
        fullstep = stepdir / lm[0]

        neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
        print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))

        prev_params = get_flat_params_from(self.policy.network)
        """
        We need to pass the dynamic evaluation of the aciton loss to line search
        """
        success, new_params = linesearch(self.policy.network, get_loss, prev_params, fullstep,
                                     neggdotstepdir / lm[0])
        set_flat_params_to(model, new_params)

        return action_loss

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
            print("t: ", t)

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
            self.update_policy(observations, actions, advantages, prev_logprobs, kl_limit) # PASS IN PREV LOG PROB

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

    def record(self):
        """
        Recreate an env and record a video for one episode
        """
        env = gym.make(self.config.env_name)
        env.seed(self.seed)
        env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
        self.evaluate(env, 1)

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # record one game at the beginning
        # if self.config.record:
        #     self.record()
        # model
        print("About to call train")
        self.train()
        # record one game at the end
        # if self.config.record:
        #     self.record()

