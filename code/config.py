import torch.nn as nn

class config_ant_ddpg:
    def __init__(self, use_baseline, seed, alg):
        self.update_every = 50 # refer to spinning up website
        self.total_env_interacts = 200000 # can change this later, based on other configs
        self.env_name="Ant-v2"
        self.record = False
        self.num_update_steps = 20 # change this later
        self.buffer_batch_size = 50 # change this later 
        self.q_lr = 1e-3 # change this later
        self.polyak = 0.995
        baseline_str = 'baseline' if use_baseline else 'no_baseline'
        seed_str = 'seed=' + str(seed)
        alg_str = str(alg)
        # output config
        self.output_path = "results/{}-{}-{}-{}/".format(alg_str, self.env_name, baseline_str, seed_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path 
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 200 # number of steps used to compute each policy update
        self.max_ep_len             = 200 # maximum episode length
        self.learning_rate          = 3e-2
        self.gamma                  = 1.0 # the discount factor
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True 

        # parameters for the policy and baseline models
        self.n_layers               = 1
        self.layer_size             = 50

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

class config_cartpole:
    def __init__(self, use_baseline, seed, alg):
        self.env_name="CartPole-v0"
        self.record = False
        baseline_str = 'baseline' if use_baseline else 'no_baseline'
        seed_str = 'seed=' + str(seed)
        alg_str = str(alg)
        # output config
        self.output_path = "results/{}-{}-{}-{}/".format(alg_str, self.env_name, baseline_str, seed_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path 
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 2000 # number of steps used to compute each policy update
        self.max_ep_len             = 200 # maximum episode length
        self.learning_rate          = 3e-2
        self.gamma                  = 1.0 # the discount factor
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True 

        # parameters for the policy and baseline models
        self.n_layers               = 1
        self.layer_size             = 64

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

class config_pendulum:
    def __init__(self, use_baseline, seed, alg):
        self.env_name="InvertedPendulum-v2"
        self.record = False
        baseline_str = 'baseline' if use_baseline else 'no_baseline'
        seed_str = 'seed=' + str(seed)
        alg_str = str(alg)
        # output config
        self.output_path = "results/{}-{}-{}-{}/".format(alg_str, self.env_name, baseline_str, seed_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path 
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 10000 # number of steps used to compute each policy update
        self.max_ep_len             = 1000 # maximum episode length
        self.learning_rate          = 3e-2
        self.gamma                  = 1.0 # the discount factor
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True 

        # parameters for the policy and baseline models
        self.n_layers               = 1
        self.layer_size             = 64

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

class config_cheetah_trpo:
    def __init__(self, use_baseline, seed, alg):
        self.env_name="HalfCheetah-v2"
        self.record = False
        baseline_str = 'baseline' if use_baseline else 'no_baseline'
        seed_str = 'seed=' + str(seed)
        alg_str = str(alg)
        # output config
        self.output_path = "results/{}-{}-{}-{}/".format(alg_str, self.env_name, baseline_str, seed_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path 
        self.record_freq = 5
        self.summary_freq = 1
        self.kl_limit = 0.01

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 50000 # number of steps used to compute each policy update
        self.max_ep_len             = 1000 # maximum episode length
        self.learning_rate          = 3e-2
        self.gamma                  = 0.9 # the discount factor
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True 

        # parameters for the policy and baseline models
        self.n_layers               = 2
        self.layer_size             = 64

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

class config_cheetah:
    def __init__(self, use_baseline, seed, alg):
        self.env_name="HalfCheetah-v2"
        self.record = False
        baseline_str = 'baseline' if use_baseline else 'no_baseline'
        seed_str = 'seed=' + str(seed)
        alg_str = str(alg)
        # output config
        self.output_path = "results/{}-{}-{}-{}/".format(alg_str, self.env_name, baseline_str, seed_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path     = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output  = self.output_path + "scores.png"
        self.record_path  = self.output_path 
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config
        self.num_batches            = 100 # number of batches trained on
        self.batch_size             = 50000 # number of steps used to compute each policy update
        self.max_ep_len             = 1000 # maximum episode length
        self.learning_rate          = 3e-2
        self.gamma                  = 0.9 # the discount factor
        self.use_baseline           = use_baseline
        self.normalize_advantage    = True 

        # parameters for the policy and baseline models
        self.n_layers               = 2
        self.layer_size             = 64

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

def get_config(env_name, baseline, seed=15, alg='ppo'):
    if alg == 'trpo':
        return config_cheetah_trpo(baseline, seed, alg)

    if alg == 'ddpg':
        if env_name == 'ant':
            return config_ant_ddpg(baseline, seed, alg)
        elif env_name == 'pendulum':
            return config_pendulum_ddpg(baseline, seed, alg)
        elif env_name == 'cheetah':
            return config_cheetah_ddpg(baseline, seed, alg)

    if env_name == 'cartpole':
        return config_cartpole(baseline, seed, alg)
    elif env_name == 'pendulum':
        return config_pendulum(baseline, seed, alg)
    elif env_name == 'cheetah':
        return config_cheetah(baseline, seed, alg)
