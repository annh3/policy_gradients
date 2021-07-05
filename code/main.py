# -*- coding: UTF-8 -*-

import argparse
import numpy as np
import torch
import gym
from policy_gradient import PolicyGradient
from ppo import PPO
from config import get_config
import random

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', required=True, type=str,
                    choices=['cartpole', 'pendulum', 'cheetah'])
parser.add_argument('--baseline', dest='use_baseline', action='store_true')
parser.add_argument('--no-baseline', dest='use_baseline', action='store_false')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--alg', type=str, default='ppo')

parser.set_defaults(use_baseline=True)

"""
class C:
    def m(self):
        return "result"

an_object = C()

class_method = getattr(C, "m")
result = class_method(an_object)
"""

"""
eval(expression, globals=None, locals=None)
"""
alg_names = {'ppo': 'PPO',
            'vpg': 'PolicyGradient',
            'ddpg': 'DDPG'
            }


if __name__ == '__main__':
    args = parser.parse_args()

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config = get_config(args.env_name, args.use_baseline, args.seed, args.alg)
    env = gym.make(config.env_name)
    # train model
    #model = PolicyGradient(env, config, args.seed)
    kwargs = {'env': env, 'config': config, 'seed': args.seed}
    ModelConstructor = globals()[alg_names[args.alg]]
    model = ModelConstructor(**kwargs)
    #model = eval(alg_names[args.alg], local_args)
    #model = PPO(env, config, args.seed)
    model.run()
