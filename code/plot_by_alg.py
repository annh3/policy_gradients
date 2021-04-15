from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats as stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt


ENV_MAP = {
    'cartpole': 'CartPole-v0',
    'pendulum': 'InvertedPendulum-v2',
    'cheetah': 'HalfCheetah-v2',
}

def early_exit(message):
    print(message)
    exit()

def plot_combined(name, results):
    results = np.array(results)
    xs = np.arange(results.shape[1])
    ys = np.mean(results, axis=0)
    yerrs = stats.sem(results, axis=0)
    plt.fill_between(xs, ys - yerrs, ys + yerrs, alpha=0.25)
    plt.plot(xs, ys, label=name)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--directory', default='results', help='Directory containing TensorFlow event files')
    parser.add_argument('--env-name', required=True, help='Name of environment')
    parser.add_argument('--seeds', required=True, help='Comma-separated list of seeds to plot')
    parser.add_argument('--algs', required=True, help='Comma-separated list of algs to plot')
    args = parser.parse_args()

    if args.env_name not in ENV_MAP:
        early_exit('--env-name should be one of cartpole, pendulum, or cheetah')
    gym_env_name = ENV_MAP[args.env_name]

    seeds = [int(seed) for seed in args.seeds.split(',')]
    algs = [str(alg) for alg in args.algs.split(',')]

    all_results = {alg: [] for alg in algs}
    #print("all_results: ", all_results)

    directory = Path(args.directory)
    if not directory.is_dir():
        early_exit(f'{directory.resolve()} is not a directory')

    for seed in seeds:
        for alg in algs:
            format_str = f'{alg}-{gym_env_name}-{{}}-seed={seed}'
            #print("format")
            all_results[alg].append(np.load(directory/format_str.format('baseline')/'scores.npy'))

    plt.figure()
    plt.title(args.env_name)
    plt.xlabel('Iteration')
    for name, results in all_results.items():
        plot_combined(name, results)
    plt.legend()
    plt.savefig(directory / f'results-{args.env_name}', bbox_inches='tight')