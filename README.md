# Policy Gradient Methods

Added ipython notebook to walk through the derivation and implementation of vanilla policy gradient, derivation of TRPO. Added derivation and implementation of PPO. Beginning notes and code implementation for DDPG.

The contents of the notebook can now be found in a blog post [here](https://annhe.xyz/2021/04/12/policy-gradients/)

Currently Implemented:
1. VPG
2. PPO

In Progress:
1. TRPO
2. DDPG

* 06-22-21 Update: Starting implementation of TRPO
* 06-22-21 To-Do: Read Fisher information matrix = Hessian of KL divergence, decide how to compute sample H
* 06-25-21 Update: Added notes folder and notes on KL divergence
* 07-02-21 Update: Fleshed out code for DDPG
* 07-02-21 To-Do: Test and debug DDPG on MuJoCo tasks
* 07-05-21 Update: "Plumbing" for DDPG implementation
* 07-19-21 Update: continue "plumbing" DDPG
* 07-22-21 Update: current status: avg reward and avg action norm going to negative infinity, causing MuJoCo simulation error

### Notes for Running

python main.py --env-name ENV --seed SEED --alg ALG-NAME --no-baseline

python plot_by_alg.py --d DIRECTORY --env-name ENV --seeds SEEDS (comma-separated) --algs ALGS (comma-separated)

DDPG takes only continuous action envs - use env 'ant'
