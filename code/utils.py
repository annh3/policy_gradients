import numpy as np 
import random
import pdb

def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer(object):

    def __init__(self, max_size):
        print("Initializing ReplayBuffer with maximum size: ", max_size)
        self.max_size = max_size
        self.num_in_buffer = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.done_mask = []

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        #pdb.set_trace()
        return batch_size + 1 <= self.num_in_buffer

    def update_buffer(self, states, actions, rewards, done_mask):
        """
        We treat this like an LRU cache
        Check whether we need to clear when this function is called
        """
        if self.num_in_buffer > self.max_size:
            clear_buffer()

        self.states.extend(states)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.done_mask.extend(done_mask)
        self.num_in_buffer += len(states)

    def clear_buffer():
        # easy way - clear buffer after UPDATE step of DDPG algorithm
        self.states = []
        self.actions = []
        self.rewards = []
        self.done_mask = []
        
    """
    To-Do: Modify this
    """
    def _encode_sample(self, idxes):
        #pdb.set_trace()
        #act_batch      = self.actions[idxes]
        # Let's try this
        rew_batch = np.array([self.rewards[idx] for idx in idxes])
        act_batch = np.array([self.actions[idx] for idx in idxes])
        #rew_batch      = self.rewards[idxes]
        #pdb.set_trace()
        #obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        obs_batch = np.array([self.states[idx] for idx in idxes])
        next_obs_batch = np.array([self.states[idx + 1] for idx in idxes])
        #next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask_batch      = np.array([1.0 if self.done_mask[idx] else 0.0 for idx in idxes], dtype=np.float32)

        #pdb.set_trace()

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask_batch

    def sample(self, batch_size):
        """Sample `batch_size` different transitions.
        i-th sample transition is the following:
        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

"""
06-22-21: Adding conjugate gradient descent, line search for TRPO
"""

def conjugate_gradient(A,b,tol):
    x = np.random.rand(b.shape[0]) # unif in [0,1)
    r = b - A @x    # calculate residual
    if np.linalg.norm(r) < tol:
        return x
    p = r
    k = 0
    
    while(True):
        alpha = np.dot(r,r) / float(p.T @ A @ p)
        x = x + alpha * p
        r_old = r
        r = r_old - alpha * A @ p
        if np.linalg.norm(r) < tol:
            return x
        Beta = float(r.T @ r) / float(r_old.T @ r_old)
        p = r + Beta * p
        k = k + 1
        if k > 10 * b.shape[0]:
            return x

def line_search(f, x_c, c, search_dir, rho, grad_dir, tol=1e-3, max_iter=100):
    alpha = 1
    iter = 0
    while iter < max_iter and f(x_c + alpha * search_dir) > f(x_c) + (c * alpha * grad_dir.T @ search_dir) + tol:
        alpha = rho * alpha
        iter = iter + 1
        if iter % 20 == 0:
            print("iter: ", iter)
            print("lhs: ", f(x_c + alpha * search_dir))
            print("rhs: ", f(x_c) + (c * alpha * grad_dir.T @ search_dir) + tol)
            print("\n")
    return alpha, f(x_c + alpha * search_dir)
