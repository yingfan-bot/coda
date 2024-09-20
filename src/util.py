import csv
from datetime import datetime
import json
from pathlib import Path
import random
import string
import sys

import numpy as np
import torch
import torch.nn as nn


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 
class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


def mlp(dims, activation=nn.ReLU, output_activation=None, squeeze_output=False):
    n_dims = len(dims)
    assert n_dims >= 2, 'MLP requires at least two dims (input and output)'

    layers = []
    for i in range(n_dims - 2):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    if squeeze_output:
        assert dims[-1] == 1
        layers.append(Squeeze(-1))
    net = nn.Sequential(*layers)
    net.to(dtype=torch.float32)
    return net


def compute_batched(f, xs):
    return f(torch.cat(xs, dim=0)).split([len(x) for x in xs])


def update_exponential_moving_average(target, source, alpha):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1. - alpha).add_(source_param.data, alpha=alpha)


def torchify(x):
    x = torch.from_numpy(x)
    if x.dtype is torch.float64:
        x = x.float()
    # x = x.to(device=DEFAULT_DEVICE)
    return x



def torchcuda(x):
    if x.dtype is torch.float64:
        x = x.float()
    x = x.to(device=DEFAULT_DEVICE)
    return x


def return_range(transitions, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(transitions['rewards'], transitions['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(transitions['rewards'])
    return min(returns), max(returns)


# transitions is a dict, values of which are tensors of same first dimension
def sample_batch(args, transitions, context_state_transitions, batch_size):
    # state_dim = transitions['observations'].shape[1]
    k = list(transitions.keys())[0]
    n, device = len(transitions[k]), transitions[k].device
    n1 = len(context_state_transitions[k])
    for v in transitions.values():
        assert len(v) == n, 'transitions values must have same length'

    # assigning goals
    if args.contextual_goals_only:

        # random goal sampling from contexts as goals
        indices = torch.randint(low=0, high=n, size=(int(batch_size*(1-args.absorbing_ratio)),), device=device)
        set = {k: v[indices] for k, v in transitions.items()}
        # terminals are false and rewards are -1 from the original transitions
        random_goal_indices = torch.randint(low=0, high=n1, size=(int(batch_size*(1-args.absorbing_ratio)),), device=device)
        # select goal indices from contexts as goals
        random_goals = context_state_transitions['next_observations'][random_goal_indices]
        set['observations'] = torch.cat((set['observations'],random_goals),1)
        set['next_observations'] = torch.cat((set['next_observations'],random_goals),1)
        set['terminals'] = torch.zeros(set['terminals'].shape)
        set['rewards'] = torch.zeros(set['rewards'].shape) - args.cost
        
        # reaching contexts as goals in fake transitions
        context_indices = torch.randint(low=0, high=n1, size=(int(batch_size*args.absorbing_ratio),), device=device)
        context_set = {k: v[context_indices] for k, v in context_state_transitions.items()}
        # terminals are true and rewards are zero from the context_state_transitions
        context_goals = context_state_transitions['next_observations'][context_indices]
        context_set['observations'] = torch.cat((context_set['observations'],context_goals),1)
        context_set['next_observations'] = torch.cat((context_set['next_observations'],context_goals),1)
        context_set['terminals'] = torch.ones(context_set['terminals'].shape)
        context_set['rewards'] = torch.zeros(context_set['rewards'].shape)

        for k, v in set.items():
            set[k] = torch.cat((v,context_set[k]),0)

    else: 

        # random goal sampling from contexts as goals
        indices = torch.randint(low=0, high=n, size=(int(args.context_ratio*batch_size*(1-args.absorbing_ratio))+int((1-args.context_ratio)*batch_size*(1-args.absorbing_ratio)) ,), device=device)
        set = {k: v[indices] for k, v in transitions.items()}
        # terminals are false and rewards are -1 from the original transitions
        random_context_goal_indices = torch.randint(low=0, high=n1, size=(int(args.context_ratio*batch_size*(1-args.absorbing_ratio)),), device=device)
        random_context_goals = context_state_transitions['next_observations'][random_context_goal_indices]
        random_goal_indices = torch.randint(low=0, high=n, size=(int((1-args.context_ratio)*batch_size*(1-args.absorbing_ratio)),), device=device)
        random_goals = transitions['next_observations'][random_goal_indices]
        random_goals = torch.cat((random_goals, random_context_goals),0)
        set['observations'] = torch.cat((set['observations'],random_goals),1)
        set['next_observations'] = torch.cat((set['next_observations'],random_goals),1)
        set['terminals'] = torch.zeros(set['terminals'].shape)
        set['rewards'] = torch.zeros(set['rewards'].shape) - args.cost
        
        # reaching contexts as goals in fake transitions
        context_indices = torch.randint(low=0, high=n1, size=(int(args.context_ratio*batch_size*args.absorbing_ratio),), device=device)
        context_set = {k: v[context_indices] for k, v in context_state_transitions.items()}
        # terminals are true and rewards are zero from the context_state_transitions
        context_goals = context_state_transitions['next_observations'][context_indices]
        context_set['observations'] = torch.cat((context_set['observations'],context_goals),1)
        context_set['next_observations'] = torch.cat((context_set['next_observations'],context_goals),1)
        context_set['terminals'] = torch.ones(context_set['terminals'].shape)
        context_set['rewards'] = torch.zeros(context_set['rewards'].shape)

        # reaching next observation as goals
        absorbing_indices = torch.randint(low=0, high=n, size=(int((1-args.context_ratio)*batch_size*(args.absorbing_ratio)),), device=device)
        absorbing_set = {k: v[absorbing_indices] for k, v in transitions.items()}
        absorbing_goals = absorbing_set['next_observations']
        absorbing_set['observations'] = torch.cat((absorbing_set['next_observations'],absorbing_goals),1)
        absorbing_set['next_observations'] = torch.cat((absorbing_set['next_observations'],absorbing_goals),1)
        absorbing_set['actions'] = context_set['actions'][0].repeat(absorbing_set['actions'].shape[0],1)
        absorbing_set['terminals'] = torch.ones(absorbing_set['terminals'].shape)
        absorbing_set['rewards'] = torch.zeros(absorbing_set['rewards'].shape)

        for k, v in set.items():
            set[k] = torch.cat((v,context_set[k]),0)
            set[k] = torch.cat((v,absorbing_set[k]),0)

        # raise NotImplementedError

    # move to gpu
    for k, v in set.items():
        set[k] = torchcuda(v)
    
    return set
    
def sample_batch_policy_context(transitions, context_state_transitions, batch_size):
    # no sampling of fake transitions
    state_dim = transitions['observations'].shape[1]
    k = list(transitions.keys())[0]
    n, device = len(transitions[k]), transitions[k].device
    n1 = len(context_state_transitions[k])
    for v in transitions.values():
        assert len(v) == n, 'transitions values must have same length'

    # assigning goals
    # random goals from context
    indices = torch.randint(low=0, high=n, size=(int(batch_size),), device=device)
    set = {k: v[indices] for k, v in transitions.items()}
    random_goal_indices = torch.randint(low=0, high=n1, size=(int(batch_size),), device=device)
    random_goals = context_state_transitions['next_observations'][random_goal_indices, :state_dim]
    set['observations'] = torch.cat((set['observations'],random_goals),1)
    set['next_observations'] = torch.cat((set['next_observations'],random_goals),1)

    # move to gpu
    for k, v in set.items():
        set[k] = torchcuda(v)
    

    return set

def evaluate_policy(env, policy, max_episode_steps, deterministic=True):
    obs = env.reset()
    total_reward = 0.
    for _ in range(max_episode_steps):
        with torch.no_grad():
            obs = obs
            action = policy.act(torchcuda(torchify(obs)), deterministic=deterministic).cpu().numpy()
        next_obs, reward, done, info = env.step(action[:action.shape[0]-1])
        total_reward += reward
        if done:
            break
        else:
            obs = next_obs
    return total_reward

def evaluate_policy_four_rooms(env, policy, max_episode_steps, context,args, deterministic=True):
    obs = env.reset()
    env.set_context(context)
    total_reward = 0.
    for _ in range(max_episode_steps):
        with torch.no_grad():
            new_obs = np.concatenate((np.zeros((args.context_dim,)),obs))
            goal_obs = np.zeros((args.context_dim,)) + context
            goal_obs = np.concatenate((goal_obs, np.zeros((obs.shape[0],))))
            policy_obs = np.concatenate((new_obs, goal_obs))
            # obs augmented with context
            action = policy.act(torchcuda(torchify(policy_obs)), deterministic=deterministic).cpu().numpy()
        next_obs, reward, done, info = env.step(action[:action.shape[0]-1])
        total_reward += reward
        if done:
            break
        else:
            obs = next_obs
    return total_reward

def set_seed(seed, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)


def _gen_dir_name():
    now_str = datetime.now().strftime('%m-%d-%y_%H.%M.%S')
    rand_str = ''.join(random.choices(string.ascii_lowercase, k=4))
    return f'{now_str}_{rand_str}'

class Log:
    def __init__(self, root_log_dir, cfg_dict,
                 txt_filename='log.txt',
                 csv_filename='progress.csv',
                 cfg_filename='config.json',
                 flush=True):
        self.dir = Path(root_log_dir)/_gen_dir_name()
        self.dir.mkdir(parents=True)
        self.txt_file = open(self.dir/txt_filename, 'w')
        self.csv_file = None
        (self.dir/cfg_filename).write_text(json.dumps(cfg_dict))
        self.txt_filename = txt_filename
        self.csv_filename = csv_filename
        self.cfg_filename = cfg_filename
        self.flush = flush

    def write(self, message, end='\n'):
        now_str = datetime.now().strftime('%H:%M:%S')
        message = f'[{now_str}] ' + message
        for f in [sys.stdout, self.txt_file]:
            print(message, end=end, file=f, flush=self.flush)

    def __call__(self, *args, **kwargs):
        self.write(*args, **kwargs)

    def row(self, dict):
        if self.csv_file is None:
            self.csv_file = open(self.dir/self.csv_filename, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, list(dict.keys()))
            self.csv_writer.writeheader()

        self(str(dict))
        self.csv_writer.writerow(dict)
        if self.flush:
            self.csv_file.flush()

    def close(self):
        self.txt_file.close()
        if self.csv_file is not None:
            self.csv_file.close()