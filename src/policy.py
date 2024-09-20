import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
import torch.nn.functional as F

from .util import mlp


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * (n_hidden-1))])
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        net = F.relu(self.net(obs))
        mean = self.mean_head(net)
        log_std = torch.tanh(self.std_head(net))
        log_std = LOG_STD_MIN + 0.5 * (
            LOG_STD_MAX - LOG_STD_MIN
        ) * (log_std + 1)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            return dist.mean if deterministic else dist.sample()
            
class GaussianPolicy_old(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)
        # if mean.ndim > 1:
        #     batch_size = len(obs)
        #     return MultivariateNormal(mean, scale_tril=scale_tril.repeat(batch_size, 1, 1))
        # else:
        #     return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            return dist.mean if deterministic else dist.sample()

class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp([obs_dim, *([hidden_dim] * n_hidden), act_dim],
                       output_activation=nn.Tanh)

    def forward(self, obs):
        return self.net(obs)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs)


from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

MEAN_MIN = -9.0
MEAN_MAX = 9.0
LOG_STD_MIN = -5
LOG_STD_MAX = 2
EPS = 1e-7

class TanhPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, log_std_multiplier=1.0, log_std_offset=-1.0):
        super(TanhPolicy, self).__init__()

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.sigma_head = nn.Linear(256, action_dim)

        self.log_sigma_multiplier = log_std_multiplier
        self.log_sigma_offset = log_std_offset

    def _get_outputs(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        mu = self.mu_head(a)
        mu = torch.clip(mu, MEAN_MIN, MEAN_MAX)
        log_sigma = self.sigma_head(a)
        # log_sigma = self.log_sigma_multiplier * log_sigma + self.log_sigma_offset

        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = TransformedDistribution(
            Normal(mu, sigma), TanhTransform(cache_size=1)
        )
        a_tanh_mode = torch.tanh(mu)
        return a_distribution, a_tanh_mode
    
    def forward(self, state):
        a_dist, _ = self._get_outputs(state)

        return a_dist

    def act(self, state, deterministic=False, enable_grad=False):
        a_dist, a_tanh_mode = self._get_outputs(state)
        # action = a_dist.mean()
        action = a_dist.rsample()
        return action
    # def forward(self, state):
    #     a_dist, a_tanh_mode = self._get_outputs(state)
    #     action = a_dist.rsample()
    #     logp_pi = a_dist.log_prob(action).sum(axis=-1)
    #     return action, logp_pi, a_tanh_mode

    def get_log_density(self, state, action):
        a_dist, _ = self._get_outputs(state)
        action_clip = torch.clip(action, -1. + EPS, 1. - EPS)
        logp_action = a_dist.log_prob(action_clip)
        return logp_action