import torch
import torch.nn as nn
from .util import mlp


class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)
        # special tokens for context, state?
        # self.state_tokens = torch.nn.Parameter(0.01*torch.rand([1, state_dim]))

        # special tokens for fake actions
        # self.action_goal_tokens = torch.nn.Parameter(0.01*torch.rand([1, action_dim]))
        # self.action_context_tokens = torch.nn.Parameter(0.01*torch.rand([1, action_dim]))

    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state, action):
        # preprocessing with special tokens

        # state_mask = (state == 0)
        # state += self.state_tokens * state_mask

        # goal_mask = (action == 10)
        # action += self.action_goal_tokens * goal_mask

        # context_mask = (action == 100)
        # action += self.action_context_tokens * context_mask

        return torch.min(*self.both(state, action))

class TwinQ_new(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2, last_hidden = 64):
        super().__init__()
        self.state_dim = int(state_dim/2)
        dims = [int(state_dim/2) + action_dim, *([hidden_dim] * n_hidden), last_hidden]
        dims_1 = [int(state_dim), *([hidden_dim] * n_hidden), last_hidden]
        # dims_2 = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.sa1 = mlp(dims)
        self.s1 = mlp(dims_1)
        self.shift1 = torch.nn.Parameter((torch.rand(1)))
        self.sa2 = mlp(dims)
        self.s2 = mlp(dims_1)
        self.shift2 = torch.nn.Parameter((torch.rand(1)))

        # self.q2 = mlp(dims_2, squeeze_output=True)
        # self.r_f = mlp(dims, squeeze_output=True)

    def both(self, state, action):
        sa = torch.cat([state[:,0:self.state_dim], action], 1)
        sg = state
        # s = state[:, self.state_dim:]
        out1 = self.sa1(sa)-self.s1(sg)
        q1 = self.shift1 - torch.norm(out1, p = 2, dim = 1)
        out2 = self.sa2(sa)-self.s2(sg)
        q2 = self.shift2 - torch.norm(out2, p = 2, dim = 1)
        # sa_full = torch.cat([state, action], 1)
        # q2 = self.q2(sa_full)
        return q1, q2

    def forward(self, state, action):
        return torch.min(*self.both(state, action))

class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)

class RewardModel(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.reward = mlp(dims, squeeze_output=False)
    
    def forward(self, state):
        return self.reward(state)
    
    def predict(self, state, k=None):
        # -1 to fit the cost function
        return self.forward(state)-1.0
    
class RewardModelEnsemble(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2, n_ensemble = 10):
        super().__init__()
        self.reward = [RewardModel(state_dim, hidden_dim, n_hidden) for _ in range(n_ensemble)]

    def forward(self, state):
        output = torch.stack([self.reward[i](state) for i in range(len(self.reward))], dim = 1)
        return output.squeeze(2)
    
    def predict(self, state, k = 0):
        # -1 to fit the cost function
        if k==0:
            return torch.min(self.forward(state), dim = 1).values-1.0
        else:
            pred = self.forward(state)
            return torch.min(pred, dim = 1).values - k * torch.std(pred, dim = 1)-1.0
