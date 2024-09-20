import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from .util import DEFAULT_DEVICE, compute_batched, update_exponential_moving_average


EXP_ADV_MAX = 100.


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class ImplicitQLearning(nn.Module):
    def __init__(self, qf, vf, policy, optimizer_factory, max_steps,
                 tau, beta, discount=0.99, alpha=0.005, decay = True, absorbing_ratio = 0.5, args = None):
        super().__init__()
        self.qf = qf.to(DEFAULT_DEVICE)
        self.q_target = copy.deepcopy(qf).requires_grad_(False).to(DEFAULT_DEVICE)
        self.vf = vf.to(DEFAULT_DEVICE)
        self.policy = policy.to(DEFAULT_DEVICE)
        self.v_optimizer = optimizer_factory(list(self.vf.parameters()))
        self.q_optimizer = optimizer_factory(self.qf.parameters())
        self.policy_optimizer = optimizer_factory(self.policy.parameters())
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, max_steps)
        self.tau = tau
        self.beta = beta
        self.discount = discount
        self.alpha = alpha
        self.decay = decay
        self.absorbing_ratio = absorbing_ratio
        self.args = args
    

    def update(self, observations, actions, next_observations, rewards, terminals):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v = self.vf(next_observations)

        # Update V function
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.tau) 
        # + F.mse_loss(v[new_batch_size:],torch.tensor(0.0).to(observations.device))
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # v_loss_test = torch.mean(v[:new_batch_size])
        # v_loss_0 = torch.mean(v[new_batch_size:])

        # Update Q function
        targets = (rewards + (1. - terminals) * self.discount * next_v.detach()).clamp(min = -110, max = 0)
        qs = self.qf.both(observations, actions)

        q_loss = sum((F.mse_loss(q, targets))for q in qs) / len(qs)
        # +F.mse_loss(q[new_batch_size:],torch.tensor(0.0).to(observations.device))
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        new_batch_size = int(self.args.context_ratio*self.args.batch_size*(1-self.args.absorbing_ratio))+int((1-self.args.context_ratio)*self.args.batch_size*(1-self.args.absorbing_ratio))
        q_test = torch.mean(self.q_target(observations[:new_batch_size], actions[:new_batch_size])).detach()
        q_test_0 = torch.mean(self.q_target(observations[new_batch_size:], actions[new_batch_size:])).detach()

        return v_loss.item(), q_loss.item(), q_test.item(), q_test_0.item()

    def update_reg_v(self, observations, actions, next_observations, rewards, terminals):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            next_v = self.vf(next_observations)

        # Update V function
        new_batch_size = int(observations.shape[0]*(1-self.absorbing_ratio))
        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv[:new_batch_size], self.tau)\
        + F.mse_loss(v[new_batch_size:],torch.tensor(0.0).to(observations.device))
        
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        # v_loss_test = torch.mean(v[:new_batch_size])
        # v_loss_0 = torch.mean(v[new_batch_size:])

        # Update Q function
        targets = rewards + (1. - terminals) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum((F.mse_loss(q[:new_batch_size], targets[:new_batch_size]))for q in qs) / len(qs)
        # +F.mse_loss(q[new_batch_size:],torch.tensor(0.0).to(observations.device))
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()
        q_test = torch.mean(self.q_target(observations[:new_batch_size], actions[:new_batch_size]))
        q_test_0 = torch.mean(v[new_batch_size:])

        # Update target Q network
        update_exponential_moving_average(self.q_target, self.qf, self.alpha)

        return v_loss.item(), q_loss.item(), q_test.item(), q_test_0.item()
    
    def update_policy(self, observations, actions, next_observations, rewards, terminals):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
            v = self.vf(observations)
            adv = target_q - v
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.policy(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            assert policy_out.shape == actions.shape
            bc_losses = torch.sum((policy_out - actions)**2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.clip_grad_norm)
        self.policy_optimizer.step()
        if self.decay:
            self.policy_lr_schedule.step()

        return exp_adv.mean()
