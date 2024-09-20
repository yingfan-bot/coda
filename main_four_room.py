from pathlib import Path
import gym
import d4rl
import numpy as np
import torch
from env.antmaze import make_offline_ant
from tqdm import trange
from env import maze_model_new
from src.iql import ImplicitQLearning
from src.policy import GaussianPolicy, GaussianPolicy_old, DeterministicPolicy, TanhPolicy
from src.value_functions import TwinQ, ValueFunction, TwinQ_new
from src.util import set_seed, Log, sample_batch, torchify, evaluate_policy, sample_batch_policy_context, evaluate_policy_four_rooms
from torch_ema import ExponentialMovingAverage
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./runs_test/")
import locomotion
from locomotion.maze_env import context_to_range

def satisfy_thresholds(state, thresholds):
    return ((state[:,0] > thresholds[0]) & (thresholds[1] > state[:,0]) & (state[:,1] > thresholds[2]) & (thresholds[3] > state[:,1]))

def create_context_transitions_four_rooms(transitions, context_dim, state_dim, args, fake_action = 10):
    '''
    Augmented state dim = context dim + original state dim;
    Return fake transitions from the state to the correpsonding context.
    '''
    context_state_transitions = {}
    if 'large' in args.env_name:
        maze_size = 9
    elif 'medium' in args.env_name:
        maze_size = 8
    import copy
    transitions_copy = copy.deepcopy(transitions)
    if args.perturb_state:
        transitions_copy['next_observations']+=np.random.normal(0, args.perturb_scale, size = transitions_copy['next_observations'].shape)
    for i in range(3):
        thresholds = context_to_range(maze_size, i+2)
        state = transitions_copy['next_observations'][satisfy_thresholds(transitions_copy['next_observations'],thresholds)]
        if args.ood:
            state = np.concatenate((state, state[:10000]+50),0)
        if args.less_sample:
            state = state[:20]
        state = np.concatenate((np.zeros((state.shape[0], context_dim)),state),1)
        # if 'large' in args.env_name:
        #     if i == 1:
        #         state = np.repeat(state, 3, axis=0)
        # elif 'medium' in args.env_name:
        #     if i == 0:
        #         state = np.repeat(state, 10, axis=0)
        print(state.shape)
        next_state = np.zeros((state.shape[0], context_dim))+i+2
        next_state = np.concatenate((next_state, np.zeros((next_state.shape[0], state_dim))),1)
        if i == 0: 
            context_state_transitions['observations'] = state
            context_state_transitions['next_observations'] = next_state
            context_state_transitions['actions'] = np.concatenate((np.zeros((transitions['actions'].shape[0],transitions['actions'].shape[1])), np.zeros((transitions['actions'].shape[0], 1))+ fake_action),1)
            context_state_transitions['rewards'] = np.zeros(transitions['rewards'].shape)
            context_state_transitions['terminals'] = np.ones(transitions['terminals'].shape)
        else:
            context_state_transitions['observations'] = np.concatenate((context_state_transitions['observations'], state),0)
            context_state_transitions['next_observations'] = np.concatenate((context_state_transitions['next_observations'], next_state),0)
            context_state_transitions['actions'] = np.concatenate((context_state_transitions['actions'], np.concatenate((np.zeros((transitions['actions'].shape[0],transitions['actions'].shape[1])), np.zeros((transitions['actions'].shape[0], 1))+ fake_action),1)),0)
            context_state_transitions['rewards'] = np.concatenate((context_state_transitions['rewards'], np.zeros(transitions['rewards'].shape)),0)
            context_state_transitions['terminals'] = np.concatenate((context_state_transitions['terminals'], np.ones(transitions['terminals'].shape)),0)
    
    for k, v in context_state_transitions.items():
        # torchify to cpu and move to gpu during sampling
        context_state_transitions[k] = torchify(v)

    return context_state_transitions

def create_argmented_transitions(args, transitions, context_dim):
    '''
    Return augmented original transitions
    '''
    # fill 0 for contextual dims
    transitions['observations'] = np.concatenate((np.zeros((transitions['observations'].shape[0], context_dim)),transitions['observations']),1)
    transitions['next_observations'] = np.concatenate((np.zeros((transitions['next_observations'].shape[0], context_dim)),transitions['next_observations']),1)

    # will be reassigned during goal sampling
    transitions['rewards'] = np.zeros(transitions['rewards'].shape) - args.cost
    transitions['terminals'] = np.zeros(transitions['terminals'].shape)

    # directly use the terminals from the dataset (goal distribution)
    # transitions['terminals'] = transitions['terminals']
    transitions['actions'] = np.concatenate((transitions['actions'], np.zeros((transitions['actions'].shape[0],1))), 1)

    for k, v in transitions.items():
        transitions[k] = torchify(v)

    return transitions

def get_env_and_transitions(log, env_name, max_episode_steps, args):
    env = gym.make(env_name)
    transitions = d4rl.qlearning_dataset(env)
    if env_name == 'antmaze-medium-play-v2' or env_name == 'antmaze-medium-diverse-v2':
        eval_env = gym.make('modified-antmaze-medium-v2')
    elif env_name == 'antmaze-large-play-v2' or env_name == 'antmaze-large-diverse-v2':
        eval_env = gym.make('modified-antmaze-large-v2')
    else:
        raise NotImplementedError

    # eval_env.set_target(target_location)
    # create contextual transitions
    context_state_transitions = create_context_transitions_four_rooms(transitions, args.context_dim, transitions['observations'].shape[1], args)
    aug_transitions = create_argmented_transitions(args, transitions, args.context_dim)

    return env, eval_env, aug_transitions, context_state_transitions


def main(args):
    torch.set_num_threads(1)
    set_seed(args.seed)
    log = Log(Path(args.log_dir)/args.env_name, vars(args))
    log(f'Log dir: {log.dir}')
    if args.env_name == 'antmaze-umaze-v2' or args.env_name == 'antmaze-umaze-diverse-v2':
        max_episode_steps = 700
    else:
        max_episode_steps = 1000
    env, eval_env, transitions, context_state_transitions = get_env_and_transitions(log, args.env_name, max_episode_steps, args)
    obs_dim = transitions['observations'].shape[1]*2
    act_dim = transitions['actions'].shape[1]   # this assume continuous actions
    set_seed(args.seed, env=env)
    set_seed(args.seed, env=eval_env)

    if args.deterministic_policy:
        policy = DeterministicPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)

    else:
        if args.learn_std:
            policy = TanhPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)
        else:
            policy = GaussianPolicy_old(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden)

    def eval_policy():
        for i in range(3):
            eval_returns = np.array([evaluate_policy_four_rooms(eval_env, policy, max_episode_steps, i+2, args) \
                                    for _ in range(args.n_eval_episodes)])
            normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
            log.row({
                'context': i+2,
                'return mean': eval_returns.mean(),
                'return std': eval_returns.std(),
                'normalized return mean': normalized_returns.mean(),
                'normalized return std': normalized_returns.std(),
            })
            
    if args.emd:
        iql = ImplicitQLearning(
            qf=TwinQ_new(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden, last_hidden = args.last_hidden),
            vf=ValueFunction(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
            policy=policy,
            optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            max_steps=args.n_steps,
            tau=args.tau,
            beta=args.beta,
            alpha=args.alpha,
            discount=args.discount,
            decay = args.pg_decay,
            absorbing_ratio = args.absorbing_ratio,
            args = args
        )
    else:
        iql = ImplicitQLearning(
            qf=TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
            vf=ValueFunction(obs_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden),
            policy=policy,
            optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
            max_steps=args.n_steps,
            tau=args.tau,
            beta=args.beta,
            alpha=args.alpha,
            discount=args.discount,
            decay = args.pg_decay,
            absorbing_ratio = args.absorbing_ratio,
            args = args
        )
    ema = ExponentialMovingAverage(policy.parameters(), decay=0.995)

    step = 0
    with trange(args.n_steps, disable = True) as pbar:
        for _ in pbar:
            # eval_policy()
            if not args.reg_v:
                v_loss, q_loss, q_test,  q_test_0 = iql.update(**sample_batch(args, transitions, context_state_transitions, args.batch_size))
            else: 
                v_loss, q_loss, q_test,  q_test_0 = iql.update_reg_v(**sample_batch(args, transitions, context_state_transitions, args.batch_size))
            if step%100 == 0:
                writer.add_scalar('Loss/value_loss', v_loss, step)
                writer.add_scalar('Loss/q_loss', q_loss, step)
                writer.add_scalar('Loss/q_mean_test', q_test, step)
                writer.add_scalar('Loss/q_reg_test', q_test_0, step)
             
            exp_adv = iql.update_policy(**sample_batch_policy_context(transitions, context_state_transitions, args.batch_size))
            ema.update()
            pbar.set_postfix(v = v_loss, q_ = q_test ,  q = q_loss, q__ = q_test_0, exp_adv = exp_adv.item())
            if step%100 == 0:
                writer.add_scalar('Loss/exp_adv', exp_adv.item(), step)
            if (step+1) % args.eval_period == 0:
                if args.ema_policy:
                    with ema.average_parameters():
                        eval_policy()
                else:
                    eval_policy()
   
            step += 1

    torch.save(iql.state_dict(), log.dir/'final.pt')
    log.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env-name', default = "antmaze-large-play-v2")
    parser.add_argument('--log-dir', default = "./log/four_rooms")
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--ood', type=int, default=0)
    parser.add_argument('--less-sample', type=int, default=0)
    parser.add_argument('--perturb-state', type=int, default=1)
    parser.add_argument('--perturb-scale', type=float, default=0.05)
    parser.add_argument('--context-dim', type=int, default=1)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=3)
    parser.add_argument('--n-steps', type=int, default=10**6*2)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--contextual-goals-only', type=bool, default=True, help = 'only sample context as goals in q learning')
    parser.add_argument('--context-ratio', type=float, default=1.0, help = 'the ratio of contexts as goals in goal sampling')
    parser.add_argument('--absorbing-ratio', type=float, default=0.5, help = 'the ratio of sampling fake transitions as goals')
    parser.add_argument('--cost', type=float, default=1.0, help = 'cost for each step when not reaching the goal')
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.9, help = 'try 0.9 or 0.95') #0.5 to test problems
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--emd', type=bool, default = False)
    parser.add_argument('--pg-decay', type=bool, default = False, help = 'whether to use lr decay for the policy')
    parser.add_argument('--last-hidden', type=int, default = 256, help='dim of last hidden layer for emd, only used when emd is true')
    parser.add_argument('--deterministic-policy', action='store_true')
    parser.add_argument('--learn-std', type=bool, help = 'state dependent variance for Gaussian policy', default=False)
    parser.add_argument('--reg-v', type=bool, help = 'regularize v instead of q', default=False)
    parser.add_argument('--ema-policy', type=bool, default=False)
    parser.add_argument('--eval-period', type=int, default=100000, help = 'run evaluation every n steps')
    parser.add_argument('--n-eval-episodes', type=int, default=100, help = 'run eval for n episides')
    parser.add_argument('--clip-grad-norm', type=float, default=0.1, help = 'clip grad norm for policy')
    main(parser.parse_args())
