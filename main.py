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
from src.util import set_seed, Log, sample_batch, torchify, evaluate_policy, sample_batch_policy_context
from torch_ema import ExponentialMovingAverage
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./runs_test/")

def create_context_transitions(transitions, context_dim, state_dim, args, fake_action = 10):
    '''
    Augmented state dim = context dim + original state dim;
    Return fake transitions from the state to the correpsonding context.
    '''
    # states
    # state = transitions['next_observations']
    # context distribution extracted from the terminals
    state = transitions['next_observations'][transitions['terminals'] == True]
    if args.perturb_state:
        state[:, 3:] = state[:, 3:] + np.random.normal(0, args.perturb_scale, size = state[:, 3:].shape)
    # concatenate 0 for the context dims, since those are normal states
    state = np.concatenate((np.zeros((state.shape[0], context_dim)),state),1)

    # construct corresponding contexts
    # next_state = transitions['next_observations'][:, 0:context_dim]
    next_state = transitions['next_observations'][transitions['terminals'] == True][:, 0:context_dim]
    # 2D goal locations as context; there's a mapping between context and goal sets (as long as the first two dims match)
    next_state = np.concatenate((next_state, np.zeros((next_state.shape[0], state_dim))),1)

    # construct transitions
    context_state_transitions = {}
    context_state_transitions['observations'] = state
    context_state_transitions['next_observations'] = next_state

    # Fake action is needed if 1. one state can correspond to multiple contexts (cannot directly set terminal to be true) and 2. we need to regularize q. 
    # But regularize v does not work (change reg_v to True and observe the loss, the q values are not correct)

    # !More over, if we can get a state->context dictionary for all states in the original transition dataset (as in antmaze), goal sampling is much easier since we can directly sample within future 1/(1-discount) steps as both state goals and context goals to accelerate training. However, it is not the general case.
    
    context_state_transitions['actions'] = np.concatenate((np.zeros((transitions['actions'].shape[0],transitions['actions'].shape[1])), np.zeros((transitions['actions'].shape[0], 1))+ fake_action),1)

    # will be reassigned during goal sampling
    context_state_transitions['rewards'] = np.zeros(transitions['rewards'].shape)
    context_state_transitions['terminals'] = np.ones(transitions['terminals'].shape)

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
    if env_name == 'antmaze-umaze-v2':
        eval_env = make_offline_ant('offline_ant_umaze')
    elif env_name == 'antmaze-umaze-diverse-v2':
        eval_env = make_offline_ant('offline_ant_umaze_diverse')
    elif env_name == 'antmaze-medium-play-v2':
        eval_env = make_offline_ant('offline_ant_medium_play')
    elif env_name == 'antmaze-medium-diverse-v2':
        eval_env = make_offline_ant('offline_ant_medium_diverse')
    elif env_name == 'antmaze-large-play-v2':
        eval_env = make_offline_ant('offline_ant_large_play')
    elif env_name == 'antmaze-large-diverse-v2':
        eval_env = make_offline_ant('offline_ant_large_diverse')
    else:
        raise NotImplementedError

    # eval_env.set_target(target_location)
    # create contextual transitions
    context_state_transitions = create_context_transitions(transitions, args.context_dim, transitions['observations'].shape[1], args)
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
        eval_returns = np.array([evaluate_policy(eval_env, policy, max_episode_steps) \
                                 for _ in range(args.n_eval_episodes)])
        normalized_returns = d4rl.get_normalized_score(args.env_name, eval_returns) * 100.0
        log.row({
            'return mean': eval_returns.mean(),
            'return std': eval_returns.std(),
            'normalized return mean': normalized_returns.mean(),
            'normalized return std': normalized_returns.std(),
        })
        return normalized_returns.mean()
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
                        rew_return = eval_policy()
                else:
                    rew_return = eval_policy()
                writer.add_scalar('Loss/return', rew_return, step)
   
            step += 1

    torch.save(iql.state_dict(), log.dir/'final.pt')
    log.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env-name', default = "antmaze-medium-play-v2")
    parser.add_argument('--log-dir', default = "./log")
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--context-dim', type=int, default=2)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--n-hidden', type=int, default=3)
    parser.add_argument('--n-steps', type=int, default=10**6*2)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--perturb-state', type=int, default=1)
    parser.add_argument('--perturb-scale', type=float, default=0.05)
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
    parser.add_argument('--reg-v', type=int, help = 'regularize v instead of q', default=0)
    parser.add_argument('--ema-policy', type=bool, default=False)
    parser.add_argument('--eval-period', type=int, default=100000, help = 'run evaluation every n steps')
    parser.add_argument('--n-eval-episodes', type=int, default=100, help = 'run eval for n episides')
    parser.add_argument('--clip-grad-norm', type=float, default=0.1, help = 'clip grad norm for policy')
    main(parser.parse_args())
