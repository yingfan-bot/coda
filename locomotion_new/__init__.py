from gym.envs.registration import register
from locomotion_new import ant
from locomotion_new import maze_env

register(
    id='cell-antmaze-medium-v2',
    entry_point='locomotion_new.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_big-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)


register(
    id='cell-antmaze-large-v2',
    entry_point='locomotion_new.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'sparse',
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_True_sparse_fixed.hdf5',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'v2_resets': True,
    }
)

