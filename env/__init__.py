from gym.envs.registration import register
from .maze_model_new import MazeEnv, OPEN, U_MAZE, MEDIUM_MAZE, LARGE_MAZE, U_MAZE_EVAL, MEDIUM_MAZE_EVAL, LARGE_MAZE_EVAL
# from .antmaze import AntMaze

register(
    id='maze2d-umaze-v3',
    entry_point='env.maze_model_new:MazeEnv',
    max_episode_steps=300,
    kwargs={
        'maze_spec':U_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        # 'ref_min_score': 23.85,
        # 'ref_max_score': 161.86,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
    }
)

register(
    id='maze2d-medium-v3',
    entry_point='env.maze_model_new:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MEDIUM_MAZE,
        'reward_type':'sparse',
        'reset_target': False,
        # 'ref_min_score': 13.13,
        # 'ref_max_score': 277.39,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-medium-sparse-v1.hdf5'
    }
)