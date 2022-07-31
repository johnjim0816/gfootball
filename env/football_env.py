import os
import gym
import numpy as np
import gfootball
from gfootball import env as fe
from ray.rllib.env.multi_agent_env import MultiAgentEnv


def env_name_to_n_players(env_name):
    n_players = int(env_name[0])
    if 'auto_GK' in env_name:
        n_players -= 1
    return n_players


def create_football_env(env_name, n_controls, write_video, render, logdir):
    gfootball_dir = os.path.dirname(gfootball.__file__)
    assert os.path.exists(gfootball_dir), "Couldn't find gfootball package, make sure it is installed"
    scenarios_dir = os.path.join(gfootball_dir, "scenarios")
    assert os.path.exists(scenarios_dir), "Couldn't find gfootball scenarios folder, make sure it is installed"

    scenario_file_name = f"{env_name}.py"
    scenarios_gfootbal_file = os.path.join(scenarios_dir, scenario_file_name)
    if not os.path.exists(scenarios_gfootbal_file):
        assert os.path.exists(scenario_file_name), f"Couldn't find {scenario_file_name}, can't copy it to {scenarios_dir}"
        from shutil import copyfile
        copyfile(scenario_file_name, scenarios_gfootbal_file)

    assert os.path.exists(scenarios_gfootbal_file), f"Couldn't find {scenarios_gfootbal_file}, make sure you manually copy {scenario_file_name} to {scenarios_dir}"

    env = fe.create_environment(
        env_name=env_name,
        stacked=False,
        representation='simple115v2',
        # scoring is 1 for scoring a goal, -1 the opponent scoring a goal
        # checkpoint is +0.1 first time player gets to an area (10 checkpoint total, +1 reward max)
        rewards='checkpoints,scoring',
        logdir=logdir,
        write_goal_dumps=write_video,
        write_full_episode_dumps=write_video,
        render=render,
        write_video=write_video,
        dump_frequency=1 if write_video else 0,
        extra_players=None,
        number_of_left_players_agent_controls=n_controls,
        number_of_right_players_agent_controls=0)

    return env


class RllibGFootball(MultiAgentEnv):
    EXTRA_OBS_IDXS = np.r_[6:22,28:44,50:66,72:88,100:108]

    def __init__(self, env_name, write_video=False, render=False, logdir='/tmp/football'):
        self.n_players = env_name_to_n_players(env_name)
        self.env = create_football_env(env_name, self.n_players, write_video, render, logdir)

        self.action_space, self.observation_space = {}, {}
        for idx in range(self.n_players):
            self.action_space[f'player_{idx}'] = gym.spaces.Discrete(self.env.action_space.nvec[idx]) \
                if self.n_players > 1 else self.env.action_space
            lows = np.delete(self.env.observation_space.low[idx], RllibGFootball.EXTRA_OBS_IDXS)
            highs = np.delete(self.env.observation_space.high[idx], RllibGFootball.EXTRA_OBS_IDXS)
            self.observation_space[f'player_{idx}'] = gym.spaces.Box(
                low=lows, high=highs, dtype=self.env.observation_space.dtype) \
                if self.n_players > 1 else self.env.observation_space

        self.reward_range = np.array((-np.inf, np.inf))
        self.metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
        self.spec = None

    def _tidy_obs(self, obs):
        for key, values in obs.items():
            obs[key] = np.delete(values, RllibGFootball.EXTRA_OBS_IDXS)
        return obs

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for idx in range(self.n_players):
            obs[f'player_{idx}'] = original_obs[idx] \
                if self.n_players > 1 else original_obs
        return self._tidy_obs(obs)

    def step(self, action_dict):

        actions = []
        for idx in range(self.n_players):
            actions.append(action_dict[f'player_{idx}'])
        o, r, d, i = self.env.step(actions)

        game_info = {}
        for k, v in self.env.unwrapped._env._observation.items():
            game_info[k] = v

        scenario = self.env.unwrapped._env._config['level']
        obs, rewards, dones, infos = {}, {}, {}, {}
        for idx in range(self.n_players):
            obs[f'player_{idx}'] = o[idx] \
                if self.n_players > 1 else o
            rewards[f'player_{idx}'] = r[idx] \
                if self.n_players > 1 else r
            dones[f'player_{idx}'] = d
            dones['__all__'] = d
            infos[f'player_{idx}'] = i
            infos[f'player_{idx}']['game_scenario'] = scenario
            infos[f'player_{idx}']['game_info'] = game_info
            infos[f'player_{idx}']['action'] = action_dict[f'player_{idx}']

        return self._tidy_obs(obs), rewards, dones, infos
