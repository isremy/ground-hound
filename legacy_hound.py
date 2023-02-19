import gym
import copy
import numpy as np
import random as rd
from utils import *
from gym import spaces
from worlds.legacy_basic_grid import BasicGrid
# from stable_baselines3 import DQN

class Hound(gym.Env):

	# Dictionary of possible actions the agent can take
	MOVE_MAP = {0: [-1, 1], 1: [0, 1],	2: [1, 1],		3: [1, 0], 
							4: [1, -1], 5: [0, -1], 6: [-1, -1],	7: [-1, 0]
						 }

	def __init__(self, grid_size=15, obs_size=2, obs_density="normal", seed=None, start_pos=[14, 0]) -> None:
		super(Hound, self).__init__()
		self.__grid_size = grid_size
		self.__obs_size = obs_size
		self.__obs_density = obs_density
		self.__seed = seed
		self.__start_row = start_pos[0]
		self.__start_col = start_pos[1]
		self.__start_pos, self.__curr_pos, = copy.deepcopy(start_pos)
		
		new_grid = BasicGrid(grid_size, obs_size, obs_density, seed)
		self.__grid = new_grid.generate_level()
		# print(self.__grid)
		self.__reward_locations = new_grid.reward_locations()
		# self.rgb_dict = new_grid.get_rgb_dict()
		self.__env_rewards = new_grid.get_num_rewards()
		self.__max_obs = new_grid.get_obs_max()
		self.__grid[start_pos[0]][start_pos[1]] = 3
		self.__num_actions = 8
		max_rew = int(self.__max_obs / 2)
		self.action_space = spaces.Discrete(self.__num_actions)
		# observation = [agent-position, num-rewards, reward-1-location,...,reward-n-location]
		self.observation_space = spaces.Box(low=np.array([0] * 5 + [0] * (self.__grid_size * self.__grid_size)), 
																				high=np.array([grid_size - 1] * 4 + [max_rew] + [4] * (self.__grid_size * self.__grid_size)), 
																				shape=(5 + (self.__grid_size * self.__grid_size),), dtype=np.int32)

	def reset(self):
		self.__start_pos = [self.__start_row, self.__start_col]
		self.__curr_pos = self.__start_pos
		new_grid = BasicGrid(self.__grid_size, self.__obs_size, self.__obs_density, self.__seed)
		self.__grid = new_grid.generate_level()
		self.__env_rewards = new_grid.get_num_rewards()
		self.__grid[self.__start_pos[0]][self.__start_pos[1]] = 3
		self.__reward_locations = new_grid.reward_locations()

		flattened_grid = (np.ndarray.flatten(np.array(self.__grid))).tolist()

		obsv = [self.__curr_pos[0], self.__curr_pos[1]] + list(find_min_rew(self.__curr_pos, self.__reward_locations)) + [self.__env_rewards] + flattened_grid
		# obsv = [self.__curr_pos[0], self.__curr_pos[1]] + flattened_grid
		return np.array(obsv).astype(np.int32)

	def step(self, action):
		self.__grid[self.__curr_pos[0]][self.__curr_pos[1]] = 4
		self.__curr_pos = [self.__curr_pos[i] + self.MOVE_MAP[action][i] for i in range(2)]
		gem_reward = 0
		bad_reward = 0
		next_reward = find_min_rew(self.__curr_pos, self.__reward_locations) #change to self reward and increment
		done = False

		if self.__curr_pos[0] > self.__grid_size - 1 or self.__curr_pos[0] < 0 or self.__curr_pos[1] > self.__grid_size - 1 or self.__curr_pos[1] < 0:
			bad_reward = -1
			done = True
			self.__curr_pos = np.clip(self.__curr_pos, 0, self.__grid_size - 1)
		elif self.__grid[self.__curr_pos[0]][self.__curr_pos[1]] == 2:
			gem_reward = 1
			self.__reward_locations.remove((self.__curr_pos[0], self.__curr_pos[1]))
			self.__env_rewards -= 1

			if self.__env_rewards == 0:
				next_reward = (self.__curr_pos[0], self.__curr_pos[1])
				print("all rewards collected!")
				gem_reward += 1
				done = True
			else:
				next_reward = find_min_rew(self.__curr_pos, self.__reward_locations)

		self.__grid[self.__curr_pos[0]][self.__curr_pos[1]] = 3

		reward = (bad_reward + gem_reward - (calc_dist(self.__curr_pos, next_reward) / self.__grid_size))

		info = {0:self.__grid}

		return np.array([self.__curr_pos[0], self.__curr_pos[1], next_reward[0], next_reward[1], self.__env_rewards]
		 + (np.ndarray.flatten(np.array(self.__grid))).tolist() ).astype(np.int32), reward, done, info
		# return np.array([self.__curr_pos[0], self.__curr_pos[1]]
		#  + (np.ndarray.flatten(np.array(self.__grid))).tolist() ).astype(np.int32), reward, done, info		

	def render(self):
		pass

	def close(self):
		pass