from alpha_hound import Hound
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
from animate_grid import AnimateGrid
from worlds.basic_house import BasicHouse
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure

class RewardCollection():
	def __init__(self) -> None:
		self.__total_rewards = []
	
	def callback(self, cum_reward):
		self.__total_rewards.append(cum_reward)
	
	def get_reward_data(self):
		return self.__total_rewards

ARB_STEPS = 10
PIXEL_SIZE = 30
# TODO: Should use RGB values from container json dict instead
RGB_DICT = { -2:	"#A56A84",
						 -1:	"#FF1010",				# black
							0:	"#C0AEAB",			# light green
							1:	"#000000",	# dark grey
							2:	"#78281F",			# blue
							3:	"#6E2C00",			# red 
							4:	"#34495E",			# dark green
							5: 	"#784212"
						}

if __name__ == '__main__':
	reward_data_obj = RewardCollection()
	test_hound = Hound(BasicHouse, "living room", "magazine", reward_callback=reward_data_obj)
	check_env(test_hound, warn=True) 
	test_hound = make_vec_env(lambda: test_hound, n_envs=1)
	# model = RecurrentPPO('MlpLstmPolicy', test_hound, verbose=1)
	model = PPO('MlpPolicy', test_hound, verbose=1)
	model.learn(50_000)

	env = model.get_env()
	mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)
	print("MEAN REWARD", mean_reward)



	obs = test_hound.reset()
	# obs_size = len(obs.tolist())

	lstm_states = None

	frames = []
	rewards = []
	visited_containers = []
	total_reward = 0
	done = False
	idx = 0

	# Episode start signals are used to reset the lstm states
	episode_starts = np.ones((1,), dtype=bool)
	for i in range(ARB_STEPS):
		action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
		obs, reward, done, info = test_hound.step(action)

		visited_containers.append(info[0][2])

		total_reward += reward[0]

		# if reward != 0:
		

		# TODO: Color all locations agent has traveled to make it easier to
		#				visualize path
		local = copy.deepcopy(info[0][0])
		local[info[0][1][-1][0]][info[0][1][-1][1]] = 0
		idx = 0
		for loc in info[0][1]:
			local[loc[0]][loc[1]] = -1
			frames.append(copy.deepcopy(local))
			rewards.append(reward)
			local[loc[0]][loc[1]] = 0

		idx += 1
		if done:
			nx.draw(info[0][3], with_labels=True)
			plt.show()
			print("Finished!")
			break

	if not done:
		print(idx)
		print("Did not complete course :(")

	print("Reward: ", total_reward)
	print("LOCATIONS: ", visited_containers)
	# print("CUM_REWARDS: ", reward_data_obj.get_reward_data())
	# print("LENGTH: ", len(reward_data_obj.get_reward_data()))

	# fig, ax = plt.subplots()


	# TODO: Got plotting to work, need to make it better
	# TODO: find some other good plots to use for reward, from drone navigation DRL obstacle avoidance paper
	# x = np.linspace(0, len(reward_data_obj.get_reward_data()), len(reward_data_obj.get_reward_data()))
	# y = reward_data_obj.get_reward_data()

	# x = [int(val) for val in x]

	# print(x)
	# print(y)

	# ax.plot(x, y, linewidth=2.0)

	# ax.set(xlim=(0, reward_data_obj.get_reward_data()), xticks=np.arange(1, 8),
	# 			ylim=(0, 5), yticks=np.arange(1, 8))

	# plt.show()

	print(len(frames), "	", len(rewards))

	grid_shape = np.shape(info[0][0])
	window = AnimateGrid(grid_shape[0] * PIXEL_SIZE, grid_shape[1] * PIXEL_SIZE, PIXEL_SIZE, RGB_DICT, frames[0],)
	window.animate(frames, rewards)