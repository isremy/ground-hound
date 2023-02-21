from alpha_hound import Hound
import numpy as np
import copy
from animate_grid import AnimateGrid
from worlds.basic_house import BasicHouse
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

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
	test_hound = Hound(BasicHouse, "living room", "dvd")
	check_env(test_hound, warn=True) 
	test_hound = make_vec_env(lambda: test_hound, n_envs=1)
	model = RecurrentPPO('MlpLstmPolicy', test_hound, verbose=1).learn(10_000)
	obs = test_hound.reset()
	obs_size = len(obs.tolist())

	frames = []
	visited_containers = []
	total_reward = 0
	done = False

	idx = 0
	for i in range(ARB_STEPS):
		action, state = model.predict(obs, deterministic=True)
		obs, reward, done, info = test_hound.step(action)

		visited_containers.append(info[0][2])

		total_reward += reward[0]

		# TODO: Color all locations agent has traveled to make it easier to
		#				visualize path
		local = copy.deepcopy(info[0][0])
		local[info[0][1][-1][0]][info[0][1][-1][1]] = 0
		idx = 0
		for loc in info[0][1]:
			local[loc[0]][loc[1]] = -1
			frames.append(copy.deepcopy(local))
			local[loc[0]][loc[1]] = 0

		idx += 1
		if done:
			print("Finished!")
			break

	if not done:
		print(idx)
		print("Did not complete course :(")

	print("Reward: ", total_reward)
	print("LOCATIONS: ", visited_containers)
	grid_shape = np.shape(info[0][0])
	window = AnimateGrid(grid_shape[0] * PIXEL_SIZE, grid_shape[1] * PIXEL_SIZE, PIXEL_SIZE, RGB_DICT, frames[0])
	window.animate(frames)