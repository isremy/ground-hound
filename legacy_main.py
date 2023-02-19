from legacy_hound import Hound
from animate_grid import AnimateGrid
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

ARB_STEPS = 50
PIXEL_SIZE = 20
RGB_DICT = {-1:	(0, 0, 0),				# black
							0:	(0, 200, 10),			# light green
							1:	(165, 165, 165),	# dark grey
							2:	(0, 10, 250),			# blue
							3:	(250, 0, 5),			# red 
							4:	(51, 102, 0)			# dark green
							}

if __name__ == '__main__':
	test_hound = Hound()
	check_env(test_hound, warn=True) 
	test_hound = make_vec_env(lambda: test_hound, n_envs=1)
	model = DQN('MlpPolicy', test_hound, verbose=1, exploration_final_eps=0.2).learn(200_000)
	obs = test_hound.reset()
	obs_size = len(obs.tolist())

	frames = []
	total_reward = 0
	done = False

	for i in range(ARB_STEPS):
		action, state = model.predict(obs, deterministic=True)
		obs, reward, done, info = test_hound.step(action)
		total_reward += reward[0]
		# print(obs)
		frames.append(info[0][0])

		if done:
			print("Finished!")
			break

	if not done:
		print("Did not complete course :(")
	
	print("Reward: ", total_reward)
	# print(obs[0])
	length = PIXEL_SIZE * len(info[0][0])
	window = AnimateGrid(length, length, PIXEL_SIZE, RGB_DICT, info[0][0])
	# print(len(obs[0]))
	window.animate(frames)