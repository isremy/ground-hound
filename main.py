from alpha_hound.alpha_hound import Hound
# from beta_hound.beta_hound import Hound

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import os
import csv
import argparse
from animate_grid import AnimateGrid
from worlds.basic_house import BasicHouse
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.logger import Figure


NUM_RUNS = 50


class RewardCollectionCallback(BaseCallback):
	def __init__(self, rewards: list, check_freq: int, log_dir: str, verbose: int = 1) -> None:
		super(RewardCollectionCallback, self).__init__(verbose)
		self.check_freq = check_freq
		self.log_dir = log_dir
		self.save_path = os.path.join(log_dir, "r_log")
		self.rewards = rewards
	
	def _init_callback(self) -> None:
		# Create folder if needed
		if self.save_path is not None:
			os.makedirs(self.save_path, exist_ok=True)
	
	
	def _on_step(self) -> bool:
		if self.n_calls % self.check_freq == 0:
			x, y = ts2xy(load_results(self.log_dir), "timesteps")
			if len(x) > 0:
				mean_reward = np.mean(y[-1 * self.check_freq:])
				self.rewards.append(mean_reward)
		
		return True


ARB_STEPS = 50
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


def train(num_steps: int=10_000) -> None:
	"""
	Performs one learning run over some number of timesteps
	"""
	
	mean_rewards_list = []
	log_dir = "models/alpha_d_0.0.1/"
	reward_data_obj = RewardCollectionCallback(mean_rewards_list, 200, log_dir)
	test_hound = Hound(BasicHouse, "living room", "magazine")
	check_env(test_hound, warn=True) 

	test_hound = Monitor(test_hound, log_dir)

	# model = RecurrentPPO('MlpLstmPolicy', test_hound, verbose=1)
	model = PPO('MlpPolicy', test_hound, verbose=1)
	model.learn(num_steps, reward_data_obj)
	model.save("models/alpha_d_0.0.1/alpha_d_0.0.1_model")
	
	timestep_len = len(mean_rewards_list)

	plot = pd.DataFrame({"Average Episode Reward": mean_rewards_list, "Timesteps": np.linspace(0, num_steps, timestep_len)})
	
	sns.set(style="darkgrid", font_scale=0.8)
	sns.lineplot(x="Timesteps", y = "Average Episode Reward", data=plot, errorbar="ci", err_style="band")

	plt.savefig("models/alpha_d_0.0.1/reward_plot.png")
	# plot_results([log_dir], num_steps, results_plotter.X_TIMESTEPS, "TD3 LunarLander")

	plt.show()


def test_model(visuals: bool = False) -> list:
	# reward_data_obj = RewardCollectionCallback()
	test_hound = Hound(BasicHouse, "living room", "magazine")
	env = test_hound
	test_hound = make_vec_env(lambda: test_hound, n_envs=1)
	model = PPO.load("models/alpha_nd_0.0.1/alpha_nd_0.0.1_model", env)
	
	vec_env = model.get_env()
	mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
	print("MEAN REWARD: ", mean_reward)
	obs = vec_env.reset()

	obs = test_hound.reset()

	frames = []
	rewards = []
	visited_containers = []
	total_reward = 0
	done = False
	idx = 0

	for i in range(ARB_STEPS):
		#TODO: Try non deterministic actions
		action, lstm_states = model.predict(obs,  deterministic=True)
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
			if reward > -1:
				rewards.append(reward)
			else:
				rewards.append(0)
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

	if visuals:
		nx.draw(info[0][3], with_labels=True)
		plt.show()
		grid_shape = np.shape(info[0][0])
		window = AnimateGrid(grid_shape[0] * PIXEL_SIZE, grid_shape[1] * PIXEL_SIZE, PIXEL_SIZE, RGB_DICT, frames[0],)
		window.animate(frames, rewards)

	return [container_status for container_status in visited_containers]


def plot_search_histogram() -> None:
	"""
	Read all the learned search policies from the CSV and display a histogram
	of the search policy sequences, which are tuples.
	"""
	data_set = []
	with open("./runs_data.csv", "r") as file:
		csv_reader = csv.reader(file)
		next(csv_reader)
		for row in csv_reader:
			data_set.append(tuple(row[1:-1]))
		file.close()
	
	# Create a dictionary to count the occurrences of each tuple
	counts = {}
	for t in data_set:
			counts[t] = counts.get(t, 0) + 1

	# Get the unique tuples and their counts
	unique_tuples, counts = zip(*counts.items())

	# Sort the tuples and counts based on the counts
	sorted_tuples_counts = sorted(zip(unique_tuples, counts), key=lambda x: x[1], reverse=True)
	sorted_tuples, sorted_counts = zip(*sorted_tuples_counts)
	sorted_tuples2 = [str(item) for item in sorted_tuples]

	# Create the figure and axis objects
	fig, ax = plt.subplots()

	# Plot the frequencies as bars
	ax.bar(sorted_tuples2, sorted_counts, color='purple')

	# Set the x-axis tick labels
	short_labels = [f"({t[0]})" for t in sorted_tuples]
	ax.set_xticklabels(short_labels)

	# Decrease the font size of the x-labels
	ax.tick_params(axis='x', which='major', labelsize=5)

	# Set the y-axis range
	ax.set_ylim([0, max(sorted_counts) + 1])

	# Add labels and title
	ax.set_xlabel('Tuple')
	ax.set_ylabel('Frequency')
	ax.set_title('Occurrences of Tuples in a List (Sorted by Frequency)')

	# Show the plot
	plt.show()


if __name__ == "__main__":
	"""
	Save data on search policies from many learning runs
	"""
	parser = argparse.ArgumentParser(description="Log search policy over a large number of training sessions")
	parser.add_argument('--log', help="Specify that you want to log over large number of learned search policies")
	args, leftovers = parser.parse_known_args()

	if args.log is not None:
		with open("./runs_data.csv", "w") as file:
			tests_data = dict()
			writer = csv.writer(file)
			writer.writerow(["Iteration", "Action_1", "Action_2", "Action_3", "Action_4", "Action_5", "Action_6"])
			
			for iter in range(NUM_RUNS):
				test_data = test_model()
				tests_data[iter] = test_data
				writer.writerow([iter + 1] + test_data)

			file.close()

		plot_search_histogram()
	
	else:
		train(num_steps=100_000)
		# test_model(visuals=True)