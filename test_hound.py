"""
Test harness for world generation
"""

import matplotlib.pyplot as plt
import networkx as nx
import utils
from worlds.legacy_basic_grid import *
from worlds.legacy_indoor_grid import *
from worlds.basic_house import *
from alpha_hound import Hound
from animate_grid import AnimateGrid
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

PIXEL_SIZE = 25	# Determines the size of a single square in the env grid


class TestHound():
	"""
	Contains a collection of functions that can be run to test
	components of the GroundHound training environment.
	"""

	def test_basic_grid(self):
		"""
		Tests generation of the BasicGrid level
		"""
		# basic_world = BasicGrid()
		basic_world = BasicGrid(grid_size=12, obs_size=2, obs_density="normal", seed=None)
		env = basic_world.generate_level()
		env_size = basic_world.get_size()
		
		assert len(env) == env_size
		print(basic_world.get_num_rewards())

		length = env_size * PIXEL_SIZE
		window = AnimateGrid(length, length, PIXEL_SIZE, 
													BASIC_RGB_MAP, env)
		window.animate()

	
	def test_basic_house(self):
		env = BasicHouse()
		
		# Test living room
		graph, grid = env._living_room()

		# Test A* planner
		start = (0, 0)
		end = (2, 6)
		path = utils.a_star(grid, start, end)
		print([point for point in path])

		nx.draw(graph, with_labels = True)
		plt.show()
		window = AnimateGrid(len(grid[0]) * PIXEL_SIZE, len(grid) * PIXEL_SIZE, PIXEL_SIZE, env.CONTAINER_COLOR_MAP, grid)
		window.animate()

		# Test kitchen

	def test_indoor_grid(self):
		"""
		Tests indoor grid environment
		"""
		print("\n")
		basic_world = IndoorGrid()
		env = basic_world.generate_level()
		print(env)

		# width, height = basic_world.room_dims()
		
		window = AnimateGrid(STATIC_WIDTH * PIXEL_SIZE, STATIC_HEIGHT * PIXEL_SIZE, PIXEL_SIZE, 
													INDOOR_RGB_MAP, env)
		window.animate()

	def test_hound(self):		
		test_hound = Hound(BasicHouse, "living room", "remote")
		
		check_env(test_hound, warn=True) 
		test_hound = make_vec_env(lambda: test_hound, n_envs=1)
		
		model = RecurrentPPO('MlpLstmPolicy', test_hound, verbose=1).learn(500_000)



