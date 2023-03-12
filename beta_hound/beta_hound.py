import gym
import numpy as np
from utils import *
# from ground-hound.utils import *
from gym import spaces
import matplotlib.pyplot as plt

class Hound(gym.Env):
	"""
	Hound is a custom environment class that defines how the agent interacts with its
	environment, utilizing the OpenAI gym class.
	:param env: Environment class to use for scene graph and occupancy grid
	:param scene_parent: The root node of the scene graph to start from (such as a room or building)
	:param target_obj: Object to locate and be rewarded for finding.
	"""
	def __init__(self, env, scene_parent, target_obj, reward_callback=None) -> None:
		super(Hound, self).__init__()
		# TODO:	Alternative approach to terminal state definition: specifying number of target objects
		#				and then defining the terminal state to be when all objects are found.
		#				The current method simply has the agent try to maximize its reward from object search, rather than maximize
		#				objects found.
		# TODO: Another approach to observations: try using scene graph adjacency matrix instead of 
		# 			container locations as observation.
		self.__env_type = env
		init_env = env()
		self.__reward_callback=None
		self.__grid, self.__scene_graph = init_env.build_env()
		self.__scene_parent = scene_parent
		self.__containers_list = [edge[1] for edge in self.__scene_graph.edges(self.__scene_parent)] # need to change this when adding high-level building node
		self.__cont_locations = [self.__scene_graph.nodes[container]["location"] for container in self.__containers_list]
		self.__num_containers = len(self.__containers_list)
		self.__target_obj = target_obj
		self.__grid_shape = np.shape(self.__grid)
		self.__start_pos = (4, 7)
		self.__curr_pos = self.__start_pos
		self.__num_cont_visited = 0
		self.__grid[self.__start_pos[0]][self.__start_pos[1]] = -1 # Initialize agent on grid.
		self.__actions_taken = [0] * self.__num_containers

		self.__cum_reward = 0
		if reward_callback is not None:
			self.__reward_callback = reward_callback

		self.action_space = spaces.Discrete(self.__num_containers + 1)
		# observation = [grid space, agent-pos, action last taken, container-0-location,...,reward-n-location]
		self.observation_space = spaces.Box(low=np.array([-2] * self.__grid_shape[0] * self.__grid_shape[1] + [0] + [0] * (2 + 2 * self.__num_containers) + [0] * self.__num_containers), 
																				high=np.array([20] * self.__grid_shape[0] * self.__grid_shape[1] + [self.__num_containers] + [self.__grid_shape[0]] * (1 + self.__num_containers) + [self.__grid_shape[1]] * (1 + self.__num_containers) + [1] * self.__num_containers), 
																				shape=(3 + 3 * self.__num_containers + (self.__grid_shape[0] * self.__grid_shape[1]),), dtype=np.int32)


	def reset(self):
		self.__curr_pos = self.__start_pos
		init_env = self.__env_type()
		self.__grid, self.__scene_graph = init_env.build_env()
		self.__containers_list = [edge[1] for edge in self.__scene_graph.edges(self.__scene_parent)]
		self.__num_containers = len(self.__containers_list)
		self.__cont_locations = [self.__scene_graph.nodes[container]["location"] for container in self.__containers_list]
		self.__grid_shape = np.shape(self.__grid)
		self.__start_pos = (4, 7)
		self.__curr_pos = self.__start_pos
		self.__num_cont_visited = 0
		self.__cum_reward = 0
		self.__grid[self.__curr_pos[0]][self.__curr_pos[1]] = -1 # Initialize agent on grid.
		self.__actions_taken = [0] * self.__num_containers
		action = self.__num_containers

		# self.action_space = spaces.Discrete(self.__num_containers)
		# # observation = [grid space, agent-pos, container-0-location,...,reward-n-location]
		# self.observation_space = spaces.Box(low=np.array([-2] * self.__grid_shape[0] * self.__grid_shape[1] + [0] * (2 + 2 * self.__num_containers)), 
		# 																		high=np.array([20] * self.__grid_shape[0] * self.__grid_shape[1] + [self.__grid_shape[0]] * (1 + self.__num_containers) + [self.__grid_shape[1]] * (1 + self.__num_containers)), 
		# 																		shape=(2 + 2 * self.__num_containers + (self.__grid_shape[0] * self.__grid_shape[1]),), dtype=np.int32)

		flattened_grid = (np.ndarray.flatten(np.array(self.__grid))).tolist()

		obsv = flattened_grid + [action] + [self.__start_pos[0], self.__start_pos[1]] + (np.ndarray.flatten(np.array(self.__cont_locations))).tolist() + self.__actions_taken
		# obsv = flattened_grid + [action] + [self.__start_pos[0], self.__start_pos[1]] + (np.ndarray.flatten(np.array(self.__cont_locations))).tolist()

		# print(np.shape(obsv))
		return np.array(obsv).astype(np.int32)


	def step(self, action):
		reward = 0
		self.__num_cont_visited += 1
		done = False
		
		path = [self.__curr_pos]
		container_status = ["NaN", "NaN"]

		# If we've taken this action before and it's not a no-action, penalize
		# if action in self.__actions_taken and action != self.__num_containers:
		# 	reward -= 1

		# self.__actions_taken.add(action)

		

		if action < self.__num_containers:
			self.__actions_taken[action] = 1
			location = self.__cont_locations[action]
			# Check that there is no confusion about which container is being picked.
			assert location == self.__scene_graph.nodes[self.__containers_list[action]]["location"]

			cost = self.__scene_graph.nodes[self.__containers_list[action]]["cost"]

			path = a_star(self.__grid, self.__curr_pos, location)
			self.__grid[self.__curr_pos[0]][self.__curr_pos[1]] = 0
			self.__curr_pos = path[-1]
			self.__grid[self.__curr_pos[0]][self.__curr_pos[1]] = -1
			
			
			
			# TODO: Investigate tweaking penalties for removing occlusion and path planning
			reward -= 0.1 * len(path)
			# reward -= cost

			remove_obj = []
			obj_contained = False
			container_status = [self.__containers_list[action], False]

			for edge in self.__scene_graph.edges(self.__containers_list[action]):
				if self.__target_obj in edge[1]:
					# TODO: maybe play a little animation when a reward is found?
					container_status[1] = True
					obj_contained = True
					reward += 1
					remove_obj.append(edge[1])

			if obj_contained:
				self.__scene_graph.remove_nodes_from(remove_obj)

		# TODO: Investigate using number of target objects as terminal state.
		if self.__num_cont_visited == 50:
			done = True

		self.__cum_reward += reward
		if self.__reward_callback is not None:
			self.__reward_callback.callback(self.__cum_reward)

		flattened_grid = (np.ndarray.flatten(np.array(self.__grid))).tolist()
		obsv = flattened_grid + [action] + [self.__start_pos[0], self.__start_pos[1]] + (np.ndarray.flatten(np.array(self.__cont_locations))).tolist() + self.__actions_taken

		# Auxillary information dictionary
		info = {0:self.__grid, 1:path, 2:container_status, 3:self.__scene_graph}

		# print(np.shape(obsv))
		return np.array(obsv).astype(np.int32), reward, done, info


	def render(self):
		"""
		Required function for custom gym subclass, but implementation optional.
		"""
		pass


	def close(self):
		"""
		Required function for custom gym subclass, but implementation optional.
		"""
		pass