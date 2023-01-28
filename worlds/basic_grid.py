"""
Contains BasicGrid environment class
"""

import random as rd

class BasicGrid:
	"""
	This basic grid object will generate a square level of randomly placed obstacles
	and rewards near obstacles.
	"""
	def __init__(self, grid_size=15, obs_size=2, obs_density="normal", seed=None) -> None:
		self.__grid_size = grid_size
		self.__obs_size = obs_size
		self.__density = obs_density
		self.__seed = seed
		self.__grid = [[0] * self.__grid_size for i in range(self.__grid_size)]
		self.__densities_map = {"sparse": self.__grid_size / (self.__obs_size * 1), 
														"normal": self.__grid_size / (self.__obs_size * 0.5), 
														"dense":  self.__grid_size / (self.__obs_size * 0.25)
												 	 }

		self.state_rgb_map = {-1:	(0, 0, 0),				# black
													0:	(0, 200, 10),			# light green
													1:	(165, 165, 165),	# dark grey
													2:	(0, 10, 250),			# blue
													3:	(250, 0, 5),			# red 
													4:	(51, 102, 0)			# dark green
												 }


	def generate_level(self) -> list:
		"""
		Generates a grid for this GridWorld object, including the
		obstacles and rewards.
		"""
		max_obs = self.__densities_map[self.__density]

		if self.__seed: rd.seed(self.__seed)

		num_obs = rd.randrange(int(max_obs / 2), int(max_obs))
		obs_set = set()

		for i in range(num_obs):
			row = rd.randrange(self.__obs_size - 1, self.__grid_size - self.__obs_size)
			col = rd.randrange(self.__obs_size - 1, self.__grid_size - self.__obs_size)
			self.__grid[row][col] = 1
			obs_set.add((row, col))
		
		for obs in obs_set:
			for i in range(self.__obs_size):
				for j in range(self.__obs_size):
					self.__grid[obs[0] + i][obs[1] + j] = 1
			
		for i in range(int(num_obs / 2)):
			row = rd.randrange(self.__obs_size - 1, self.__grid_size - self.__obs_size)
			col = rd.randrange(self.__obs_size - 1, self.__grid_size - self.__obs_size)
			
			def is_obs_adj(row, col) -> bool:
				"""
				Checks that given point is adjacent to an obstacle
				"""
				for j in range(-1, 1):
					for k in range(-1, 1):
						if self.__grid[row + j][col + k] == 1:
							return True
				return False

			while(self.__grid[row][col] == 1 or not is_obs_adj(row, col)):
				row = rd.randrange(self.__obs_size - 1, self.__grid_size - self.__obs_size)
				col = rd.randrange(self.__obs_size - 1, self.__grid_size - self.__obs_size)
			
			self.__grid[row][col] = 2
		return self.__grid

	def get_rgb_dict(self) -> dict:
		"""
		Returns a dict of what color a pixel's state corresponds to as an RGB tuple
		"""
		return self.state_rgb_map

	def get_size(self) -> int:
		"""
		Returns grid-size along a side
		"""
		return self.__grid_size