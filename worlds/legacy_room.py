"""
Holds room class
"""
import networkx as nx
import random as rd
import numpy as np

class Room():
	def __init__(self, parent_node, base_graph, width=10, height=10, loc=(0,0)) -> None:
		self.__width = width
		self.__height = height
		self.__room_grid = np.array([[0] * self.__width for i in range(self.__height)])
		self.__base_graph = base_graph
		self.__parent_node = parent_node
		# base_graph.add_edge(parent_node)
		# self.__

	def room(self):
		return self.__room_grid


	def create_room(self, min_depth=2, max_depth=2):
		depth = rd.randint(min_depth, max_depth)
		self.__room_grid = self._partition_space(self.__room_grid, depth, 0, 0, self.__width, self.__height)
		self._add_borders()


	def _partition_space(self, world, depth, x1, y1, x2, y2) -> list:
		if depth == 0:
			# self.__base_node.add_node(type="room", height=y2-y1, width=x2-x1, origin = (y1, x1))
			return world

		if x2 - x1 <= 1 or y2 - y1 <= 1:
			return world

		if (y2 - y1) > (x2 - x1):
			avg = int((y1 + y2) / 2)
			split = rd.randint(avg - int((avg - y1) / 2), avg + int((y2 - avg) / 2))
			for i in range(x1, x2): world[split][i] = 1
			world = self._partition_space(world, depth-1, x1, y1, x2, split)
			world = self._partition_space(world, depth-1, x1, split, x2, y2)
		else:
			avg = int((x1 + x2) / 2)
			split = rd.randint(avg - int((avg - x1) / 2), avg + int((x2 - avg) / 2))
			for i in range(y1, y2): world[i][split] = 1
			world = self._partition_space(world, depth-1, x1, y1, split, y2)
			world = self._partition_space(world, depth-1, split, y1, x2, y2)
		
		return world

	
	def _add_borders(self):
		for i in range(self.__width):
			self.__room_grid[0][i] = 1
			self.__room_grid[self.__height - 1][i] = 1
		
		for i in range(1, self.__height - 1):
			self.__room_grid[i][0] = 1
			self.__room_grid[i][self.__width - 1] = 1