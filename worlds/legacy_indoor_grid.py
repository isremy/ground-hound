import random as rd
import numpy as np
import networkx as nx
from worlds.legacy_room import Room

STATIC_WIDTH = 120
STATIC_HEIGHT = 50
SMALL_NUM_ROOMS = (3, 4, 5, 6, 8) # Factors of 120

INDOOR_RGB_MAP	= {-1:	(0, 0, 0),				# black
										0:	(170, 170, 170),	# light grey
										1:	(100, 100, 100),	# dark grey
										2:	(0, 10, 250),			# blue
										3:	(250, 0, 5),			# red 
										4:	(51, 102, 0)			# dark green
									}

class IndoorGrid:
	def __init__(self, size="small") -> None:
		self.__grid = np.array([[0] * STATIC_WIDTH for i in range(STATIC_HEIGHT)])
		self.__base_graph = nx.Graph()
		self.__base_graph.add_node(0, type="floor", height=STATIC_HEIGHT, width=STATIC_WIDTH, loc=(0,0))

	def generate_level(self):
		"""
		Creates a whole floor/level using basic room object and organizing into a larger grid-sspace
		"""
		self.__corridor_width = rd.randrange(6, 10, 2) # Width of hallway corridor of building
		self.__num_rooms = rd.choice(SMALL_NUM_ROOMS)
		self.__room_width = int(STATIC_WIDTH / self.__num_rooms)
		self.__room_height = int((STATIC_HEIGHT - self.__corridor_width) / 2)
		base_room = Room(self.__base_graph.nodes[0], self.__base_graph, width=self.__room_width, height=self.__room_height)
		base_room.create_room(2, 3)
		room = base_room.room()

		j = self.__room_height - 1
		for i in range(self.__room_height):
			self.__grid[i] = np.tile(room[i], self.__num_rooms)
			self.__grid[i + self.__corridor_width + self.__room_height] = np.tile(room[j], self.__num_rooms)
			j -= 1

		# print(self.__base_graph.nodes[0])
		return self.__grid
	
	def room_dims(self):
		"""
		Returns dimensions of a single high-level room
		"""
		return self.__room_width, self.__room_height
