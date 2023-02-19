import networkx as nx
import random as rd
import json
import numpy as np

LIVING_ROOM_OBJ = {"couch0"				: "couch",
									"coffee-table0" : "coffee table",
									"cabinet0" 			: "cabinet",
									"television0" 	: "television",
									"speaker0"			: "speaker",
									"speaker1"			: "speaker"}

KITCHEN_OBJ 		=	{"oven", "sink", "cabinet", "shelf", "dishwasher", "table"}
BATHROOM_OBJ		= {"sink", "toilet", "bathtub", "shelf"}
BEDROOM_OBJ			= {"bed", "desk", "shelf", "house plant"}

HOUSE_COLOR_MAP	= {	0: (210, 180, 140),	# Tan
										1: (0, 0, 0),			 	# Black
										2: (135, 206, 235),	# Light blue
										3: (92, 64, 51), 		# Dark brown
										4: (3, 37, 126)} 		# Dark blue

LIVING_ROOM_GRID = [[0, 0, 1, 1, 1, 1, 0, 0],
										[0, 0, 0, 0, 0, 0, 0, 0],
										[0, 1, 0, 0, 0, 0, 1, 0],
										[0, 0, 0, 2, 2, 0, 0, 0],
										[3, 0, 0, 2, 2, 0, 0, 0],
										[3, 0, 0, 0, 0, 0, 0, 0],
										[0, 0, 4, 4, 4, 4, 0, 0],
										[0, 0, 4, 4, 4, 4, 0, 0]]

# KITCHEN_GRID = [[]]

# LIVING_ROOM_W = (8, 12)
# LIVING_ROOM_L = (8, 12)

class BasicHouse():
	"""
	Class for holding the data-structure representing a basic house environment.
	"""
	def __init__(self) -> None:
		self.__graph = nx.Graph()
		obj_data_f = open("worlds/object_data.json")
		self.__obj_data = json.load(obj_data_f)

	def build_env_grph(self, **kwargs):
		"""
		Constructs and returns the scene graph for the environment.
		:param: kwargs takes in a list of room-functions. This is to specify which room graphs to generate
		and add to the environment's whole scene graph.
		"""


		pass

	def _living_room(self):
		# grid_w = rd.randint(LIVING_ROOM_W[0], LIVING_ROOM_W[1])
		# grid_h = rd.randint(LIVING_ROOM_L[0], LIVING_ROOM_L[1])
		# room_grid = np.array([[0] * grid_w for i in range(grid_h)])
		room_graph = nx.Graph()
		room_graph.add_node("living room")
		index = 0
		for obj_k in LIVING_ROOM_OBJ:
			if LIVING_ROOM_OBJ[obj_k] in [obj_dat["name"] for obj_dat in self.__obj_data["objects"]]:
				
				location = (0, 0)
				if obj_k == "couch0":
					location = (0, 0)
				elif obj_k == "television0":
					location = (0, 0)
				elif obj_k == "cabinet0":
					location = (0, 0)
				elif obj_k == "coffee-table0":
					location = (0, 0)
				elif obj_k == "speaker0":
					location = (0, 0)
				elif obj_k == "speaker1":
					location = (0, 0)
				room_graph.add_nodes_from([(obj_k, {"occlusion" : self.__obj_data["objects"][index]["occlusion"],
																						"cost" 			: self.__obj_data["objects"][index]["cost"], 
																						"location"	: location})])
				room_graph.add_edge("living room", obj_k)
			
			index += 1


		return room_graph, LIVING_ROOM_GRID

	def _bathroom(self):
		pass

	def _bedroom(self):
		pass

	def _kitchen(self):
		pass