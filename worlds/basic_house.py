import networkx as nx
import random as rd
import json
import numpy as np

class BasicHouse():
	"""
	Class for holding the data-structure representing a basic house environment.
	"""
	def __init__(self) -> None:
		self.__graph = nx.Graph()
		cont_data_f = open("worlds/container_data.json")
		self.__cont_data = json.load(cont_data_f)
		obj_data_f = open("worlds/object_data.json")
		self.__obj_data = json.load(obj_data_f)


		# Map between container type and color
		self.CONTAINER_COLOR_MAP = dict()
		self.CONTAINER_MAP = dict()
		for idx,CONT in enumerate(self.__cont_data["containers"]): 
			self.CONTAINER_COLOR_MAP[idx] = CONT["color"]
			self.CONTAINER_MAP[idx] = CONT["name"]

		# Assign probabilities for objects exisiting in a certain container
		self.CONTAINER_PROB = dict()
		
		for cont_code,cont_name in self.CONTAINER_MAP.items():
			OBJECT_PROB = dict()
			obj_list = np.array([obj_dat["name"] for obj_dat in self.__obj_data["objects"]])
			np.random.seed(seed=cont_code)
			obj_prob = np.random.dirichlet(np.ones(np.size(obj_list))*10) # parameter can be modified to determine skewedness
			for idx,OBJ in enumerate(obj_list):
				OBJECT_PROB[OBJ] = obj_prob[idx]
			self.CONTAINER_PROB[cont_name] = OBJECT_PROB
			# print(obj_list)
			# print("PROBABILITIES FOR ", cont_name.upper(), obj_prob)
		np.random.seed()


	def build_env(self, **kwargs):
		"""
		Constructs and returns the scene graph and grid representation for the environment.
		:param kwargs: Expects a list of room-functions. This is to specify which room graphs to generate
		and add to the environment's whole scene graph.
		"""

		self.__graph, self.HOUSE_GRID = self._living_room()

		return self.__graph,self.HOUSE_GRID


	def _living_room(self):
		
		# Generic representation of a living room
		self.LIVING_ROOM_CONT = {	"couch0"				: "couch",
															"coffee_table0" : "coffee table",
															"cabinet0" 			: "cabinet",
															"television0" 	: "television",
															"shelf0"				: "shelf",
															"shelf1"				: "shelf"
														}

		# Container occupancy grid for living room
		self.LIVING_ROOM_GRID = [	[0, 0, 1, 1, 1, 1, 0, 0],
															[0, 0, 0, 0, 0, 0, 0, 0],
															[0, 5, 0, 0, 0, 0, 5, 0],
															[0, 0, 0, 2, 2, 0, 0, 0],
															[3, 0, 0, 2, 2, 0, 0, 0],
															[3, 0, 0, 0, 0, 0, 0, 0],
															[0, 0, 4, 4, 4, 4, 0, 0],
															[0, 0, 4, 4, 4, 4, 0, 0]]

		room_graph = nx.Graph()
		room_graph.add_node("living room")

		for cont_k in self.LIVING_ROOM_CONT:
			index = 0
			for cont_dat in self.__cont_data["containers"]:
				if self.LIVING_ROOM_CONT[cont_k] == cont_dat["name"]:
					break
				index += 1
			
			# Hand-picked locations for each container
			location = (0, 0)
			if cont_k == "couch0":
				location = (6, 2)
			elif cont_k == "television0":
				location = (0, 2)
			elif cont_k == "cabinet0":
				location = (4, 0)
			elif cont_k == "coffee_table0":
				location = (3, 3)
			elif cont_k == "shelf0":
				location = (2, 1)
			elif cont_k == "shelf1":
				location = (2, 6)
			room_graph.add_nodes_from([(cont_k, {	"cost" 			: self.__cont_data["containers"][index]["cost"], 
																						"location"	: location})])
			room_graph.add_edge("living room", cont_k)

			# Add in objects by sampling CONT_PROB and space avilable
			if self.__cont_data["containers"][index]["space"] > 0:
				# print("SPACE IN ", cont_k,": ", self.__cont_data["containers"][index]["space"])
				for num_obj in range(np.random.randint(0,self.__cont_data["containers"][index]["space"])):
					OBJ_PROB = self.CONTAINER_PROB[self.__cont_data["containers"][index]["name"]]
					obj_name = list(OBJ_PROB.keys())[np.random.choice(range(len(list(OBJ_PROB.values()))),p=list(OBJ_PROB.values()))]
					node_name = cont_k + "_" +str(num_obj) + "_" + obj_name
					room_graph.add_nodes_from([(node_name, {"object" 			: obj_name})])
					room_graph.add_edge(cont_k, node_name)
			
		return room_graph, self.LIVING_ROOM_GRID

	def _bathroom(self):
		pass

	def _bedroom(self):
		pass

	def _kitchen(self):
		pass