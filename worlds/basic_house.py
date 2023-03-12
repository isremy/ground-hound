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

	def build_env(self, **kwargs):
		"""
		Constructs and returns the scene graph and grid representation for the environment.
		:param kwargs: Expects a list of room-functions. This is to specify which room graphs to generate
		and add to the environment's whole scene graph.
		"""

		# Make sure to record if no seed is passed in
		if 'seed' not in kwargs:
			kwargs['seed'] = None

		self.HOUSE_GRID, self.__graph = self._living_room(kwargs['seed'])

		return self.HOUSE_GRID, self.__graph
	

	def _living_room(self,seed=None):
		"""
		Randomly generates the grid and object graph for a living room environment
		"""
		if seed != None:
			np.random.seed(seed=seed)
			rd.seed(seed)

# add a binary signal for ecery container to indicate search status
# increase horizon limit to be rather large
		# Define min and max number of each furniture item
		num_rows = np.random.randint(10, 11)
		num_cols = np.random.randint(10, 11)

		# Specify range of occurances of each container
		self.LIVING_ROOM_CONT = {	"couch"					: [1, 2],	#[1,2],
															"coffee_table"	: [1, 2],	#[0,1],
															"cabinet" 			: [1, 2],	#[0,2],
															"television" 		: [1, 2],	#[1,1],
															"shelf"					: [2, 3]	#[0,3],
														}
		
		room_graph = nx.Graph()
		room_graph.add_node("living room")

		container_dims_dict = dict()
		for cont_k in self.LIVING_ROOM_CONT:
			indiv_containers = [cont_k + str(cont) for cont in range(np.random.randint(self.LIVING_ROOM_CONT[cont_k][0], 
									      																				self.LIVING_ROOM_CONT[cont_k][1]))]

			index = 0
			for cont_dat in self.__cont_data["containers"]:
				if cont_k == cont_dat["name"]:
					break
				index += 1

			for cont in indiv_containers:
				container_dims_dict[cont] = {"dimensions":	(self.__cont_data["containers"][index]["length"], 
				 															self.__cont_data["containers"][index]["width"]), "id": index}

		locations = [(row, col) for row in range (num_rows) for col in range(num_cols)]
		
		items = list(container_dims_dict.items())
		rd.shuffle(items)

		# Initialize the grid-world environment with None values
		grid = [[0 for j in range(num_cols)] for i in range(num_rows)]

		# Initialize a two-dimensional boolean array to keep track of occupied cells
		occupied = [[False for j in range(num_cols)] for i in range(num_rows)]

		# Fill grid with furniture and construct object-container graph
		for key, value in items:
			# print(key)
			placed = False
			while placed is False:
				location = rd.choice(locations)
				i, j = location[0], location[1]
				h, w = value["dimensions"]
				if i + h > num_rows or j + w > num_cols:  # check if the object fits within the grid
					continue
				overlaps = False
				for r in range(i, i+h):
					for c in range(j, j+w):
						if occupied[r][c]:
							overlaps = True
							break
					if overlaps:
						break

				if not overlaps:  # the object fits and does not overlap with any occupied cells
					placed = True
					room_graph.add_nodes_from([(key, {	"cost" 			: self.__cont_data["containers"][value["id"]]["cost"], 
																						"location"	: location})])
					room_graph.add_edge("living room", key)

					# Add in objects by sampling CONT_PROB and space avilable
					if self.__cont_data["containers"][value["id"]]["space"] > 0:
						for num_obj in range(np.random.randint(0,self.__cont_data["containers"][value["id"]]["space"])):
							OBJ_PROB = self.CONTAINER_PROB[self.__cont_data["containers"][value["id"]]["name"]]
							obj_name = list(OBJ_PROB.keys())[np.random.choice(range(len(list(OBJ_PROB.values()))),p=list(OBJ_PROB.values()))]
							node_name = key + "_" +str(num_obj) + "_" + obj_name
							room_graph.add_nodes_from([(node_name, {"object" 			: obj_name})])
							room_graph.add_edge(key, node_name)

					for r in range(i, i+h):
						for c in range(j, j+w):
							occupied[r][c] = True
							grid[r][c] = value["id"]

		return grid, room_graph


	def _living_room_old(self):
		"""
		DEPRECATED
		"""
		# Generic representation of a living room
		self.LIVING_ROOM_CONT = {	"couch0"				: "couch",
															"coffee_table0" : "coffee_table",
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
			
		return self.LIVING_ROOM_GRID, room_graph


	def _bathroom(self):
		pass

	def _bedroom(self):
		pass

	def _kitchen(self):
		pass