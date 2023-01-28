import random as rd
from leaf import Leaf

class IndoorGrid:
	def __init__(self, min_rooms, max_rooms) -> None:
		super.__init__()
		self.min_rooms = min_rooms
		self.max_rooms = max_rooms

	def generate_level(self):
		self.num_rooms = rd.randrange(self.min_rooms, self.max_rooms)
		

	def _base_room(self):
		pass
