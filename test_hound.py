"""
Test harness for world generation
"""

from worlds.basic_grid import BasicGrid
from animate_grid import AnimateGrid


PIXEL_SIZE = 35	# Determines the size of a single square in the env grid


class TestHound():
	"""
	Contains a collection of functions that can be run to test
	components of the GroundHound training environment.
	"""

	def test_basic_grid(self):
		"""
		Tests generation of the BasicGrid level
		"""
		basic_world = BasicGrid()
		env = basic_world.generate_level()
		env_size = basic_world.get_size()
		
		assert len(env) == env_size
		length = env_size * PIXEL_SIZE
		window = AnimateGrid(length, length, PIXEL_SIZE, env, 
													basic_world.get_rgb_dict())
		window.animate()
		