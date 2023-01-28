'''
Holds basic leaf class to be used in our BSP algorithm
'''

class Leaf():
	def __init__(self, width, height) -> None:
		self.width = width
		self.height = height
		self.left_child = None
		self.right_child = None
	
	def split(self) -> bool:
		if self.left_child is not None or self.right_child is not None:
			return False
		return True