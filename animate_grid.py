"""
Contains Animation class to visualize environment
"""
import pygame

WHITE = (200, 200, 200)

class AnimateGrid:
	def __init__(self, window_width, window_height, pixel_size, grid, rgb_dict) -> None:
		self.window_width = window_width
		self.window_height = window_height
		self.pixel_size = pixel_size
		self.grid = grid
		self.rgb_dict = rgb_dict

	def _draw_grid(self, grid, rgb_dict) -> None:
		"""
		Draws the current grid
		"""
		x_i = 0
		for x in range(0, self.window_width, self.pixel_size):
			y_i = 0
			for y in range(0, self.window_height, self.pixel_size):
				rect = pygame.Rect(x, y, self.pixel_size, self.pixel_size)
				pygame.draw.rect(SCREEN, rgb_dict[grid[x_i][y_i]], rect)
				pygame.draw.rect(SCREEN, WHITE, rect, 1)
				y_i += 1
			x_i += 1

	def animate(self) -> None:
		"""
		Updates the window and runs the main loop
		"""
		global SCREEN, CLOCK
		pygame.init()
		SCREEN = pygame.display.set_mode((self.window_width, self.window_height))
		CLOCK = pygame.time.Clock()
		
		while True:
			self._draw_grid(self.grid, self.rgb_dict)
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					return

			pygame.display.update()
	
	def update_grid(self, new_grid) -> None:
		self.grid = new_grid