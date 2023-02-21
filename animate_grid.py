"""
Contains Animation class to visualize environment
"""
import pygame
import numpy as np

WHITE = (200, 200, 200)

class AnimateGrid:
	def __init__(self, window_width, window_height, pixel_size, rgb_dict, grid=None) -> None:
		self.__window_width = window_width
		self.__window_height = window_height
		self.__pixel_size = pixel_size
		self.__grid = grid
		self.__rgb_dict = rgb_dict

	def animate(self, frames=[]) -> None:
		"""
		Updates the window and runs the main loop
		"""
		global SCREEN, CLOCK
		pygame.init()
		SCREEN = pygame.display.set_mode((self.__window_width, self.__window_height))
		CLOCK = pygame.time.Clock()
		
		num_frames = len(frames)
		i = 0
		while True:
			if i < num_frames:
				self.__grid = frames[i]
				i += 1

			self._draw_grid()
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					return

			pygame.display.update()
			pygame.time.wait(500)

	def _draw_grid(self) -> None:
		"""
		Draws the current grid
		"""
		y_i = 0
		for x in range(0, self.__window_height, self.__pixel_size):
			x_i = 0
			for y in range(0, self.__window_width, self.__pixel_size):
				rect = pygame.Rect(y, x, self.__pixel_size, self.__pixel_size)
				pygame.draw.rect(SCREEN, self.__rgb_dict[self.__grid[y_i][x_i]], rect)
				pygame.draw.rect(SCREEN, WHITE, rect, 1)
				x_i += 1
			y_i += 1
	
	def update_grid(self, new_grid) -> None:
		self.grid = new_grid