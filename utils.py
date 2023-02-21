"""
Utility functions and A* path planner
"""

import numpy as np
import heapq

class Node:
	def __init__(self, x, y, f=float('inf'), g=float('inf')):
		self.x = x
		self.y = y
		self.f = f
		self.g = g
		self.parent = None

	def __lt__(self, other):
		return self.f < other.f


def a_star(grid, start, end):
	"""
	A* path planner. 
	:param grid: 	The occupancy grid of the environment. Free space should be a 0, obstacles are otherwise
	:param start: Tuple of start position, represented as (row, col)
	:param end:		Tuple of end position, represented as (row, col)
	:return: 			List of tuples representing each point in the A* path
	:rtype:				list
	"""
	open_list = []
	closed_list = set()

	start_node = Node(start[0], start[1], 0, 0)
	end_node = Node(end[0], end[1])

	heapq.heappush(open_list, start_node)

	while open_list:
		current_node = heapq.heappop(open_list)

		# if current_node.x == end_node.x and current_node.y == end_node.y:
		if _euclid_distance(current_node, end_node) < 2:	
			path = []
			while current_node:
				path.append((current_node.x, current_node.y))
				current_node = current_node.parent
			return path[::-1]

		closed_list.add((current_node.x, current_node.y))

		for neighbor in _get_neighbors(grid, current_node):
			if (neighbor.x, neighbor.y) in closed_list:
				continue

			tentative_g_score = current_node.g + 1
			if tentative_g_score < neighbor.g:
				neighbor.parent = current_node
				neighbor.g = tentative_g_score
				neighbor.f = tentative_g_score + _euclid_distance(neighbor, end_node)
				if neighbor not in open_list:
					heapq.heappush(open_list, neighbor)

	return None

def calc_dist(p1, p2):
	"""
	L2 distance function
	"""
	return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def find_min_rew(curr_point, rew_list):
	min = rew_list[0]
	for i in rew_list:
		if calc_dist(curr_point, i) < calc_dist(curr_point, min):
			min = i
	return min


def _euclid_distance(node1, node2):
	# return abs(node1.x - node2.x) + abs(node1.y - node2.y)
	return calc_dist((node1.x, node1.y), (node2.x, node2.y))


def _get_neighbors(grid, node):
	neighbors = []
	for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, 1), (1, -1)]:
		x2, y2 = node.x + dx, node.y + dy
		if 0 <= x2 < len(grid) and 0 <= y2 < len(grid[0]) and grid[x2][y2] == 0:
			neighbors.append(Node(x2, y2))
	return neighbors