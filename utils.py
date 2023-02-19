"""
Random utility functions
"""

import numpy as np

def calc_dist(p1, p2):
	return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

def find_min_rew(curr_point, rew_list):
	min = rew_list[0]
	for i in rew_list:
		if calc_dist(curr_point, i) < calc_dist(curr_point, min):
			min = i
	return min