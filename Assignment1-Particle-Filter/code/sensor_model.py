'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
import os
from matplotlib import pyplot as plt
from scipy.stats import norm
import multiprocessing as mp

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 0.85
        self._z_short = 0.3
        self._z_max = 0.1
        self._z_rand = 500

        self._sigma_hit = 100
        self._lambda_short = 0.1

        self._occupancy_map = occupancy_map

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 8183

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2

        self._laser_offset = 25
        self._resolution = 10

        self._ray_table = np.zeros(
            (occupancy_map.shape[0], occupancy_map.shape[1], 360), dtype=np.float64)

        if os.path.isfile('./ray_table.npy'):
            self._ray_table = np.load('./ray_table.npy')
        else:
            self.initialize_ray_table()

    def ray_cast_precomputing(self, x, y, step=250):
        """
        param[in] x_laser : x coordinate of the specified position
        param[in] y_laser : y coordinate of the specified position
        param[in] theta : angle in degrees

        param[out] : the true laser range measurement
        """

        # Initialize the true laser range to the maximum range
        z_stars = np.ones(360) * self._max_range
        dist_step = np.linspace(0, self._max_range, step)

        h, w = self._occupancy_map.shape
        for theta in range(360):
            # degree to radian
            theta_laser = theta * np.pi / 180
            x_laser = x + np.cos(theta_laser) * dist_step
            y_laser = y + np.sin(theta_laser) * dist_step

            zx = (x_laser / self._resolution).astype(int)
            zy = (y_laser / self._resolution).astype(int)

            for i in range(len(zx)):
                if 0 <= zx[i] < w and 0 <= zy[i] < h:
                    if self._occupancy_map[zy[i], zx[i]] >= self._min_probability or \
                            self._occupancy_map[zy[i], zx[i]] == -1:
                        z_stars[theta] = np.sqrt(
                            (x_laser[i] - x)**2 + (y_laser[i] - y)**2)
                        break
                else:
                    break

        return z_stars

    # def run_parallel(self, operation, input, pool):
    #     pool.map(operation, input)

    def initialize_ray_table(self):

        h, w = self._occupancy_map.shape
        self._ray_table = np.zeros((h, w, 360), dtype=np.float64)

        # Iterate over the occupancy map
        for x_laser in range(w):
            for y_laser in range(h):
                # Only iterate over the obstacle-free positions in the map
                # Obstacle-free positions are positions where the occupancy is
                # lower than the threshold [min_probability]
                if self._occupancy_map[y_laser, x_laser] != 0:
                    continue
                # Ray cast 360 laser beams from the obstacle-free position,
                # which span 360 degrees
                self._ray_table[y_laser, x_laser] = self.ray_cast_precomputing(
                    x_laser * self._resolution, y_laser * self._resolution)

        # save
        np.save('./ray_table.npy', self._ray_table)

    def wrap(self, angle):
        """
        param[in] angle: an angle in radians

        param[out] : the same angle in range [-pi, pi]
        """
        return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))

    def get_probability(self, z_stars, z_readings):
        # p_hit
        eta = norm.cdf(self._max_range, loc=z_stars, scale=self._sigma_hit) - \
            norm.cdf(0, loc=z_stars, scale=self._sigma_hit)
        p_hit = norm.pdf(z_readings, loc=z_stars, scale=self._sigma_hit) / eta
        p_hit[z_readings > self._max_range] = 0
        p_hit[z_readings < 0] = 0

        # p_short
        eta = np.zeros_like(z_readings)
        eta[z_stars != 0] = 1 / \
            (1 - np.exp(-self._lambda_short * z_stars[z_stars != 0]))
        p_short = eta * self._lambda_short * \
            np.exp(-self._lambda_short * z_readings)
        p_short[np.where((z_readings < 0) & (z_readings > z_stars))] = 0

        # p_max
        p_max = np.zeros_like(z_readings)
        p_max[z_readings == self._max_range] = 1.0

        # p_rand
        p_rand = np.zeros_like(z_readings)
        p_rand[np.where((z_readings >= 0) & (z_readings < self._max_range))
               ] = 1 / self._max_range

        prob_zt1 = self._z_hit * p_hit + self._z_short * \
            p_short + self._z_max * p_max + self._z_rand * p_rand

        prob_zt1 = np.delete(prob_zt1, np.where(prob_zt1 == 0.0))
        prob_zt1 = np.sum(np.log(prob_zt1))

        return np.exp(prob_zt1)

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """

        x, y, theta = x_t1

        # The laser is 25cm offset forward from the true center of the robot.
        # Calculate the laser origin:
        x_laser = int(x + self._laser_offset *
                      np.cos(theta)) // self._resolution
        y_laser = int(y + self._laser_offset *
                      np.sin(theta)) // self._resolution
        # Scan a range of 180 degrees
        # Note that he base of the range scan sector is perpendicular
        # to the line connecting the origin and the robot.
        # Therefore, `theta_laser` is calculated as theta + deg * pi / 180 - pi / 2
        theta_laser = [theta + deg * np.pi /
                       180 - np.pi / 2 for deg in range(180)]
        theta_laser = [(np.degrees(theta) % 360).astype(int)
                       for theta in theta_laser]
        z_stars = np.array([self._ray_table[y_laser, x_laser, theta]
                           for theta in theta_laser])

        # print("z_readings shape: ", z_t1_arr.shape)
        # print("z_stars shape: ", z_stars.shape)

        probs = self.get_probability(z_stars, z_t1_arr)
        # probs = np.delete(probs, np.where(probs == 0.0))
        # prob_zt1 = np.sum(np.log(probs))

        return probs, z_stars
