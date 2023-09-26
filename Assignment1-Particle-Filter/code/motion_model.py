'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """

    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.001
        self._alpha2 = 0.001
        self._alpha3 = 0.001
        self._alpha4 = 0.001

    def wrap(self, angle):
        """
        param[in] angle: an angle in radians

        param[out] : the same angle in range [-pi, pi]
        """
        return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))

    # def sample(self, b_squared):
    #     b = np.sqrt(b_squared)

    #     res = 0.0
    #     for i in range(12):
    #         res += np.random.uniform(low=-b, high=b)

    #     return res / 2

    def sample(self, mu, sigma):
        return np.random.normal(mu, sigma)

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]

        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """

        delta_rot1 = np.arctan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        delta_trans = np.sqrt((u_t0[0] - u_t1[0])**2 + (u_t0[1] - u_t1[1])**2)
        delta_rot2 = u_t1[2] - u_t0[2] - delta_rot1

        delta_rot1_hat = delta_rot1 - \
            self.sample(0, self._alpha1 * delta_rot1**2 +
                        self._alpha2 * delta_trans**2)
        delta_trans_hat = delta_trans - \
            self.sample(0, self._alpha3 * delta_trans**2 + self._alpha4 *
                        delta_rot1**2 + self._alpha4 * delta_rot2**2)
        delta_rot2_hat = delta_rot2 - \
            self.sample(0, self._alpha1 * delta_rot2**2 +
                        self._alpha2 * delta_trans**2)
        delta_rot1_hat = self.wrap(delta_rot1_hat)
        delta_rot2_hat = self.wrap(delta_rot2_hat)

        x_t1 = np.zeros(3)
        x_t1[0] = x_t0[0] + delta_trans_hat * np.cos(x_t0[2] + delta_rot1_hat)
        x_t1[1] = x_t0[1] + delta_trans_hat * np.sin(x_t0[2] + delta_rot1_hat)
        x_t1[2] = x_t0[2] + delta_rot1_hat + delta_rot2_hat
        x_t1[2] = self.wrap(x_t1[2])

        return x_t1
