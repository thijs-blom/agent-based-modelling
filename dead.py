import numpy as np
from obstacle import Obstacle


class Dead(Obstacle):
    """A one-dimensional wall, represented by a line-segment"""
    def __init__(self, pos: np.array, r: np.array):
        """Initialise a wall obstacle

        Note that both arguments must be one-dimensional arrays of size 2.

        Args:
            p1 (np.array): The first endpoint
            p2 (np.array): The second endpoint
        """
        self.pos = pos
        self.r = r

    def get_closest_point(self, other_point: np.array) -> np.array:
        """Compute the closest point of the obstacle to the given point"""

        # TODO return closest point from circle with pos and r
