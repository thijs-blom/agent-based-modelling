import numpy as np
from obstacle import Obstacle


class Dead(Obstacle):
    """A one-dimensional wall, represented by a line-segment"""
    def __init__(self, pos, r):
        """Initialise a wall obstacle

        Note that both arguments must be one-dimensional arrays of size 2.

        Args:
            pos: psoition
            r: radius
        """
        self.pos = np.array(pos)
        self.r = r

    def get_closest_point(self, other_point: np.array) -> np.array:
        """Compute the closest point of the obstacle to the given point"""

        # TODO return closest point from circle with pos and r
        return self.pos
