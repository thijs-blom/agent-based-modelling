import numpy as np

from .obstacle import Obstacle


class Wall(Obstacle):
    """A one-dimensional wall, represented by a line-segment"""
    def __init__(self, p1: np.ndarray, p2: np.ndarray):
        """Initialise a wall obstacle

        Note that both arguments must be one-dimensional arrays of size 2.

        Args:
            p1 (np.ndarray): The first endpoint
            p2 (np.ndarray): The second endpoint
        """
        self.p1 = p1
        self.p2 = p2

        self.width = np.linalg.norm(self.p1 - self.p2)
        self.width2 = self.width * self.width

    def get_closest_point(self, agent_point: np.ndarray) -> np.ndarray:
        """Compute the closest point of the obstacle to the given point"""
        d = np.dot(agent_point - self.p1, self.p2 - self.p1) / self.width2

        if d <= 0:
            return self.p1
        if d >= 1:
            return self.p2
        else:
            return self.p1 + d * (self.p2 - self.p1)
