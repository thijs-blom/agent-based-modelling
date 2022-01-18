import numpy as np
from obstacle import Obstacle


class Wall(Obstacle):
    """A one-dimensional wall, represented by a line-segment"""
    def __init__(self, p1: np.array, p2: np.array):
        """Initialise a wall obstacle

        Note that both arguments must be one-dimensional arrays of size 2.

        Args:
            p1 (np.array): The first endpoint
            p2 (np.array): The second endpoint
        """
        self.p1 = p1
        self.p2 = p2

    def get_closest_point(self, other_point: np.array) -> np.array:
        """Compute the closest point of the obstacle to the given point"""
        line_vec = self.p2 - self.p1
        # Compute a normal of p2 - p1
        v = np.copy(np.flip(line_vec))
        v[0] *= -1.0

        # Solve the system defined by p1 + a*(p2-p1) = other_point + b*v)
        # TODO: check if this efficient, or can be done in a clearer way
        A = np.stack((line_vec, -v), axis=1)
        res = np.linalg.solve(A, other_point - self.p1)

        # Check whether the project of other_point on the (infinite) line
        # through p1 and p2 is between the endpoints. If so, return the
        # projection. Otherwise, return the relevant endpoint as the
        # closest point.
        alpha = res[0]
        if alpha >= 1:
            return self.p2
        elif alpha <= 0:
            return self.p1
        else:
            return self.p1 + alpha * (self.p2 - self.p1)
