import abc

import numpy as np


class Obstacle:
    """Base class for an obstacle that agents must avoid"""

    @abc.abstractmethod
    def get_closest_point(self, agent_pos: np.ndarray) -> np.ndarray:
        """Compute the closest point of the obstacle to the given point

        Args:
            agent_pos (np.ndarray): The position of the agent

        Returns:
            The point of the obstacle closest to the agent:
        """
        raise NotImplementedError()
