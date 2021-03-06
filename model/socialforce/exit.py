import numpy as np
from dataclasses import dataclass


@dataclass
class Exit:
    left: np.ndarray
    right: np.ndarray

    def get_size(self) -> float:
        """Return the width of the door"""
        return np.linalg.norm(self.right - self.left)

    def get_center(self) -> np.ndarray:
        """Return the coordinate of the center of the door"""
        return (self.left + self.right) / 2

    def in_exit(self, pos: np.ndarray, radius: float) -> bool:
        """Check whether an agent is in the exit

        Args:
            pos (np.ndarray): The position of the agent
            radius (float): The radius of the agent

        Returns:
            True if the agent is in the exit, false otherwise
        """
        to_left = np.linalg.norm(pos - self.left)
        to_right = np.linalg.norm(pos - self.right)

        # Make sure they are completely in the exit
        if to_left < radius or to_right < radius:
            return False

        width = np.linalg.norm(self.left - self.right)
        return to_left + to_right - width < 0.05

    def get_random_focus(self) -> np.ndarray:
        """Compute the random focus point of the agent to the given exit"""
        return self.left + np.random.random() * (self.right - self.left)
