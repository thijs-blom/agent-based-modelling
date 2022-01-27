import numpy as np
from dataclasses import dataclass


@dataclass
class Exit:
    left: np.ndarray
    right: np.ndarray

    def get_size(self) -> float:
        return np.linalg.norm(self.right - self.left)

    def get_center(self) -> np.ndarray:
        return (self.left + self.right) / 2

    def in_exit(self, pos: np.ndarray) -> bool:
        to_left = np.linalg.norm(pos - self.left)
        to_right = np.linalg.norm(pos - self.right)
        width = np.linalg.norm(self.left - self.right)
        return to_left + to_right - width < 0.05

    def get_random_focus(self, agent: np.ndarray) -> np.ndarray:
        """Compute the random focus point of the agent to the given exit"""
        return self.left + np.random.random() * (self.right - self.left)
