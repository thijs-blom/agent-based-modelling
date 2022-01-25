import numpy as np
from dataclasses import dataclass


@dataclass
class Exit:
    left: np.ndarray
    right: np.ndarray

    def get_center(self) -> np.ndarray:
        return (self.left + self.right) / 2

    def in_exit(self, pos: np.ndarray) -> bool:
        to_left = np.linalg.norm(pos - self.left)
        to_right = np.linalg.norm(pos - self.right)
        width = np.linalg.norm(self.left - self.right)
        return to_left + to_right - width < 1e-3
