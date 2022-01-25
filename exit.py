import numpy as np
from dataclasses import dataclass

@dataclass
class Exit:
    left: np.array
    right: np.array

    def get_size(self):
        return self.right - self.left

    def get_center(self):
        return (self.left + self.right) / 2

    def in_exit(self, pos):
        to_left = np.linalg.norm(pos - self.left)
        to_right = np.linalg.norm(pos - self.right)
        width = np.linalg.norm(self.left - self.right)
        return to_left + to_right - width < 1e-3

    def get_random_focus(self, agent: np.array) -> np.array:
        """Compute the random focus point of the agent to the given exit"""

        p1 = self.left
        p2 = self.right

        line_vec = p2 - p1
        # Compute a normal of p2 - p1
        v = np.copy(np.flip(line_vec))
        v[0] *= -1.0

        # Solve the system defined by p1 + a*(p2-p1) = other_point + b*v)
        A = np.stack((line_vec, -v), axis=1)
        res = np.linalg.solve(A, agent.pos - p1)

        # Check whether the project of other_point on the (infinite) line
        # through p1 and p2 is between the endpoints. If so, return the
        # projection. Otherwise, return the relevant endpoint as the
        # closest point.
        exit_size = self.get_size()
        rand = np.random.uniform(0.2,0.5)


        alpha = res[0]
        if alpha > 1:
            return p2 - rand * exit_size 
        elif alpha < 0:
            return p1 + rand * exit_size
        else:
            return p1 + alpha * (p2 - p1) 

        # if alpha >= 1:
        #     return p2 
        # elif alpha <= 0:
        #     return p1 
        # else:
        #     return p1 + alpha * (p2 - p1) 
            