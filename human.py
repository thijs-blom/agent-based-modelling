import numpy as np
from mesa import Agent


class Human(Agent):
    """
    An agent following rules from the social force model.
    Adapted from:
    - http://www.cs.uu.nl/docs/vakken/mcrws/papers_new/Helbing_Molnar%20-%201995%20-%20Social%20force%20model%20for%20pedestrian%20dynamics.pdf
    - https://github.com/projectmesa/mesa/tree/main/examples/boid_flockers
    """

    def __init__(
        self,
        unique_id,
        model,
        pos,
        dest,
        speed,
        velocity,
        vision,
    ):
        """
        Create a new Human agent

        Args:
            unique_id: Unique agent identifyer.
            model: Reference to the model object this agent is part of
            pos: Starting position
            dest: The destination the agent wants to reach
            speed: Distance to move per step.
            velocity: Velocity (unit) vector indicating direction of movement
            vision: Radius to look around for nearby agents.
        """
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.dest = np.array(dest)
        self.speed = speed
        self.velocity = velocity
        self.vision = vision

    @staticmethod
    def _gradient(func, val):
        """Compute the gradient of func at val using finite differencing"""
        # Settings for gradient descent
        delta = 1e-3
        dx = delta * np.array([1, 0])
        dy = delta * np.array([0, 1])

        # Compute the gradient per variable
        grad_x = (func(val - dx) + func(val + dx)) / (2 * delta)
        grad_y = (func(val - dy) + func(val + dy)) / (2 * delta)

        # Return the complete gradient
        return np.array([grad_x, grad_y])

    def desired_dir(self):
        """Compute the desired direction of the agent"""
        dir = self.dest - self.pos
        return dir / np.linalg.norm(dir)

    def dest_effect(self):
        """Attractive effect to the goal"""
        return (self.speed * self.desired_dir() - self.velocity) / self.tau

    def people_effect(self, other):
        """Repulsive effect from other people"""

        def b(r):
            """Semiminor axis of ellipsis as defined in paper"""
            r_norm = np.linalg.norm(r)
            step_avoidance = r - other.speed * other.desired_dir()
            return 0.5 * np.sqrt(np.power(r_norm + np.linalg.norm(step_avoidance), 2) - np.power(other.speed, 2))

        def v(b):
            """Potential function"""
            # Parameters from paper
            v0 = 2.1
            sigma = 0.3
            return v0 * np.exp(-b/sigma)

        # Compute the gradient of the potential using finite differencing
        r = self.pos - other.pos

        # We do gradient descent, so return the negative gradient
        def f(x): v(b(x))
        return -1.0 * self._gradient(f, r)

    def obstacle_effect(self, obstacle_point):
        """Repulsive effect from an obstacle"""
        def u(r_norm):
            """Potential function"""
            u0 = 10
            R = 0.2
            return u0 * np.exp(-r_norm/R)

        # We do gradient descent, so return the negative gradient
        r_norm = np.linalg.norm(self.pos - obstacle_point)
        return -1.0 * self._gradient(u, r_norm)

    def attract_effect(self, other_point):
        """Attractive effect to places/people of interest"""
        raise NotImplementedError

    def sight_weight(self, f):
        """Compute the weight to account for sight"""
        # Parameters from paper
        c = 0.5
        cosphi = np.cos(np.radians(100))

        # Compare direction of the effect with our desired direction
        if np.dot(self.desired_dir(), f) >= np.linalg.norm(f) * cosphi:
            return 1
        else:
            return c

    def step(self):
        """
        Compute all forces acting on this agent, update its velocity and move
        """
        # Compute attractive effect to destination
        self.velocity += self.dest_effect()

        # Compute repulsive effect from other people
        neighbours = self.model.space.get_neighbors(self.pos, self.vision, False)
        for other in neighbours:
            effect = self.people_effect(other)
            self.velocity += self.sight_weight(-effect) * effect

        # Compute repulsive effect from obstacles
        # TODO. This requires some knowledge of the obstacles.
        # For each obstacle, we need the closest point of that obstacle to the agent
        # Should probably be modelled in the environment/model as a function that
        # retrieves the closest point
        for obstacle in self.model.obstacles:
            obstacle_point = obstacle.get_closest_point(self.pos)
            self.velocity += self.obstacle_effect(obstacle_point)

        # Compute attractive effect to points/people of interest
        # TODO. Currently not implemented

        # Update the position
        self.velocity /= np.linalg.norm(self.velocity)
        new_pos = self.pos + self.velocity * self.speed
        self.model.space.move_agent(self, new_pos)
