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
        self.max_speed = speed
        self.velocity = velocity
        self.vision = vision

        # Default relaxation parameter
        self.tau = 1

    @staticmethod
    def _gradient(func, val):
        """Compute the gradient of func at val using finite differencing"""
        # Settings for gradient descent
        delta = 1e-5
        dx = delta * np.array([1, 0])
        dy = delta * np.array([0, 1])

        # Compute the gradient per variable
        grad_x = (func(val + dx) - func(val - dx)) / (2 * delta)
        grad_y = (func(val + dy) - func(val - dy)) / (2 * delta)

        # Return the complete gradient
        return np.array([grad_x, grad_y])

    def desired_dir(self):
        """Compute the desired direction of the agent"""
        dir = self.dest - self.pos
        return dir / np.linalg.norm(dir)

    def dest_effect(self):
        """Attractive effect to the goal"""
        return (self.max_speed * self.desired_dir() - self.velocity) / self.tau

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
        def f(x): return v(b(x))
        return -1.0 * Human._gradient(f, r)

    def obstacle_effect(self, obstacle):
        """Repulsive effect from an obstacle"""
        def u(r):
            """Potential function"""
            u0 = 10
            R = 0.2
            return u0 * np.exp(-np.linalg.norm(r)/R)

        # We do gradient descent, so return the negative gradient
        def f(x):
            obstacle_point = obstacle.get_closest_point(x)
            r = self.pos - obstacle_point
            return u(r)

        return -1.0 * self._gradient(f, self.pos)

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
        for obstacle in self.model.obstacles:
            self.velocity += self.obstacle_effect(obstacle)

        # Compute attractive effect to points/people of interest
        # TODO. Currently not implemented

        # Update the position
        self.speed = np.clip(np.linalg.norm(self.velocity), 0, self.max_speed)
        self.velocity /= np.linalg.norm(self.velocity)
        self.velocity *= self.speed
        new_pos = self.pos + self.velocity

        # if out of bounds, put at bound
        if new_pos[0] > self.model.space.width:
            new_pos[0] = self.model.space.width
        elif new_pos[0] < 0:
            new_pos[0] = 0

        if new_pos[1] > self.model.space.height:
            new_pos[1] = self.model.space.height
        elif new_pos[1] < 0:
            new_pos[1] = 0
        self.model.space.move_agent(self, new_pos)

        # Remove once the desitination is reached
        if self.pos[0] == 0 and self.pos[1] == 0:
            self.model.remove_agent(self)
