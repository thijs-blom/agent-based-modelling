"""
Social Force Model
=============================================================
A Mesa implementation of the social force model.
Uses numpy arrays to represent vectors.
"""

import numpy as np

from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation

from human import Human


class SocialForce(Model):
    """
    Flocker model class. Handles agent creation, placement and scheduling.
    """

    def __init__(
        self,
        population=100,
        width=100,
        height=100,
        speed=1,
        vision=10,
        obstacles=[],
        dest=np.array([0, 0])
    ):
        """
        Create a new Flockers model.

        Args:
            population: Number of Boids
            width, height: Size of the space.
            speed: How fast should the Boids move.
            vision: How far around should each Boid look for its neighbors
            obstacles: A list of obstacles agents must avoid
        """
        self.population = population
        self.vision = vision
        self.speed = speed
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width, height, False)
        self.obstacles = obstacles
        self.dest = dest
        self.make_agents()
        self.running = True

    def make_agents(self):
        """
        Create self.population agents, with random positions and starting headings.
        """
        for i in range(self.population):
            x = self.random.random() * self.space.x_max
            y = self.random.random() * self.space.y_max
            pos = np.array((x, y))
            velocity = np.random.random(2) * 2 - 1
            human = Human(
                i,
                self,
                pos,
                self.dest,
                self.speed,
                velocity,
                self.vision
            )
            self.space.place_agent(human, pos)
            self.schedule.add(human)

    def step(self):
        '''Let the agent move/act.'''
        self.schedule.step()

    def remove_agent(self, agent):
        '''
        Method that removes an agent from the grid and the correct scheduler.
        '''
        self.space.remove_agent(agent)
        self.schedule.remove(agent)
