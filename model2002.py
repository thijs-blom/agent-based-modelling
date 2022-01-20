"""
Social Force Model
=============================================================
A Mesa implementation of the social force model.
Uses numpy arrays to represent vectors.
"""

import numpy as np
import random
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from human2002 import Human


class SocialForce(Model):
    """
    Social Force model. Handles agent creation, placement, exiting and scheduling.
    """

    def __init__(
        self,
        population=100,
        width=100,
        height=100,
        max_speed=2,
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
        self.speed = 1
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width, height, False)
        self.obstacles = obstacles
        self.dest = dest
        self.max_speed = max_speed
        self.make_agents()

        self.datacollector = DataCollector({"Human": lambda m: self.schedule.get_agent_count()})
        self.running = True
        self.datacollector.collect(self)

    def make_agents(self):
        """
        Create self.population agents, with random positions and starting headings.
        """
        radii_option = [0.6,0.8,1.0,1.2]
        lam_option =[0.7,0.8,0.9]
        for i in range(self.population):
            x = self.random.random() * self.space.x_max
            y = self.random.random() * self.space.y_max
            pos = np.array((x, y))
            lam = np.random.choice(lam_option)
            velocity = np.random.random(2) * 2 - 1
            # dont know what is mass yet
            mass = 50
            radii = random.choice(radii_option)
            current_timestep = 0
            init_speed = 1
            human = Human(
                i,
                self,
                pos,
                self.dest,
                velocity,
                self.max_speed,
                self.vision,
                mass,
                radii,
                lam,
                current_timestep,
                init_speed,
                init_speed,
                False
            )
            self.space.place_agent(human, pos)
            self.schedule.add(human)

    def step(self):
        '''Let the agent move/act.'''
        self.schedule.step()

        # Save the statistics
        self.datacollector.collect(self)

        if self.schedule.get_agent_count() == 0:
            self.running = False

    def remove_agent(self, agent):
        '''
        Method that removes an agent from the grid and the correct scheduler.
        '''
        self.space.remove_agent(agent)
        self.schedule.remove(agent)
