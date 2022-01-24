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
        max_speed=5,
        vision=10,
        obstacles=[],
        exits=[],
        init_amount_obstacles = 5,
    ):
        """
        Create a new Flockers model.

        Args:
            population: Number of Boids
            width, height: Size of the space.
            speed: How fast should the Boids move.
            vision: How far around should each Boid look for its neighbors
            obstacles: A list of obstacles agents must avoid
            exits: A list of exits where users leave the room
        """
        self.population = population
        self.vision = vision
        self.speed = 1
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width, height, False)
        self.obstacles = obstacles
        self.exits = exits
        self.max_speed = max_speed
        self.make_agents()
        self.init_amount_obstacles = init_amount_obstacles

        # self.datacollector = DataCollector({"Human": lambda m: self.schedule.get_agent_count()})

        # NOT WORKING YET, NEED TO COUNT THE INCREASE IN OBSTACLES
        # "Caused Deaths": lambda m: len(self.obstacles) - self.init_amount_obstacles,

        self.datacollector = DataCollector(
            model_reporters={
            "Number of Humans in Environment": lambda m: self.schedule.get_agent_count(),
            "Number of Casualties": lambda m: len(self.obstacles) - self.init_amount_obstacles,
            "Average Energy": lambda m: self.count_energy(m) / self.population,
            "Average Speed" : lambda m: self.count_speed(m) / self.schedule.get_agent_count() if self.schedule.get_agent_count() > 0 else 0
            })
        
          # 'Amount of death': self.caused_death(),
        self.running = True
        self.datacollector.collect(self)
    
    
    @staticmethod
    def count_energy(model):
        """
        Helper method to count trees in a given condition in a given model.
        """
        count = 0
        for human in model.schedule.agents:
            if human.energy >= 0:
                count += human.energy
        return count

    @staticmethod
    def count_speed(model):
        """
        Helper method to count trees in a given condition in a given model.
        """
        count = 0
        for human in model.schedule.agents:
            speed = np.linalg.norm(human.velocity)
            if speed >= 0:
                count += speed
        return count

    def make_agents(self):
        """
        Create self.population agents, with random positions and starting headings.
        """
        radii_option = [0.2,0.25,0.3]
        lam_option =[0.7,0.8,0.9]
        strategy_option = ['nearest exit', 'follow the crowd', 'least crowded exit']
        for i in range(self.population):
            x = self.random.random() * self.space.x_max
            y = self.random.random() * self.space.y_max
            pos = np.array((x, y))
            lam = np.random.choice(lam_option)
            velocity = np.random.random(2) * 2 - 1
            # dont know what is mass yet
            mass = 80
            radii = random.choice(radii_option)
            current_timestep = 0
            init_speed = np.random.random()
            strategy = np.random.choice(strategy_option)
            human = Human(
                i,
                self,
                pos,
                velocity,
                self.max_speed,
                self.vision,
                mass,
                radii,
                lam,
                current_timestep,
                init_speed,
                init_speed,
                False,
<<<<<<< HEAD
                'nearest exit'
=======
                'follow the crowd',
>>>>>>> rina
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
