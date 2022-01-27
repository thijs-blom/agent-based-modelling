"""
Social Force Model
=============================================================
A Mesa implementation of the social force model.
Uses numpy arrays to represent vectors.
"""
# Python imports
from typing import List
import numpy as np
import random

# Mesa imports
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

# Project imports
from human2002 import Human
from obstacle import Obstacle


class SocialForce(Model):
    """
    Social Force model. Handles agent creation, placement, exiting and scheduling.
    """
    def __init__(
        self,
        population: int = 100,
        width: float = 100,
        height: float = 100,
        max_speed: float = 5,
        vision: float = 10,
        relaxation_time: float = 1,
        obstacles: List[Obstacle] = None,
        exits: List[Obstacle] = None,
        timestep: float = 0.01
    ):
        """
        Create a new instance of the social force model.

        Args:
            population: Number of Boids
            width, height: Size of the space.
            max_speed: TODO
            vision: How far around should each agent look for its neighbours
            obstacles: A list of obstacles agents must avoid
            exits: A list of exits where users leave the room
        """
        self.population = population
        self.vision = vision
        self.relaxation_time = relaxation_time
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width, height, False)
        self.obstacles = obstacles if obstacles else []
        self.exits = exits if exits else []
        self.max_speed = max_speed
        self.make_agents()
        self.init_amount_obstacles = len(self.obstacles)
        self.ending_energy_lst = np.ones(self.population)
        self.timestep = timestep
        self.exit_times = []

        # self.datacollector = DataCollector({"Human": lambda m: self.schedule.get_agent_count()})

        self.datacollector = DataCollector(
            model_reporters={
                "Number of Humans in Environment": lambda m: self.schedule.get_agent_count(),
                "Number of Casualties": lambda m: len(self.obstacles) - self.init_amount_obstacles,
                "Average Panic": lambda m: self.count_panic() / self.schedule.get_agent_count() if self.schedule.get_agent_count() > 0 else 0,
                "Average Speed": lambda m: self.count_speed() / self.schedule.get_agent_count() if self.schedule.get_agent_count() > 0 else 0
            })
        
          # 'Amount of death': self.caused_death(),
        self.running = True
        self.datacollector.collect(self)
        
          # 'Amount of death': self.caused_death(),
        self.running = True
        self.datacollector.collect(self)

    def count_energy(self):
        """
        Helper method to count trees in a given condition in a given model.
        """
        count = 0
        for human in self.schedule.agents:
            if human.energy >= 0:
                count += human.energy
        return count

    def count_speed(self):
        """
        Helper method to count trees in a given condition in a given model.
        """
        speed = 0
        for human in self.schedule.agents:
            speed += np.linalg.norm(human.velocity)
        return speed
    
    def count_panic(self):
        """
        Helper method to count trees in a given condition in a given model.
        """
        panics = 0
        for human in self.schedule.agents:
            panics += human.panic_index()
        return panics
    
    def make_agents(self):
        """
        Create self.population agents, with random positions and starting headings.
        """
        strategy_option = ['nearest exit', 'follow the crowd', 'least crowded exit']
        for i in range(self.population):
            x = self.random.random() * self.space.x_max
            y = self.random.random() * self.space.y_max
            pos = np.array((x, y))
            lam = np.random.uniform(0.7,0.95)
            velocity = (np.random.random(2)-0.5) 
            # don't know what is mass yet
            mass = np.random.uniform(50, 80)
            radius = np.random.uniform(0.37,0.55)/2
            current_timestep = 0
            init_speed = np.random.random()
            init_desired_speed = 2
            relax_t = self.relaxation_time
            strategy = np.random.choice(strategy_option)
            human = Human(
                i,
                self,
                pos,
                velocity,
                self.max_speed,
                self.vision,
                mass,
                radius,
                lam,
                current_timestep,
                init_speed,
                init_desired_speed,
                False,
                relax_t,
                'nearest exit'
            )
            self.space.place_agent(human, pos)
            self.schedule.add(human)

    def step(self):
        """Let the agent move/act."""
        self.schedule.step()

        # Save the statistics
        self.datacollector.collect(self)

        if self.schedule.get_agent_count() == 0:
            self.running = False
            print(sum(self.ending_energy_lst)/self.population)

    def remove_agent(self, agent):
        """
        Method that removes an agent from the grid and the correct scheduler.
        """
        self.ending_energy_lst[agent.unique_id] = agent.energy
        self.space.remove_agent(agent)
        self.schedule.remove(agent)
