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
from human import Human
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
            vision: float = 1,
            relaxation_time: float = 1,
            obstacles: List[Obstacle] = None,
            exits: List[Obstacle] = None,
            timestep: float = 0.01,
            init_desired_speed: float = 2.0,
            prob_nearest: float = 0.5,
            lst_strategy: list = None,
            soc_strength: float = 2000,
            soc_range: float = 0.08,
            bfc: float = 120000,
            sfc: float = 240000,
            obs_strength: float = 5000,
            obs_range: float = 0.08
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
        # Set model parameters
        self.population = population
        self.vision = vision
        self.relaxation_time = relaxation_time
        self.obstacles = obstacles if obstacles else []
        self.exits = exits if exits else []
        self.max_speed = max_speed
        self.init_desired_speed = init_desired_speed
        self.timestep = timestep

        # Model parameters for the forces
        self.soc_strength = soc_strength
        self.soc_range = soc_range
        self.bfc = bfc
        self.sfc = sfc
        self.obs_strength = obs_strength
        self.obs_range = obs_range

        # Variables to keep track of relevant statistics
        self.exit_times = []
        self.evacuation_time = np.inf
        self.flow = 0

        # Set default strategies if none are given
        if lst_strategy is None:
            lst_strategy = ['nearest exit', 'hesitator']

        # Set up the model
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width, height, False)
        self.make_agents(lst_strategy, prob_nearest)

        # self.datacollector = DataCollector(
        #     model_reporters={
        #         "Number of Humans in Environment": lambda m: self.schedule.get_agent_count(),
        #         "Average Panic": lambda m: self.count_panic() / self.schedule.get_agent_count() if self.schedule.get_agent_count() > 0 else 0,
        #         "Average Speed": lambda m: self.count_speed() / self.schedule.get_agent_count() if self.schedule.get_agent_count() > 0 else 0
        #     })

        self.running = True
        # self.datacollector.collect(self)

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

    @staticmethod
    def random_select_strategy(strategy_option, prob_nearest):
        """Randomly select the population strategy based on some probabilities"""
        rand_num = np.random.random()
        if rand_num <= prob_nearest:
            return strategy_option[0]
        else:
            return strategy_option[1]

    def make_agents(self, strategy_option, prob_nearest):
        """
        Create self.population agents, with random positions and starting headings.
        """
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
            init_desired_speed = self.init_desired_speed
            relax_t = self.relaxation_time
            strategy = self.random_select_strategy(strategy_option, prob_nearest)
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
                relax_t,
                'nearest exit'
            )
            self.space.place_agent(human, pos)
            self.schedule.add(human)

    def step(self):
        """Let the agent move/act."""
        self.schedule.step()

        # Save the statistics
        # self.datacollector.collect(self)

        if self.schedule.get_agent_count() == 0:
            self.flow = (len(self.exit_times) - 1) / (self.exit_times[-1] - self.exit_times[0])
            print(self.flow)
            self.evacuation_time = self.exit_times[-1]
            self.running = False

    def remove_agent(self, agent):
        """
        Method that removes an agent from the grid and the correct scheduler.
        """
        self.space.remove_agent(agent)
        self.schedule.remove(agent)
