"""
Social Force Model
=============================================================
A Mesa implementation of the social force model.
Uses numpy arrays to represent vectors.
"""
# Python imports
from typing import List

import numpy as np

# Mesa imports
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

# Project imports
from .exit import Exit
from .human import Human
from .obstacle import Obstacle


class SocialForce(Model):
    """
    Implementation of the Social Force model. Handles agent creation, placement, exiting and scheduling.
    """
    def __init__(
            self,
            population: int = 100,
            width: float = 15,
            height: float = 15,
            max_speed: float = 5,
            vision: float = 1,
            relaxation_time: float = 0.5,
            obstacles: List[Obstacle] = None,
            exits: List[Exit] = None,
            timestep: float = 0.01,
            init_desired_speed: float = 2.0,
            prob_stressed: float = None,
            prob_nearest: float = 1.0,
            lst_strategy: list = None,
            soc_strength: float = 2000,
            soc_range: float = 0.08,
            bfc: float = 120000,
            sfc: float = 240000,
            obs_strength: float = 5000,
            obs_range: float = 0.08,
            datacollector: DataCollector = None
    ):
        """
        Create a new instance of the social force model.

        Args:
            population: Number of agents in the space.
            width, height: Size of the space.
            max_speed: the maximum speed the agents can achieve.
            vision: How far around should each agent look for its neighbours.
            obstacles: A list of obstacles agents must avoid.
            exits: A list of exits where agents leave the room.
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
        self.panic_level = []
        self.datacollector = datacollector

        # Set default strategies if none are given
        if lst_strategy is None:
            lst_strategy = ['nearest exit', 'follow the leader']

        # Set up the model
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width, height, False)
        self.make_agents(lst_strategy, prob_nearest, prob_stressed)

        if self.datacollector:
            self.datacollector.collect(self)

        self.running = True

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
        panic = 0
        for human in self.schedule.agents:
            panic += human.panic_index()
        n = self.schedule.get_agent_count()
        if n > 0:
            panic /= n
        return panic

    def count_desired_speed(self):
        desired_speed = 0
        for human in self.schedule.agents:
            desired_speed += human.init_desired_speed
        n = self.schedule.get_agent_count()
        if n > 0:
            desired_speed /= n
        return desired_speed

    @staticmethod
    def random_select_strategy(strategy_option, prob_nearest):
        """Randomly select the population strategy based on some probabilities"""
        rand_num = np.random.random()
        if rand_num <= prob_nearest:
            return strategy_option[0]
        else:
            return strategy_option[1]

    def make_agents(self, strategy_option, prob_nearest, prob_stressed):
        """
        Create self.population agents, with random positions and starting headings.
        """
        for i in range(self.population):
            x = self.random.random() * self.space.x_max
            y = self.random.random() * self.space.y_max
            pos = np.array((x, y))
            lam = np.random.uniform(0.7,0.95)
            velocity = (np.random.random(2)-0.5)
            mass = np.random.uniform(50, 80)
            radius = np.random.uniform(0.37,0.55)/2
            current_timestep = 0
            init_speed = np.random.random()
            if prob_stressed != None:
                init_desired_speed = np.random.normal(2, 0.15) if np.random.random() < prob_stressed else np.random.normal(1, 0.15)
            else:
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
                strategy
            )
            self.space.place_agent(human, pos)
            self.schedule.add(human)

    def step(self):
        """Let the agent move/act."""
        self.schedule.step()

        # Save the statistics
        if self.datacollector:
            self.datacollector.collect(self)

        # For experiment with different initial speeds
        # self.panic_level.append(self.count_panic())

        if self.schedule.get_agent_count() == 0:
            self.running = False

    def remove_agent(self, agent):
        """
        Method that removes an agent from the grid and the correct scheduler.
        """
        self.space.remove_agent(agent)
        self.schedule.remove(agent)

    def evacuation_percentage(self):
        return (self.population - self.schedule.get_agent_count()) / self.population * 100

    def evacuation_time(self):
        if len(self.exit_times) == 0:
            return 0
        
        return self.exit_times[-1]

    def flow(self):
        if len(self.exit_times) < 2:
            return 0

        return (len(self.exit_times) - 1) / (self.exit_times[-1] - self.exit_times[0])
