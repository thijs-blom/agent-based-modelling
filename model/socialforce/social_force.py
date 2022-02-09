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
    """Social Force model. Handles agent creation, placement, exiting and scheduling."""

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
            prob_stressed: float = 1.0,
            strategies: list = None,
            strategy_weights: list = None,
            soc_strength: float = 2000,
            soc_range: float = 0.08,
            bfc: float = 120000,
            sfc: float = 240000,
            obs_strength: float = 5000,
            obs_range: float = 0.08,
            datacollector: DataCollector = None
    ):
        """Initialize an instance of the social force model

        Args:
            population: The initial number of agents in the system
            width: The width of the simulated space (m)
            height: The height of the simulated space (m)
            max_speed: The maximum speed an agent can move at (m/s)
            vision: The maximum distance an agent can see neighbours from (m)
            relaxation_time: The time it takes an agent to adjust to their desired direction (s)
            obstacles: A list of obstacles in the simulated space that agents must avoid
            exits: A list of exits in the simulated space that allow agents to leave the system
            timestep: The amount of time simulated by a single step of the simulation (s)
            init_desired_speed: The speed an agent initially wants to move at (m/s)
            prob_stressed: TODO set this
            strategies: The strategies an agent may be assigned on initialization
            strategy_weights: The probabilities corresponding to the passed strategies.
                              May only be set if strategies is set.
            soc_strength: A model parameter specifying the strength of the repulsive force due to other agents
            soc_range: A model parameter specifying a reference range for the social repulsive force
            bfc: A model parameter specifying the body force coefficient
            sfc: A model parameter specifying the sliding force coefficient
            obs_strength: A model parameter specifying the strength of the repulsive force due to obstacles
            obs_range: A model parameter specifying a reference range for the obstacle repulsive force
            datacollector: Object to keep track of statistics during model execution. Will be run after initial setup
                           and after each tick of the model.
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
        if datacollector:
            self.datacollector = datacollector

        # Set default strategies if none are given
        if strategies is None:
            strategies = ['nearest exit']

            # Do not allow strategy weights to be set without setting the strategies
            if strategy_weights is not None:
                raise ValueError("Strategy weights passed without corresponding strategies")

        # Set up the model
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width, height, False)
        self.make_agents(strategies, strategy_weights, prob_stressed)
        self.running = True

        # Collect statistics after initial setup
        if hasattr(self, 'datacollector'):
            self.datacollector.collect(self)

    def make_agents(self, strategies, strategy_weights, prob_stressed):
        """Initialize the number of agents specified by self.population, with random positions and initial headings

        Args:
            strategies: A list of strategies an agent may be initialized with
            strategy_weights: A list with probabilities corresponding to the passed strategies
            prob_stressed:
        """
        for i in range(self.population):
            # Determine the agent's position
            x = self.random.random() * self.space.x_max
            y = self.random.random() * self.space.y_max
            pos = np.array((x, y))

            # Determine the agent's properties
            lam = np.random.uniform(0.7, 0.95)
            mass = np.random.uniform(50, 80)
            radius = np.random.uniform(0.37, 0.55) / 2
            relax_t = self.relaxation_time
            strategy = np.random.choice(strategies, strategy_weights)
            current_timestep = 0

            # Determine the agent's initial speed
            velocity = np.random.random(2) - 0.5
            init_speed = np.linalg.norm(velocity)
            init_desired_speed = self.init_desired_speed  # np.random.normal(2, 0.15) if np.random.random() < prob_stressed else np.random.normal(1, 0.15)

            # Initialize the agent
            human = Human(
                unique_id=i,
                model=self,
                pos=pos,
                velocity=velocity,
                max_speed=self.max_speed,
                vision=self.vision,
                mass=mass,
                radius=radius,
                lam=lam,
                current_timestep=current_timestep,
                init_speed=init_speed,
                init_desired_speed=init_desired_speed,
                relax_t=relax_t,
                strategy=strategy
            )

            # Add the agent to the system
            self.space.place_agent(human, pos)
            self.schedule.add(human)

    def step(self):
        """Progress the simulation one tick, allowing the agents to move and act"""
        # Let every agent perform action
        self.schedule.step()

        # Save the statistics
        if hasattr(self, 'datacollector'):
            self.datacollector.collect(self)

        # For experiment with different initial speeds
        # self.panic_level.append(self.count_panic())

        # Stop the simulation if there are no agents remaining in the system
        if self.schedule.get_agent_count() == 0:
            self.running = False

    def remove_agent(self, agent):
        """Method that removes an agent from the grid and the correct scheduler."""
        self.space.remove_agent(agent)
        self.schedule.remove(agent)

    def average_speed(self) -> float:
        """Returns the current average actual speed of all agents in the system"""
        speed = 0
        for human in self.schedule.agents:
            speed += np.linalg.norm(human.velocity)

        # Return the average speed, or 0 if there are no agents in the system
        n = self.schedule.get_agent_count()
        return speed / n if n > 0 else 0

    def average_panic(self) -> float:
        """Returns the average panic index of all agents in the system"""
        panic = 0
        for human in self.schedule.agents:
            panic += human.panic_index()

        # Return the average panic, or 0 if there are no agents in the system
        n = self.schedule.get_agent_count()
        return panic / n if n > 0 else 0

    # TODO: check for difference between desired_spee and init_desired_speed
    def average_desired_speed(self) -> float:
        """Returns the average desired speed of all agents in the system"""
        desired_speed = 0
        for human in self.schedule.agents:
            desired_speed += human.init_desired_speed

        # Return the average desired speed, or 0 if there are no agents in the system
        n = self.schedule.get_agent_count()
        return desired_speed / n if n > 0 else 0

    def evacuation_percentage(self) -> float:
        """Returns the percentage of the population that has evacuated the space"""
        return (self.population - self.schedule.get_agent_count()) / self.population * 100

    # TODO: change this to something else if evacuation is not complete?
    def evacuation_time(self) -> float:
        """Returns the real-life time it took to evacuate the space.

        Note: the time reported is the _last_ time an agent left the system. The evacuation may
        not be complete yet. Check SocialForce.evacuation_percentage() to verify all agents have left the system.

        If no agents have left the system yet, the reported time is 0.
        """
        if len(self.exit_times) == 0:
            return 0

        return self.exit_times[-1]

    def flow(self) -> float:
        """Returns the average pedestrian (out-)flow of the system"""
        if len(self.exit_times) < 2:
            return 0

        return (len(self.exit_times) - 1) / (self.exit_times[-1] - self.exit_times[0])
