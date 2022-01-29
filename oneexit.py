from model2002 import SocialForce
from exit import Exit
from wall import Wall
import numpy as np


class OneExit(SocialForce):
    '''Social Force but then for different doorsizes.'''
    def __init__(
        self,
        population: int = 200,
        width: float = 100,
        height: float = 100,
        max_speed: float = 5,
        vision: float = 1,
        relaxation_time: float = 1,
        timestep: float = 0.01,
        init_desired_speed: float = 2.0,
        door_size: float = 1,
    ):
        # Pass along
        super().__init__(population, width, height, max_speed, vision, relaxation_time, [], [], timestep, init_desired_speed)

        # Define walls
        self.obstacles = [
            Wall(np.array([0, 0]), np.array([0, height])),
            Wall(np.array([0, 0]), np.array([width, 0])),
            Wall(np.array([width, 0]), np.array([width, height])),
            Wall(np.array([0, height]), np.array([width / 2 - door_size / 2, height])),
            Wall(np.array([width / 2 + door_size / 2, height]), np.array([width, height]))
        ]

        # Define exit
        self.exits = [Exit(np.array([width / 2 - door_size / 2, height]), np.array([width / 2 + door_size / 2, height]))]
        