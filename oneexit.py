from typing import  List

from model import SocialForce
from exit import Exit
from sample import Sample
from wall import Wall
import numpy as np


class OneExit(SocialForce):
    """Social Force but then for different door sizes."""

    def __init__(
            self,
            population: int = 100,
            width: float = 15,
            height: float = 15,
            max_speed: float = 5,
            vision: float = 2,
            relaxation_time: float = 0.5,
            timestep: float = 0.005,
            init_desired_speed: float = 2.0,
            prob_nearest: float = 0.6,
            door_size: float = 1,
            soc_strength: float = 2000,
            soc_range: float = 0.08,
            bfc: float = 120000,
            sfc: float = 240000,
            obs_strength: float = 5000,
            obs_range: float = 0.08,
            sample: Sample = None
    ):
        # Check if any argument is missing
        if (max_speed is None or vision is None or soc_strength is None or obs_strength is None) \
                and sample is None:
            raise ValueError("Incomplete argument list. max_speed, vision, soc_strength, or obs_strength is missing")

        # Check if any of the arguments in the sample is also passed separately
        if sample is not None and \
                (max_speed, vision, soc_strength, obs_strength) != (None, None, None, None):
            raise ValueError("Either max_speed, vision, soc_strength or obs_strength is passed " +
                             "both as a keyword argument and as a sample")

        # If variable parameters are used, unpack them
        if sample is not None:
            max_speed = sample.max_speed
            vision = sample.vision
            soc_strength = sample.soc_strength
            obs_strength = sample.obs_strength

        # Pass along
        super().__init__(population=population,
                         width=width,
                         height=height,
                         max_speed=max_speed,
                         vision=vision,
                         relaxation_time=relaxation_time,
                         obstacles=[],
                         exits=[],
                         timestep=timestep,
                         init_desired_speed=init_desired_speed,
                         prob_nearest=prob_nearest,
                         lst_strategy=None,
                         soc_strength=soc_strength,
                         soc_range=soc_range,
                         bfc=bfc,
                         sfc=sfc,
                         obs_strength=obs_strength,
                         obs_range=obs_range)

        # Define walls
        self.obstacles = [
            Wall(np.array([0, 0]), np.array([0, height])),
            Wall(np.array([0, 0]), np.array([width, 0])),
            Wall(np.array([width, 0]), np.array([width, height])),
            Wall(np.array([0, height]), np.array([width / 2 - door_size / 2, height])),
            Wall(np.array([width / 2 + door_size / 2, height]), np.array([width, height]))
        ]

        # Define exit
        self.exits = [
            Exit(np.array([width / 2 - door_size / 2, height]), np.array([width / 2 + door_size / 2, height]))]
