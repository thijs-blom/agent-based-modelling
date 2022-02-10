import numpy as np

from .exit import Exit
from .social_force import SocialForce
from .wall import Wall


class OneExit(SocialForce):
    """Social force model with predefined walls and an exit"""

    def __init__(self, width: float = 15, height: float = 15, door_size: float = 1, **kwargs):
        """
        Args:
            width (float): Width of the simulated space
            height (float): Height of the simulated space
            door_size (float): Width of the exit
            **kwargs: Other keyword arguments passed along to the model
        """
        # Define walls
        obstacles = [
            Wall(np.array([0, 0]), np.array([0, height])),
            Wall(np.array([0, 0]), np.array([width, 0])),
            Wall(np.array([width, 0]), np.array([width, height])),
            Wall(np.array([0, height]), np.array([width / 2 - door_size / 2, height])),
            Wall(np.array([width / 2 + door_size / 2, height]), np.array([width, height]))
        ]

        # Define the exit
        exits = [
            Exit(np.array([width / 2 - door_size / 2, height]), np.array([width / 2 + door_size / 2, height]))]

        # Add any additional obstacles passed
        if "obstacles" in kwargs:
            obstacles += kwargs["obstacles"]
            del kwargs["obstacles"]

        # Do not allow additional exits
        if "exits" in kwargs:
            del kwargs["exits"]

        # Instantiate SocialForce with the specified walls and exit, passing along any other keyword arguments
        super().__init__(width=width, height=height, obstacles=obstacles, exits=exits, **kwargs)
