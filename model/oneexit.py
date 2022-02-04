import numpy as np

from model.exit import Exit
from model.model import SocialForce
from model.wall import Wall


class OneExit(SocialForce):
    """Social Force but then for different door sizes."""
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

        # Define exit
        exits = [
            Exit(np.array([width / 2 - door_size / 2, height]), np.array([width / 2 + door_size / 2, height]))]

        if "obstacles" in kwargs:
            obstacles += kwargs["obstacles"]
            del kwargs["obstacles"]

        if "exits" in kwargs:
            # TODO: do we want to add them to the model? Since it's called _One_ExitModel
            del kwargs["exits"]


# Instantiate SocialForce with the specified walls and exit, passing along any other keyword arguments
        super().__init__(obstacles=obstacles, exits=exits, **kwargs)
