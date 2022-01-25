from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule

from SimpleContinuousModule import SimpleCanvas
from exit import Exit
from wall import Wall
from dead import Dead
from human2002 import Human
from model2002 import SocialForce

import numpy as np
from typing import Dict


def human_draw(agent: Human) -> Dict:
    return {"Shape": "circle", "r": 1, "Filled": "true", "Color": "Red"}


def wall_draw(wall: Wall) -> Dict:
    return {"Shape": "line", "w": 5, "Color": "Black"}


def dead_draw(dead: Dead) -> Dict:
    return {"Shape": "circle", "r": 1, "Filled": "true", "Color": "Black"}


# Define canvas and charts
canvas = SimpleCanvas(human_draw, wall_draw, dead_draw, canvas_width=500, canvas_height=500)
chart0 = ChartModule([{"Label": "Number of Humans in Environment", "Color": "#AA0000"}], 10, 25)
chart1 = ChartModule([{"Label": "Number of Casualties", "Color": "#AA0000"}], 10, 25)
chart2 = ChartModule([{"Label": "Average Energy", "Color": "#AA0000"}], 10, 25)
chart3 = ChartModule([{"Label": "Average Speed", "Color": "#AA0000"}], 10, 25)

# Define the dimensions of the simulation space
width = 100
height = 100

# Define all walls in the system
wall1 = Wall(np.array([50, 0]), np.array([50, 50]))
side_wall1 = Wall(np.array([0, 2]), np.array([0, height]))
side_wall2 = Wall(np.array([0, 0]), np.array([width, 0]))
side_wall3 = Wall(np.array([width, 0]), np.array([width, height]))
side_wall4 = Wall(np.array([0, height]), np.array([48, height]))
side_wall5 = Wall(np.array([50, height]), np.array([width, height]))

init_obstacles = [side_wall1, side_wall2, side_wall3, side_wall4, side_wall5, wall1]

# Define all exits in the system
exit1 = Exit(np.array([0, 0]), np.array([0, 2]))
exit2 = Exit(np.array([48, 100]), np.array([50, 100]))

# Set up all the parameters to be entered into the model
model_params = {
    "population": UserSettableParameter(
        "slider",
        "Population",
        100,
        10,
        1000,
        description="The initial population",
    ),
    "width": width,
    "height": height,
    "vision": UserSettableParameter(
        "slider",
        "Vision",
        1,
        1,
        10,
        description="Vision of the agents",
    ),
    "obstacles": init_obstacles,
    "exits": [exit1, exit2]
}

# Define and launch the server
server = ModularServer(SocialForce, [canvas, chart0, chart1, chart2, chart3], "Escape Panic", model_params)
server.launch()
