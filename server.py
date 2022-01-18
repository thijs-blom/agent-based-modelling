from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule

from model import SocialForce
from SimpleContinuousModule import SimpleCanvas
from wall import Wall
import numpy as np


def human_draw(agent):
    return {"Shape": "circle", "r": 2, "Filled": "true", "Color": "Red"}


def wall_draw(wall):
    return {"Shape": "line", "w": 5, "Color": "Black"}

canvas = SimpleCanvas(human_draw, wall_draw, 500, 500)
chart =  ChartModule([{"Label": "Human", "Color": "#AA0000"}], 10, 25)

wall1 = Wall(np.array([50, 0]), np.array([50, 200]))
side_wall1 = Wall(np.array([0, 10]), np.array([0, 500]))
side_wall2 = Wall(np.array([0, 0]), np.array([500, 0]))
side_wall3 = Wall(np.array([500, 0]), np.array([500, 500]))
side_wall4 = Wall(np.array([0, 500]), np.array([500, 500]))


model_params = {
    "population": UserSettableParameter(
        "slider",
        "Population",
        100,
        10,
        1000,
        description="The initial population",
    ),
    "width": 500,
    "height": 500,
    "speed": UserSettableParameter(
        "slider",
        "Speed",
        1,
        1,
        10,
        description="Speed of Agent",
    ),
    "vision": UserSettableParameter(
        "slider",
        "Vision",
        1,
        1,
        100,
        description="Vision of the agents",
    ),
    "obstacles": [wall1, side_wall1, side_wall2, side_wall3, side_wall4],
    "dest": np.array([0, 0])
}

server = ModularServer(SocialForce, [canvas, chart], "Escape Panic", model_params)

server.launch()
