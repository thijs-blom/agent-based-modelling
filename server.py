from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
#from mesa.visualization.modules import ChartModule

from model import SocialForce
from SimpleContinuousModule import SimpleCanvas
from wall import Wall
import numpy as np


def human_draw(agent):
    return {"Shape": "circle", "r": 2, "Filled": "true", "Color": "Red"}


def wall_draw(wall):
    return {"Shape": "line", "w": 5, "Color": "Black"}

canvas = SimpleCanvas(human_draw, wall_draw, 500, 500)
#chart =  ChartModule([{"Label": "Human", "Color": "#AA0000"}])

wall = Wall(np.array([50, 0]), np.array([50, 200]))


model_params = {
    "population": UserSettableParameter(
        "slider",
        "Population",
        10,
        10,
        1000,
        description="The initial population",
    ),
    "width": 100,
    "height": 100,
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
    "obstacles": [wall],
    "dest": np.array([0, 0])
}

#server = ModularServer(SocialForce, [canvas, chart], "Boids", model_params)
server = ModularServer(SocialForce, [canvas], "Boids", model_params)

server.launch()
