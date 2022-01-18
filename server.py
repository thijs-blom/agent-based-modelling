from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
#from mesa.visualization.modules import ChartModule

from model import SocialForce
from SimpleContinuousModule import SimpleCanvas
from wall import Wall
import numpy as np


def human_draw(agent):
    return {"Shape": "circle", "r": 2, "Filled": "true", "Color": "Red"}

canvas = SimpleCanvas(human_draw, 500, 500)
#chart =  ChartModule([{"Label": "Human", "Color": "#AA0000"}])

wall = Wall(np.array([50, 0]), np.array([50, 200]))

# Draws the wall (but not yet both agents and wall
# def wall_draw(wall):
#   return {"Shape": "line", "pos1": [50, 0], "pos2": [50, 200], "w": 5, "Color": "Black"}
# canvas = SimpleCanvas(wall_draw, 500, 500)

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
