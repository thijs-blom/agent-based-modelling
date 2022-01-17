from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from model import SocialForce
from SimpleContinuousModule import SimpleCanvas
from wall import Wall
import numpy as np


def boid_draw(agent):
    return {"Shape": "circle", "r": 2, "Filled": "true", "Color": "Red"}


boid_canvas = SimpleCanvas(boid_draw, 500, 500)

wall = Wall(np.array([0, 30]), np.array([50, 0]))

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

server = ModularServer(SocialForce, [boid_canvas], "Boids", model_params)

server.launch()
