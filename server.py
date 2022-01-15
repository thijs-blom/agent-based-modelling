from mesa.visualization.ModularVisualization import ModularServer

from model import SocialForce
from SimpleContinuousModule import SimpleCanvas
from wall import Wall
import numpy as np


def boid_draw(agent):
    return {"Shape": "circle", "r": 2, "Filled": "true", "Color": "Red"}


boid_canvas = SimpleCanvas(boid_draw, 500, 500)

wall = Wall(np.array([0, 30]), np.array([50, 0]))
model_params = {
    "population": 400,
    "width": 100,
    "height": 100,
    "speed": 5,
    "vision": 10,
    "obstacles": [wall],
    "dest": np.array([0, 0])
}

server = ModularServer(SocialForce, [boid_canvas], "Boids", model_params)

server.launch()
