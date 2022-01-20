from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule

from model import SocialForce
from SimpleContinuousModule import SimpleCanvas
from wall import Wall
from exit import Exit
import numpy as np


def human_draw(agent):
    return {"Shape": "circle", "r": 1, "Filled": "true", "Color": "Red"}


def wall_draw(wall):
    return {"Shape": "line", "w": 5, "Color": "Black"}

canvas = SimpleCanvas(human_draw, wall_draw, 500, 500)
chart =  ChartModule([{"Label": "Human", "Color": "#AA0000"}], 10, 25)

width = 10
height = 10

#wall1 = Wall(np.array([50, 0]), np.array([50, 200]))
side_wall1 = Wall(np.array([0, 2]), np.array([0, height]))
side_wall2 = Wall(np.array([0, 0]), np.array([width, 0]))
side_wall3 = Wall(np.array([width, 0]), np.array([width, height]))
#side_wall4 = Wall(np.array([0, height]), np.array([width, height]))
side_wall4 = Wall(np.array([0, height]), np.array([48, height]))
side_wall5 = Wall(np.array([50, height]), np.array([width, height]))

exit1 = Exit(np.array([0,0]), np.array([0,2]))
exit2 = Exit(np.array([48,100]), np.array([50,100]))


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
    "obstacles": [side_wall1, side_wall2, side_wall3, side_wall4, side_wall5],
    "exits": [exit1, exit2]
}

server = ModularServer(SocialForce, [canvas, chart], "Escape Panic", model_params)

server.launch()
