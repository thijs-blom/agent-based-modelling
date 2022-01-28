from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.ModularVisualization import VisualizationElement

from SimpleContinuousModule import SimpleCanvas
from exit import Exit
from oneexit import OneExit
from wall import Wall
from dead import Dead
from human2002 import Human
from model2002 import SocialForce

import numpy as np
from typing import Dict


class HistogramModule(VisualizationElement):
    package_includes = ["Chart.min.js"]
    local_includes = ["histogram.js"]

    def __init__(self, bins, canvas_height, canvas_width):
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.bins = bins
        new_element = "new HistogramModule({}, {}, {})"
        new_element = new_element.format(bins,
                                         canvas_width,
                                         canvas_height)
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        vals = [agent.speed for agent in model.schedule.agents]
        hist = np.histogram(vals, bins=self.bins)[0]
        return [int(x) for x in hist]


# Define the dimensions of the simulation space
width = 20
height = 20

def human_draw(agent: Human) -> Dict:
    return {"Shape": "circle", "r": 1, "Filled": "true", "Color": "Red"}

def wall_draw(wall: Wall) -> Dict:
    return {"Shape": "line", "w": 5, "Color": "Black"}

def dead_draw(dead: Dead) -> Dict:
    return {"Shape": "circle", "r": 0.25*25, "Filled": "true", "Color": "Black"}


# Define canvas and charts
canvas = SimpleCanvas(human_draw, wall_draw, dead_draw, canvas_width=500, canvas_height=500)
chart0 = ChartModule([{"Label": "Number of Humans in Environment", "Color": "#AA0000"}], 10, 25)
chart1 = ChartModule([{"Label": "Number of Casualties", "Color": "#AA0000"}], 10, 25)
chart2 = ChartModule([{"Label": "Average Panic", "Color": "#AA0000"}], 10, 25)
chart3 = ChartModule([{"Label": "Average Speed", "Color": "#AA0000"}], 10, 25)

# hist1 = HistogramModule(np.arange(0, 5, 0.1), height, width)

# Define all walls in the system
side_wall1 = Wall(np.array([0, 0]), np.array([0, height]))
side_wall2 = Wall(np.array([0, 0]), np.array([width, 0]))
side_wall3 = Wall(np.array([width, 0]), np.array([width, height]))
side_wall4 = Wall(np.array([0, height]), np.array([width/2, height]))
side_wall5 = Wall(np.array([width/2 + 2, height]), np.array([width, height]))

init_obstacles = [side_wall1, side_wall2, side_wall3, side_wall4, side_wall5]

# Define all exits in the system
#exit1 = Exit(np.array([0, 0]), np.array([1, 0]))
exit2 = Exit(np.array([width/2, height]), np.array([width/2 + 2, height]))

# Set up all the parameters to be entered into the model
model_params = {
    "population": UserSettableParameter(
        "slider",
        "Population",
        100,
        1,
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
    "relaxation_time": UserSettableParameter(
        "slider",
        "Relaxation Time (1/tau)",
        1,
        0.5,
        1,
        0.01,
        description="Relaxation Time"),
    "obstacles": init_obstacles,
    "exits": [exit2]
}

# Define and launch the server
server = ModularServer(OneExit, [canvas, chart0, chart2, chart3], "Escape Panic", model_params)

# server.launch()