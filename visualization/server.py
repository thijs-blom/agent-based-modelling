from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.ModularVisualization import VisualizationElement

from model.obstacle import Obstacle
from visualization.SimpleContinuousModule import SimpleCanvas
from model.exit import Exit
from model.wall import Wall
from model.human import Human
from model.model import SocialForce

import numpy as np
from typing import Dict, List, Callable


# TODO: Check if we want to keep histogram, and perhaps cite it?
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


def graphical_elements(width: float, height: float) -> List[VisualizationElement]:
    canvas = SimpleCanvas(canvas_width=width, canvas_height=height)
    chart1 = ChartModule([{"Label": "Number of Humans in Environment", "Color": "#0073ff"}], 10, 25)
    chart2 = ChartModule([{"Label": "Average Panic", "Color": "#AA0000"}], 10, 25)
    chart3 = ChartModule([{"Label": "Average Speed", "Color": "#47c12f"}], 10, 25)
    # hist1 = HistogramModule(np.arange(0, 5, 0.1), height, width)

    return [canvas, chart1, chart2, chart3]


def obstacles(width: float, height: float, doorsize: float) -> List[Obstacle]:
    # Define all walls in the system
    return [
        Wall(np.array([0, 0]), np.array([0, height/2 - doorsize/2])),
        Wall(np.array([0, height/2 + doorsize/2]), np.array([0, height])),
        Wall(np.array([0, 0]), np.array([width, 0])),
        Wall(np.array([width, 0]), np.array([width, height])),
        Wall(np.array([0, height]), np.array([width/2 - doorsize/2, height])),
        Wall(np.array([width/2 + doorsize/2, height]), np.array([width, height])),
    ]


def exits(width: float, height: float, doorsize: float):
    return [
        Exit(np.array([width/2 - doorsize/2, height]), np.array([width/2 + doorsize/2, height])),
        Exit(np.array([0, height/2 - doorsize/2]), np.array([0, height/2 + doorsize/2])),
    ]


def model_parameters(width: float, height: float, obstacle_list: List[Obstacle], exits: List[Exit]) -> Dict:
    return {
        "population": UserSettableParameter(
            "slider",
            "Population",
            100,
            10,
            1000,
            10,
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
            0.5,
            0.02,
            0.6,
            0.01,
            description="Relaxation Time"),
        "obstacles": obstacle_list,
        "exits": exits,
    }


# TODO: Find nicer way to do all these default settings, perhaps just OneExit?
def launch():
    width = 15
    height = 15
    doorsize = 1

    server = ModularServer(
        model_cls=SocialForce,
        visualization_elements=graphical_elements(width, height),
        name="Room evacuation with panic",
        model_params=model_parameters(width, height, obstacles(width, height, doorsize), exits(width, height, doorsize))
    )

    server.launch()
