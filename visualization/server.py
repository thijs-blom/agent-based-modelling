from mesa.datacollection import DataCollector
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.ModularVisualization import VisualizationElement

from visualization.simple_continuous_module import SimpleCanvas
from model.exit import Exit
from model.wall import Wall
from model.social_force import SocialForce

import numpy as np


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


# Visualization parameters
canvas_width = 500
canvas_height = 500

# Model parameters
width = 15
height = 15
door_size = 1

visualization_elements = [
    SimpleCanvas(canvas_width=canvas_width, canvas_height=canvas_height),
    ChartModule([{"Label": "Number of Humans in Environment", "Color": "#0073ff"}], 10, 25),
    ChartModule([{"Label": "Average Panic", "Color": "#AA0000"}], 10, 25),
    ChartModule([{"Label": "Average Speed", "Color": "#47c12f"}], 10, 25),
    # hist1 = HistogramModule(np.arange(0, 5, 0.1), height, width)
]

obstacles = [
        Wall(np.array([0, 0]), np.array([0, height/2 - door_size/2])),
        Wall(np.array([0, height/2 + door_size/2]), np.array([0, height])),
        Wall(np.array([0, 0]), np.array([width, 0])),
        Wall(np.array([width, 0]), np.array([width, height])),
        Wall(np.array([0, height]), np.array([width/2 - door_size/2, height])),
        Wall(np.array([width/2 + door_size/2, height]), np.array([width, height])),
    ]

exits = [
        Exit(np.array([width/2 - door_size/2, height]), np.array([width/2 + door_size/2, height])),
        Exit(np.array([0, height/2 - door_size/2]), np.array([0, height/2 + door_size/2])),
    ]

datacollector = DataCollector(
    model_reporters={
        "Number of Humans in Environment": lambda m: m.schedule.get_agent_count(),
        "Average Panic": lambda m: m.count_panic(),
        "Average Speed": lambda m: m.count_speed() / m.schedule.get_agent_count() if m.schedule.get_agent_count() > 0 else 0
    })

model_parameters = {
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
        "obstacles": obstacles,
        "exits": exits,
        "datacollector": datacollector
    }


def launch():
    # TODO: perhaps use OneExit for default visualization?
    server = ModularServer(
        model_cls=SocialForce,
        visualization_elements=visualization_elements,
        name="Room evacuation with panic",
        model_params=model_parameters
    )

    server.launch()
