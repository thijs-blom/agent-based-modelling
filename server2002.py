from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
<<<<<<< Updated upstream
from mesa.batchrunner import BatchRunner
=======
from mesa.batchrunner import FixedBatchRunner
>>>>>>> Stashed changes

from model2002 import SocialForce
from SimpleContinuousModule import SimpleCanvas
from wall import Wall
from exit import Exit

import numpy as np

def human_draw(agent):
    return {"Shape": "circle", "r": 1, "Filled": "true", "Color": "Red"}

def wall_draw(wall):
    return {"Shape": "line", "w": 5, "Color": "Black"}

def dead_draw(dead):
    return {"Shape": "circle", "r": 1, "Filled": "true", "Color": "Black"}

canvas = SimpleCanvas(human_draw, wall_draw, dead_draw, canvas_width=500, canvas_height=500)
chart0 =  ChartModule([{"Label": "Number of Humans in Environment", "Color": "#AA0000"}], 10, 25)
chart1 =  ChartModule([{"Label": "Number of Casualties", "Color": "#AA0000"}], 10, 25)
chart2 =  ChartModule([{"Label": "Average Energy", "Color": "#AA0000"}], 10, 25)
chart3 =  ChartModule([{"Label": "Average Speed", "Color": "#AA0000"}], 10, 25)

width = 100
height = 100

wall1 = Wall(np.array([50, 0]), np.array([50, 10]))
side_wall1 = Wall(np.array([0, 2]), np.array([0, height]))
side_wall2 = Wall(np.array([0, 0]), np.array([width, 0]))
side_wall3 = Wall(np.array([width, 0]), np.array([width, height]))
side_wall4 = Wall(np.array([0, height]), np.array([width/2 - 1, height]))
side_wall5 = Wall(np.array([width/2 + 1, height]), np.array([width, height]))

init_obstacles = [side_wall1, side_wall2, side_wall3, side_wall4, side_wall5, wall1]

<<<<<<< Updated upstream
# exit1 = Exit(np.array([0,0]), np.array([0,2]))
# exit2 = Exit(np.array([48,100]), np.array([50,100]))

exit1 = Exit(np.array([0,0]), np.array([0,5]))
exit2 = Exit(np.array([45,100]), np.array([50,100]))
=======
exit1 = Exit(np.array([0,0]), np.array([0,2]))
exit2 = Exit(np.array([width/2 - 1,height]), np.array([width/2 + 1,height]))
>>>>>>> Stashed changes

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
    "exits": [exit1, exit2],
    "init_amount_obstacles": len(init_obstacles)
}

<<<<<<< Updated upstream
server = ModularServer(SocialForce, [canvas, chart0, chart3], "Escape Panic", model_params)

model_reporters={
    "Number of Humans in Environment": lambda m: SocialForce.schedule.get_agent_count(),
    # "Number of Casualties": lambda m: len(self.obstacles) - self.init_amount_obstacles,
    # "Average Energy": lambda m: self.count_energy(m) / self.population,
    "Average Speed" : lambda m: SocialForce.count_speed(m) / SocialForce.schedule.get_agent_count() if SocialForce.schedule.get_agent_count() > 0 else 0
    }

batch = BatchRunner(SocialForce,
                    max_steps=1000,
                    iterations=10,
                    model_reporters= model_reporters,
                    display_progress=True)
batch.run_all()
=======
server = ModularServer(SocialForce, [canvas, chart0, chart1, chart2, chart3], "Escape Panic", model_params)
#server = FixedBatchRunner.run_all(SocialForce)
>>>>>>> Stashed changes

server.launch()
