from mesa.visualization.ModularVisualization import VisualizationElement


class SimpleCanvas(VisualizationElement):
    local_includes = ["simple_continuous_canvas.js"]
    portrayal_method = None
    canvas_height = 500
    canvas_width = 500

    def __init__(self, agent_portrayal, wall_portrayal, canvas_height=500, canvas_width=500):
        """
        Instantiate a new SimpleCanvas
        """
        self.agent_portrayal = agent_portrayal
        self.wall_portrayal = wall_portrayal
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        new_element = "new Simple_Continuous_Module({}, {})".format(
            self.canvas_width, self.canvas_height
        )
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        space_state = []
        for obj in model.schedule.agents:
            portrayal = self.agent_portrayal(obj)
            x, y = obj.pos
            x = (x - model.space.x_min) / (model.space.x_max - model.space.x_min)
            y = (y - model.space.y_min) / (model.space.y_max - model.space.y_min)
            portrayal["x"] = x
            portrayal["y"] = y
            space_state.append(portrayal)

        for obj in model.obstacles:
            portrayal = self.wall_portrayal(obj)
            portrayal["pos1"] = obj.p1.tolist()
            portrayal["pos2"] = obj.p2.tolist()
            space_state.append(portrayal)

        return space_state
