from mesa.visualization.ModularVisualization import VisualizationElement


class SimpleCanvas(VisualizationElement):
    local_includes = ["visualization/simple_continuous_canvas.js"]
    portrayal_method = None
    canvas_height = 500
    canvas_width = 500

    def __init__(self, canvas_width, canvas_height):
        """
        Instantiate a new SimpleCanvas
        """
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        new_element = "new Simple_Continuous_Module({}, {})".format(
            self.canvas_width, self.canvas_height
        )
        self.js_code = "elements.push(" + new_element + ");"
        
    def render(self, model):
        space_state = []
        for obj in model.schedule.agents:
            brightness = round(100 - obj.panic * 100)
            portrayal = {"Shape": "circle", "r": obj.radius * (self.canvas_width / model.space.x_max), "Filled": "true", "Color": f"rgb(255, {brightness}, {brightness * 1.4})"}
            if obj.strategy != 'nearest exit':
                portrayal['Color'] = f"rgb({brightness}, {brightness * 2}, 255)"
            x, y = obj.pos
            x1 = x / model.space.x_max
            y1 = y / model.space.y_max
            portrayal["x"] = x1
            portrayal["y"] = y1
            space_state.append(portrayal)

        for obj in model.obstacles:
            portrayal = {"Shape": "line", "w": 5, "Color": "Black"}

            x1 = obj.p1[0] * (self.canvas_width / model.space.x_max)
            y1 = obj.p1[1] * (self.canvas_height / model.space.y_max)
            x2 = obj.p2[0] * (self.canvas_width / model.space.x_max)
            y2 = obj.p2[1] * (self.canvas_height / model.space.y_max)

            portrayal["pos1"] = [x1, y1]
            portrayal["pos2"] = [x2, y2]
            space_state.append(portrayal)

        return space_state
