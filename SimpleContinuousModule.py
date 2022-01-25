from mesa.visualization.ModularVisualization import VisualizationElement
from dead import Dead


class SimpleCanvas(VisualizationElement):
    local_includes = ["simple_continuous_canvas.js"]
    portrayal_method = None
    #canvas_height = 500
    #canvas_width = 500

    def __init__(self, agent_portrayal, wall_portrayal, dead_portrayal, canvas_width, canvas_height):
        """
        Instantiate a new SimpleCanvas
        """
        self.agent_portrayal = agent_portrayal
        self.wall_portrayal = wall_portrayal
        self.dead_portrayal = dead_portrayal
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
            x1 = x / model.space.x_max
            y1 = y / model.space.y_max
            portrayal["x"] = x1
            portrayal["y"] = y1
            space_state.append(portrayal)

            # portrayal = self.wall_portrayal(obj)
            # portrayal["pos1"] = [x1*self.canvas_width, y1* self.canvas_height]
            # portrayal["pos2"] = [0, 1]
            # portrayal['w'] = 1
            # space_state.append(portrayal)


        for obj in model.obstacles:
            if isinstance(obj, Dead):
                portrayal = self.dead_portrayal(obj)
                x, y = obj.pos
                x /= model.space.x_max
                y /= model.space.y_max
                portrayal["x"] = x
                portrayal["y"] = y
                space_state.append(portrayal)
            else:
                portrayal = self.wall_portrayal(obj)
                
                x1 = obj.p1[0] * (self.canvas_width / model.space.x_max)
                y1 = obj.p1[1] * (self.canvas_height / model.space.y_max)
                x2 = obj.p2[0] * (self.canvas_width / model.space.x_max)
                y2 = obj.p2[1] * (self.canvas_height / model.space.y_max)
                    
                portrayal["pos1"] = [x1, y1]
                portrayal["pos2"] = [x2, y2]
                space_state.append(portrayal)

        return space_state
