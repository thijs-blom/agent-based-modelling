import numpy as np
from mesa import Agent


class CommonHuman(Agent):
    """
    An agent following rules from the social force model.

    List of arguments fixed for every agent:
    
        tau: the default relaxation parameter
        min_noise: the min scale of noise added to agent's movement, SAME for each agent
        max_noise: the max scale of noise added to agent's movement, SAME for each agent
        lead_strength: the leading strength of any leader neighbor agent
        lead_range: the leading force impact range of any leader neighbor agent

    """
    def __init__(
        self,
        unique_id,
        model,

    ):
        """
        Create a new Common Human agent

        Args:
            unique_id: Unique agent identifier.
            model: Reference to the model object this agent is part of
        """
        super().__init__(unique_id, model)

   
# parameter from 2002
    min_noise = 0.5
    max_noise = 1
    lead_strength = 2000
    lead_range = 0.08
    soc_strength = 2000
    soc_range = 0.08
    bfc = 120000
    sfc = 240000
    sfc_wall = 200
<<<<<<< Updated upstream
    obs_strength = 500
=======
    obs_strength = 2000
>>>>>>> Stashed changes
    obs_range = 0.08
