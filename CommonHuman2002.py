import numpy as np
from mesa import Agent

class CommonHuman(Agent):
    """
    An agent following rules from the social force model.

    List of arguements fixed for every agent:
    
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
            unique_id: Unique agent identifyer.
            model: Reference to the model object this agent is part of
        """
        super().__init__(unique_id, model)

    # TODO: All the below parameters are not well defined yet !!!
    # TODO: Check 2002 for Default relaxation parameter
    # TODO: Check if more parameters can be fixed, or maybe varied per group of population, such as male and female
        # self.tau = 0.5
        # self.min_noise = 0.5
        # self.max_noise = 1
        # self.lead_strength = 2
        # self.lead_range = 10
        # self.soc_strength = 2
        # self.soc_range = 10
        # self.obs_strength = 5
        # self.obs_range = 0.5
        # self.bfc = 5
        # self.sfc = 5
   
# parameter from Li, M., Zhao, Y., He, L., Chen, W., & Xu, X. (2015).
# The parameter calibration and optimization of social force model for the real-life 2013 
# Ya’an earthquake evacuation in China. Safety science, 79, 243-253.
    tau = 0.5
    min_noise = 0.5
    max_noise = 1
    lead_strength = 2000
    lead_range = 5
    soc_strength = 2000
    soc_range = 0.08
    bfc = 120000
    sfc = 240000
    sfc_wall = 200
    obs_strength = 2000
    obs_range = 0.08