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
    tau = 0.5
    min_noise = 0.1
    max_noise = 1
    lead_strength = 1
    lead_range = 5