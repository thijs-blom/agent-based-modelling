import numpy as np
# from mesa import Agent
from CommonHuman2002 import *

class Human2002(CommonHuman):
    """
    An agent following rules from the social force model.
    Adapted from:
        Helbing, D., Farkas, I. J., Molnar, P., & Vicsek, T. (2002). 
        Simulation of pedestrian crowds in normal and evacuation situations. 
        Pedestrian and evacuation dynamics, 21(2), 21-58.

    The following parameter values are set in CommonHuman, since they are the same among agents
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
        pos,
        dest,
        velocity,
        max_speed,
        vision,
        mass,
        radii,
        lam,
        panic,
        current_timestep,
        avg_speed,
        init_speed,
        soc_strength,
        soc_range,
        is_leader,
    ):
        """
        Create a new Human agent

        Args:
            unique_id: Unique agent identifyer.
            model: Reference to the model object this agent is part of
            pos: Starting position, center of mass for agent
            dest: The destination the agent wants to reach
            velocity: Velocity vector indicating speed of movement
            max_speed: the maximum speed of agent
            vision: Radius to look around for nearby agents.
            mass: the weight / mass of the agent 
            radii: the radii of the agent
            lam: the 'front impact' parameter of agent to describe the anisotropic character of pedestrian interaction
            panic: The panic level of agent
            current_timestep: The current time step t
            avg_speed: The speed updated at t-1 or initialized at t = 0
            init_speed: The initial speed of agent
            soc_strength: the interaction strength between agent and the others
            soc_range: the range of replusive interaction 
            is_leader: whether the agent is a leader
        """
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.dest = np.array(dest)
        self.max_speed = max_speed
        self.velocity = velocity
        self.vision = vision
        self.mass = mass
        self.radii = radii
        self.lam = lam
        self.panic = panic
        self.timestep = current_timestep
        self.avg_speed = avg_speed
        self.init_speed = init_speed
        self.soc_strength = soc_strength
        self.soc_range = soc_range
        self.is_leader = is_leader


        
        self.tau = 0.5

    def desired_dir(self):
        """Compute the desired direction of the agent"""
        # TODO: TRY AT SYSTEM FILE -> Redirect 'stuck-at-wall' agents
        # Add the switch of desired_direction when view to the exit is blocked by wall
        dir = self.dest - self.pos
        return dir / np.linalg.norm(dir)

    def panic_index(self):
        """Compute the panic index of agent using average speed"""
        # eq 11 of baseline paper
        if self.timestep > 1:
            self.avg_speed = (self.avg_speed * (self.timestep - 1) + self.speed) / self.timestep
        else:
            # if timestep = 0, then the avg_velocity is just the first velocity
            self.avg_speed = self.init_speed
            # so the individual's panic_index is initialized as 0 at the begining
            # TODO: Is initial velocity the max_speed ???
        
        # testing testing 
        if self.avg_speed / self.init_speed > 1:
            raise ValueError

        return 1 - self.avg_speed / self.init_speed

    def desired_speed(self):
        """ Compute the current desired speed of agent : v0_i(t)"""
        # eq 12 of baseline paper
        n = self.panic_index()
        return (1-n) * self.init_speed + n * self.max_speed


    def panic_noise_force(self, panic_index):
        """Compute the force of noise scaled by individual's panic level"""
        # scale of noise : eq 10 from baseline paper
        noise_scale = (1-panic_index)*super().min_noise + panic_index * super().max_noise
        # the random force is assumed to be random normal with scale = noise scale
        return np.random.normal(loc=0.0, scale=noise_scale)

    def acceleration_term(self):
        """Compute the acceleration Term of agent"""
        return (self.desired_speed * self.desired_dir() - self.velocity) / super().tau

    def people_effect(self, other):
        """Compute People effect = Repulsive effect from other people + attraction effect from leaders"""
        # TODO: Check if other is only the neighbors
        
        # eq 4 in baseline
        def n_ij(agent1, agent2):
            """The normalized vector pointing from agent 2 to 1"""
            return (agent1.pos - agent2.pos) / d_ij(agent1, agent2)

        def cos_phi_ij(agent1, agent2):
            """The cos(angle=phi), 
                phi is the angle between agent 2 to agent 1's force and the disired direction"""
            vi = agent1.volecity
            return - n_ij(agent1, agent2) * vi / np.linalg.norm(vi)

        def cos_phi_ik(agent, leader):
            """The cos(angle=phi), 
                phi is the angle between agent 2 to agent 1's force and the disired direction"""
            vi = agent.volecity
            # for attraction force of leader we need to revert the direction 
            # such that n_ki points from the agent i to the leader
            return - n_ij(leader, agent) * vi / np.linalg.norm(vi)

        def r_ij(agent1, agent2):
            """The sum of radii of both agents"""
            return agent1.radii + agent2.radii

        def d_ij(agent1, agent2):
            return np.linalg.norm(agent1.pos - agent2.pos)

        # eq 3 in baseline
        temp = ( r_ij(self,other) - d_ij(self,other) ) / self.soc_range
        soc_force = self.soc_strength * np.exp(temp) * n_ij(self,other) * (self.lam + (1-self.lam)* 0.5 * (1+cos_phi_ij(self, other)) )
        
        if other.is_leader:
            # when the other agent is a leader, the agent is attracted
            # TODO: Check if we agree on the formulation of leading force

            # att_force is not explicitely given but hints of design is provided on page 11 paragraph 2
            temp_leader = ( r_ij(self,other) - d_ij(self,other) ) / super().lead_range
            # lead_strength is provided
            # for attraction force of leader we need to revert the direction 
            # such that n_ki points from the agent i to the leader 
            att_force = super().lead_strength * np.exp(temp_leader) * n_ij(other,self) * (self.lam + (1-self.lam)* 0.5 * (1+cos_phi_ik(self, other)) )
        else:
            att_force = 0

        # the total force cause by the other agent j to self agent i is repulsive + attraction force  
        return soc_force + att_force

    def obstacle_effect(self, obstacle):
        """Repulsive effect from an obstacle"""
        # TODO: Check if obstacle is a neighbors
        # TODO: NOT FINISHED!!!
        # adaptation of eq 4 in baseline
        def n_ij(agent1, agent2):
            """The normalized vector pointing from agent 2 to 1"""
            return (agent1.pos - agent2.pos) / d_ij(agent1, agent2)

        def cos_phi_ij(agent1, agent2):
            """The cos(angle=phi), 
                phi is the angle between agent 2 to agent 1's force and the disired direction"""
            vi = agent1.volecity
            return - n_ij(agent1, agent2) * vi / np.linalg.norm(vi)

        def r_ij(agent1, agent2):
            """The sum of radii of both agents"""
            return agent1.radii + agent2.radii

        def d_ij(agent1, agent2):
            return np.linalg.norm(agent1.pos - agent2.pos)

        # eq 3 in baseline
        temp = ( r_ij(self,obstacle) - d_ij(self,obstacle) ) / self.soc_range
        soc_force = self.soc_strength * np.exp(temp) * n_ij(self,obstacle) * (self.lam + (1-self.lam)* 0.5 * (1+cos_phi_ij(self, obstacle)) )

    # def attract_effect(self, other_point):
    #     """Attractive effect to places/people of interest"""
    #     raise NotImplementedError

    # def sight_weight(self, f):
    #     """Compute the weight to account for sight"""
    #     # Parameters from paper
    #     c = 0.5
    #     cosphi = np.cos(np.radians(100))

    #     # Compare direction of the effect with our desired direction
    #     if np.dot(self.desired_dir(), f) >= np.linalg.norm(f) * cosphi:
    #         return 1
    #     else:
    #         return c

    
