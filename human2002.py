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
        bfc: body_force_constant
        sfc: sliding friction force constant
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


    def panic_noise_effect(self, panic_index):
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

        def t_ij(agent1, agent2):
            """Compute the tangential direction on the closest point on obstacle"""
            return np.flip(n_ij((agent1, agent2)))
        
        # define some temperal value for the following computations
        contact_diff = r_ij(self,other) - d_ij(self, other)
        n_ij_val = n_ij(self,other)

        # the social replusive (distancing) force: eq 3 in baseline
        temp = contact_diff / self.soc_range
        soc_force = self.soc_strength * np.exp(temp) * n_ij_val * (self.lam + (1-self.lam)* 0.5 * (1+cos_phi_ij(self, other)) )
        
        # the attraction force from leaders
        if other.is_leader:
            # when the other agent is a leader, the agent is attracted
            # TODO: Check if we agree on the formulation of leading force

            # att_force is not explicitely given but hints of design is provided on page 11 paragraph 2
            temp_leader = contact_diff / super().lead_range
            # lead_strength is provided
            # for attraction force of leader we need to revert the direction 
            # such that n_ki points from the agent i to the leader 
            att_force = super().lead_strength * np.exp(temp_leader) * n_ij_val * (self.lam + (1-self.lam)* 0.5 * (1+cos_phi_ik(self, other)) )
        else:
            att_force = 0
        
        # the crashing force: if and only if the distance between two agents is smaller than the sum of their radiis
        
        # if contact_diff >= 0 then two agents crash into each other
        if contact_diff >= 0:
            delta_ji = (other.velocity - self.velocity) * t_ij(self, other)
            sliding_force = super().sfc * contact_diff * delta_ji * n_ij_val 
            body_force = super().bfc * contact_diff * n_ij_val 
            crashing_force = sliding_force + body_force
        else:
            crashing_force = 0
        # the total force cause by the other agent j to self agent i is repulsive + attraction force  
        return soc_force + att_force + crashing_force

    def boundary_effect(self, obstacle):
        """Repulsive effect from an obstacle"""
        # TODO: Check if obstacle is a neighbors

        def d_ib(agent1,obstacle_point):
            """Compute the distance from agent i to the effect point of obstacle"""
            return agent1.pos - obstacle_point

        def n_ib(agent1, obstacle_point):
            """The normalized vector pointing from the effect point of obstacle to agent"""
            return (agent1.pos - obstacle_point) / d_ib(agent1, obstacle_point)
        
        def t_ib(agent1, obstacle_point):
            """Compute the tangential direction on the closest point on obstacle"""
            return np.flip(n_ib(agent1, obstacle_point))
        
        def theta(z):
            """The identifier function return z if z >= 0, otherwise 0"""
            if z >= 0:
                return z
            else:
                return 0

        # TODO: There is more efficient way to write the function

        obstacle_point = obstacle.get_closest_point(self)
        contact_diff = self.radii - d_ib(self,obstacle_point)
        temp = contact_diff / self.soc_range
        theta_val = theta(contact_diff)
        tib_val = t_ib(self,obstacle_point)

        # eq 7 in baseline
        obt_force =  self.soc_strength * np.exp(temp) + super().bfc * theta_val
        obt_force *= n_ib(self,obstacle_point)
        obt_force -= super().sfc * theta_val * (self.velocity * tib_val) * tib_val
        
    
    def comfortness(self):
        """Compute the comfortness of agent by the time he escape"""
        # formula is given but it is optional for now
        pass

    def step(self):
        """
        Compute all forces acting on this agent, update its velocity and move
        """
        # Compute attractive effect to destination

        # Compute repulsive effect from other people

        # Compute repulsive effect from obstacles

        # Compute attractive effect to points/people of interest

        # Update the position

        # if out of bounds, put at bound

        # Remove once the desitination is reached
