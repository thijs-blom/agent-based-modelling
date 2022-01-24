import numpy as np
# from mesa import Agent
from CommonHuman2002 import CommonHuman
from obstacle import Obstacle
from dead import Dead


class Human(CommonHuman):
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
        soc_strength: the interaction strength between agent and the others
        soc_range: the range of replusive interaction 
    """

    def __init__(
        self,
        unique_id,
        model,
        pos,
        velocity,
        max_speed,
        vision,
        mass,
        radii,
        lam,
        current_timestep,
        avg_speed,
        init_speed,
        is_leader,
        strategy,
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
            is_leader: whether the agent is a leader
        """
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.max_speed = max_speed
        self.velocity = velocity
        self.vision = vision
        # mass is a random value from uniform 0-100 ?
        # self.mass = np.random.random() * 100
        self.mass = mass
        self.radii = radii
        self.lam = lam
        self.timestep = current_timestep
        self.avg_speed = avg_speed
        self.init_speed = 1#init_speed
        # Go to the (center of) the nearest exit
        self.dest = self.nearest_exit().get_center()
        
        self.tau = 0.5

    def desired_dir(self):
        """ Compute the desired direction of the agent
            When strategy is 'nearest exit', the desired direction is the closest exit;
            When strategy is 'follow the crowd', the desired direction is closest exit (80% likihood) or neighbor_direction (20% likelihood);
            When strategy is 'least crowded exit', the desired direction is the least crowded exit.
        """
        # Go to the (center of) the nearest exit
        self.dest = self.nearest_exit().get_center()
        dir = self.dest - self.pos
        dir /= np.linalg.norm(dir)
        return dir

    def nearest_exit(self):
        """Find the nearest exit relative to this agent"""
        closest = None
        smallest_dist = np.inf
        for exit in self.model.exits:
            dist = np.linalg.norm(exit.get_center() - self.pos)
            if dist < smallest_dist:
                closest = exit
                smallest_dist = dist
        return closest
    
    # def panic_index(self):
    #     """Compute the panic index of agent using average speed"""        
    #     # Compute average speed of the neighbourhood
    #     neighbourhood_speed = 0
    #     neighbours = self.model.space.get_neighbors(self.pos, self.vision, False)
    #     if len(neighbours) > 0:
    #         for neighbour in neighbours:
    #             neighbourhood_speed += neighbour.avg_speed
    #         neighbourhood_speed /= len(neighbours)

    #     # testing testing
    #     if neighbourhood_speed / self.max_speed > 1:
    #         raise ValueError

    #     # Return the panic index (eq 12, but then the divisor and divided flipped)
    #     if neighbourhood_speed > self.init_speed:
    #         return 1 - self.init_speed / neighbourhood_speed
    #     else:
    #         return 0

    def desired_speed(self):
        """ Compute the current desired speed of agent : v0_i(t)"""
        # eq 11 of baseline paper
        n = 0 #self.panic_index()
        return (1-n) * self.init_speed + n * self.max_speed

    # def panic_noise_effect(self):
    #     """Compute the force of noise scaled by individual's panic level"""
    #     # scale of noise : eq 10 from baseline paper
    #     panic_index = self.panic_index()
    #     noise_scale = (1-panic_index)* Human.min_noise + panic_index * Human.max_noise
    #     # the random force is assumed to be random normal with scale = noise scale
    #     return np.random.normal(loc=0.0, scale=noise_scale)
    #     # return 0

    def acceleration_term(self):
        """Compute the acceleration Term of agent"""
        print(self.desired_dir())
        return (self.desired_speed() * self.desired_dir() - self.velocity) / self.tau

    def people_effect(self, other):
        """Compute People effect = Repulsive effect from other people + attraction effect from leaders"""
        
        # eq 4 in baseline
        def n_ij(agent1, agent2):
            """The normalized vector pointing from agent 2 to 1"""
            return (agent1.pos - agent2.pos) / d_ij(agent1, agent2)

        def cos_phi_ij(agent1, agent2):
            """The cos(angle=phi), 
                phi is the angle between agent 2 to agent 1's force and the desired direction"""
            vi = agent1.velocity
            return - n_ij(agent1, agent2) * vi / np.linalg.norm(vi)

        def cos_phi_ik(agent, leader):
            """The cos(angle=phi), 
                phi is the angle between agent 2 to agent 1's force and the desired direction"""
            vi = agent.velocity
            # for attraction force of leader we need to revert the direction 
            # such that n_ki points from the agent i to the leader
            return - n_ij(leader, agent) * vi / np.linalg.norm(vi)

        def r_ij(agent1, agent2):
            """The sum of radii of both agents"""
            return agent1.radii + agent2.radii

        def d_ij(agent1, agent2):
            """The distance between two agents."""
            return np.linalg.norm(agent1.pos - agent2.pos)

        def t_ij(agent1, agent2):
            """Compute the tangential direction on the closest point on obstacle"""
            return np.flip(n_ij(agent1, agent2))
        
        # define some temperal value for the following computations
        contact_diff = r_ij(self,other) - d_ij(self, other)
        n_ij_val = n_ij(self,other)

        # the social replusive (distancing) force: eq 3 in baseline
        temp = contact_diff / Human.soc_range
        soc_force = Human.soc_strength * np.exp(temp) * n_ij_val * (self.lam + (1-self.lam)* 0.5 * (1+cos_phi_ij(self, other)) )

        # the total force cause by the other agent j to self agent i is repulsive + attraction force  
        return 0 #soc_force

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
        temp = contact_diff / Human.obs_range
        theta_val = theta(contact_diff)
        tib_val = t_ib(self,obstacle_point)
        n_ib_val = n_ib(self,obstacle_point)

        # eq 7 in baseline
        obt_force =  Human.obs_strength * np.exp(temp) + Human.bfc * theta_val
        obt_force *= n_ib_val
        obt_force -= Human.sfc * theta_val * (self.velocity * tib_val) * tib_val
        
        return 0 #obt_force

    def step(self):
        """
        Compute all forces acting on this agent, update its velocity and move.
        """
        # Compute accelaration term of agent
        #self.velocity = 0
        print(self.velocity, self.init_speed, self.desired_dir())
        self.velocity += self.acceleration_term()

        # neighbours = self.model.space.get_neighbors(self.pos, self.vision, False)
        # for other in neighbours:
        #     #If we also define obstacles as agent, then we should first classify the type of agent then we apply human or obs force
        #     # Compute repulsive effect from other people
        #     if isinstance(other, Human):
        #         self.velocity += self.people_effect(other) / self.mass
        #     # Compute repulsive effect from obstacles
        #     elif isinstance(other, Obstacle):
        #         self.velocity += self.boundary_effect(other) / self.mass

        # Compute random noise force
        #self.velocity += self.panic_noise_effect()
        # Update the movement, position features of the agent
        self.speed = np.clip(np.linalg.norm(self.velocity), 0, self.max_speed)
        self.velocity /= np.linalg.norm(self.velocity)
        self.velocity *= self.speed
        new_pos = self.pos + self.velocity
        
        # if out of bounds, put at bound
        if new_pos[0] > self.model.space.width:
            new_pos[0] = self.model.space.width - 0.00001
        elif new_pos[0] < 0:
            new_pos[0] = 0
        if new_pos[1] > self.model.space.height:
            new_pos[1] = self.model.space.height - 0.00001
        elif new_pos[1] < 0:
            new_pos[1] = 0
        self.model.space.move_agent(self, new_pos)

        self.timestep += 1
        # Remove once the desitination is reached
        # TODO: list all the parameter of agent
        # velocity
        exit = self.nearest_exit()
        if exit.in_exit(self.pos):
            self.model.remove_agent(self)
