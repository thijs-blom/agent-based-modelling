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
        init_pos,
        velocity,
        max_speed,
        vision,
        mass,
        radii,
        lam,
        current_timestep,
        init_speed,
        init_desired_speed,
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
            init_desire_speed : the initial desired speed
            is_leader: whether the agent is a leader
        """
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.init_pos = np.array(init_pos)
        self.max_speed = max_speed
        self.min_speed = 0.2
        self.velocity = velocity
        self.vision = vision
        self.mass = mass
        self.radii = radii
        self.lam = lam
        self.timestep = current_timestep
        self.init_desired_speed = init_desired_speed
        self.init_speed = init_speed
        self.is_leader = is_leader
        self.energy = 1
        self.strategy = strategy
        # Go to the (center of) the nearest exit
        self.dest = self.nearest_exit().get_center()
        
        # Tau it the characteristic reaction time, as they move once a step/time unit, this is set to 1
        self.tau = 2

    def desired_dir(self):
        """ Compute the desired direction of the agent
            When strategy is 'nearest exit', the desired direction is the closest exit;
            When strategy is 'follow the crowd', the desired direction is closest exit (80% likihood) or neighbor_direction (20% likelihood);
            When strategy is 'least crowded exit', the desired direction is the least crowded exit.
        """
        if self.strategy == 'nearest exit':
            # Go to the (center of) the nearest exit
            self.dest = self.nearest_exit().get_center()
            dir = self.dest - self.pos
            dir /= np.linalg.norm(dir)

        elif self.strategy == 'follow the crowd':
            self.dest = self.nearest_exit().get_center()
            dest_dir = self.dest - self.pos
            dest_dir /= np.linalg.norm(dest_dir)
            neighbor_dir = self.neighbor_direction(dest_dir)
            neighbor_dir /= np.linalg.norm(neighbor_dir)

            # if exit is within 15 meters, the destination is the nearest exit
            # otherwise the destination is a mixed a nearest exit and the neighbors
            if np.linalg.norm(self.pos - self.dest) > 50:
                rand = np.random.random()
                print(rand)
                if rand > 0.8:
                    dir = neighbor_dir
                else:
                    dir = dest_dir
            else:
                dir = dest_dir

            dir /= np.linalg.norm(dir)

        elif self.strategy == 'least crowded exit':
            self.dest = self.least_crowded_exit().get_center()
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

    def least_crowded_exit(self):
        # define exit business as a dictionary
        busyness = {}
        for i in range(len(self.model.exits)):
            exit = self.model.exits[i]
            # exit_name = f'exit{i}'
            busyness[i] = len(self.model.space.get_neighbors(exit.get_center(), 10, False))
        nb_exit = min(busyness, key=busyness.get)
        return self.model.exits[nb_exit]

    def neighbor_direction(self,origin_dir):
        # find the neighbors' direction
        neighbours = self.model.space.get_neighbors(self.pos, self.vision, False)
        # original direction is the same as the nearest exit
        sum_of_direction = origin_dir

        if len(neighbours) >0 :
            for other in neighbours:
                v = other.velocity
                sum_of_direction += v / np.linalg.norm(v)
                sum_of_direction /= np.linalg.norm(sum_of_direction)
        
        return sum_of_direction
    
    # def panic_index(self):
    #     """Compute the panic index of agent using average speed""" 
    #     # Compute average speed into desired direction for the agent
    #     pos_change = self.pos - self.init_pos
    #     desire_direction = self.desired_dir()
    #     proj_distance = np.dot(pos_change, desire_direction) / np.linalg.norm(desire_direction)
        
    #     # if the past movement is on the opposite direction the agent panic
    #     if proj_distance < 0:
    #         return 1
    #     else:
    #         avg_speed = proj_distance / self.timestep
    #         if self.init_desired_speed > avg_speed:
    #             return 1 - avg_speed / self.init_desired_speed
    #         else:
    #             return 0

    def panic_index(self):
        """Compute the panic index of agent using average speed"""        
        # Compute average speed of the neighbourhood
        neighbourhood_speed = 0
        neighbours = self.model.space.get_neighbors(self.pos, self.vision, False)
        if len(neighbours) > 0:
            for neighbour in neighbours:
                neighbourhood_speed += np.linalg.norm(neighbour.velocity)
            neighbourhood_speed /= len(neighbours)
            print(neighbourhood_speed)

<<<<<<< Updated upstream
            agent_speed = np.linalg.norm(self.velocity)
            # Return the panic index (eq 12)
            if neighbourhood_speed < self.init_desired_speed:
                return 1 - neighbourhood_speed / self.init_desired_speed
=======
            # testing testing
            if neighbourhood_speed / self.max_speed > 1:
                raise ValueError

            # Return the panic index (eq 12 baseline)
            if neighbourhood_speed < self.init_speed:
                return 1 - neighbourhood_speed / self.init_speed
>>>>>>> Stashed changes
            else:
                return 0
        return 0

    def desired_speed(self):
        """ Compute the current desired speed of agent : v0_i(t)"""
        # eq 11 of baseline paper
        n = self.panic_index()
        return (1-n) * self.init_desired_speed + n * self.max_speed

    def panic_noise_effect(self):
        """Compute the force of noise scaled by individual's panic level"""
        # scale of noise : eq 10 from baseline paper
        panic_index = self.panic_index()
        noise_scale = (1-panic_index)* Human.min_noise + panic_index * Human.max_noise
        # the random force is assumed to be random normal with scale = noise scale
        return 0 #np.random.normal(loc=0.0, scale=noise_scale)
        # return 0

    def acceleration_term(self):
        """Compute the acceleration Term of agent"""
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
        
        # the attraction force from leaders
        if other.is_leader:
            # when the other agent is a leader, the agent is attracted
            # TODO: Check if we agree on the formulation of leading force

            # att_force is not explicitely given but hints of design is provided on page 11 paragraph 2
            temp_leader = contact_diff / Human.lead_range
            # lead_strength is provided
            # for attraction force of leader we need to revert the direction 
            # such that n_ki points from the agent i to the leader
            att_force = Human.lead_strength * np.exp(temp_leader) * n_ij_val * (self.lam + (1-self.lam)* 0.5 * (1+cos_phi_ik(self, other)) )
        else:
            att_force = 0
        
        # the crashing force: if and only if the distance between two agents is smaller than the sum of their radiis
        
        # if contact_diff >= 0 then two agents crash into each other
        if contact_diff >= 0:
            delta_ji = (other.velocity - self.velocity) * t_ij(self, other)
            sliding_force = Human.sfc * contact_diff * delta_ji * n_ij_val
            body_force = Human.bfc * contact_diff * n_ij_val
            crashing_force = sliding_force + body_force
            crashing_strength = np.linalg.norm(crashing_force)
            deduction_param = 0.000001
            energy_lost = ( crashing_strength / self.mass ) * deduction_param
            # very big force can just kill people? seems not very realistic? but it's also not good to say maximum damage is a constant?
            if energy_lost > 0.25:
               energy_lost = 0.25 
            self.energy -= energy_lost
            self.energy = np.clip(self.energy,0,1)
            # print(f'crashed with another guy! : energy lost {energy_lost}')
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
        temp = contact_diff / Human.obs_range
        theta_val = theta(contact_diff)
        tib_val = t_ib(self,obstacle_point)
        n_ib_val = n_ib(self,obstacle_point)

        # eq 7 in baseline
        obt_force =  Human.obs_strength * np.exp(temp) + Human.bfc * theta_val
        obt_force *= n_ib_val
        obt_force -= Human.sfc * theta_val * (self.velocity * tib_val) * tib_val

        crashing_force = Human.bfc * theta_val * n_ib_val - Human.sfc_wall * theta_val * (self.velocity * tib_val) * tib_val
        crashing_strength = np.linalg.norm(crashing_force)
        deduction_param = 0.000005
        energy_lost = theta_val * ( crashing_strength / self.mass ) * deduction_param
        self.energy -= energy_lost
        self.energy = np.clip(self.energy,0,1)
        print(f'crashed with the walls! : energy lost {energy_lost}')

        return obt_force
<<<<<<< Updated upstream
=======
    
    def comfortness(self):
        """Compute the comfortness of agent by the time he escape"""
        # formula is given but it is optional for now
        pass
>>>>>>> Stashed changes

    def step(self):
        """
        Compute all forces acting on this agent, update its velocity and move.
        """
        # Compute accelaration term of agent
        self.velocity += self.acceleration_term()

        # neighbours = self.model.space.get_neighbors(self.pos, 0.5, False)
        
        # for other in neighbours:
        #     # resource for this calculation?
        #     distance = np.sqrt((self.pos[0] - other.pos[0])**2 + (self.pos[1] - other.pos[1])**2)
        #     self.energy -= distance / 10
        
        # if self.energy <= 0:
        #     # casualty = Dead(self.pos, self.mass/100)
        #     # perhaps makes more sense to let the dead agent obstacle radii be a function of the original radii?
        #     casualty = Dead(self.pos, self.radii*5)
        #     self.model.obstacles.append(casualty)
        #     self.model.remove_agent(self)
        #     print('another died!')
        #     return
 
        neighbours = self.model.space.get_neighbors(self.pos, self.vision, False)

        for other in neighbours:
            #If we also define obstacles as agent, then we should first classify the type of agent then we apply human or obs force
            # Compute repulsive effect from other people
            if isinstance(other, Human):
                self.velocity += self.people_effect(other) / self.mass
            # Compute repulsive effect from obstacles
            elif isinstance(other, Obstacle):
                self.velocity += self.boundary_effect(other) / self.mass

        # Compute random noise force
        self.velocity += self.panic_noise_effect()
        # Update the movement, position features of the agent
<<<<<<< Updated upstream
        self.speed = np.clip(np.linalg.norm(self.velocity), 0, self.max_speed)
        # so speed is impacked by the remainly energy of individual, the minimum speed is applied to ensured badly injured agent still move out
        # uncommented line 343 - 350 and comment below line if we want energy = 0 agent to be dead and become an obstacle
        #self.speed = np.clip(self.speed * self.energy, self.min_speed, self.max_speed)
=======
        print(np.linalg.norm(self.velocity))
        self.avg_speed = np.clip(np.linalg.norm(self.velocity), 0, self.max_speed)
>>>>>>> Stashed changes
        self.velocity /= np.linalg.norm(self.velocity)
        self.velocity *= self.avg_speed
        new_pos = self.pos + self.velocity
        
        # if out of bounds, put at bound
        if new_pos[0] > self.model.space.width:
            new_pos[0] = self.model.space.width - 0.00001
        elif new_pos[0] < 0:
            new_pos[0] = 0

        if new_pos[1] > self.model.space.height:
            new_pos[1] = self.model.space.height -0.00001
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
