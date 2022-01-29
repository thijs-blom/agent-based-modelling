# Python imports
from __future__ import annotations
from logging import raiseExceptions
import numpy as np
import matplotlib.pyplot as plt

# Project imports
from CommonHuman import CommonHuman
from obstacle import Obstacle
from dead import Dead
from exit import Exit


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
        soc_range: the range of repulsive interaction
    """

    def __init__(
            self,
            unique_id,
            model,
            pos: np.ndarray,
            velocity: np.ndarray,
            max_speed: float,
            vision: float,
            mass: float,
            radius: float,
            lam: float,
            current_timestep: int,
            init_speed: float,
            init_desired_speed: float,
            is_leader: bool,
            relax_t: float,
            strategy: str,
    ):
        """
        Create a new Human agent

        Args:
            unique_id: Unique agent identifier.
            model: Reference to the model object this agent is part of
            pos: Starting position, center of mass for agent
            dest: The destination the agent wants to reach
            velocity: Velocity vector indicating speed of movement
            max_speed: the maximum speed of agent
            vision: Radius to look around for nearby agents.
            mass: the weight / mass of the agent 
            radius: the radii of the agent
            lam: the 'front impact' parameter of agent to describe the anisotropic character of pedestrian interaction
            panic: The panic level of agent
            current_timestep: The current time step t
            init_speed: The initial speed of agent
            init_desire_speed : the initial desired speed
            is_leader: whether the agent is a leader
        """
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.max_speed = max_speed
        self.min_speed = 0.2
        self.velocity = velocity
        self.vision = vision
        self.mass = mass
        self.radius = radius
        self.lam = lam
        self.timestep = current_timestep
        self.init_desired_speed = init_desired_speed
        self.init_speed = init_speed
        self.is_leader = is_leader
        self.energy = 1
        self.tau = 1 / relax_t
        self.strategy = strategy
        # Go to the (center of) the nearest exit
        # self.dest = self.nearest_exit().get_center()

        self.speed = init_speed


    def desired_dir(self) -> np.ndarray:
        """ Compute the desired direction of the agent
            When strategy is 'nearest exit', the desired direction is the closest exit;
            When strategy is 'follow the crowd', the desired direction is closest exit (80% likelihood) or neighbor_direction (20% likelihood);
            When strategy is 'least crowded exit', the desired direction is the least crowded exit.
        """
        if self.strategy == 'nearest exit':
            # Go to the (center of) the nearest exit
            self.dest = self.nearest_exit().get_center()
            dir = self.dest - self.pos
            dir /= np.linalg.norm(dir)

        elif self.strategy == 'follow the crowd':
            # Only follow the direction your neighbours are following
            neighbor_dir = self.neighbor_direction(self.velocity)
            neighbor_dir /= np.linalg.norm(neighbor_dir)
            dir = neighbor_dir

        elif self.strategy == 'hesitator':
            self.dest = self.nearest_exit().get_center()
            dest_dir = self.dest - self.pos
            dest_dir /= np.linalg.norm(dest_dir)
            neighbor_dir = self.neighbor_direction(dest_dir)
            neighbor_dir /= np.linalg.norm(neighbor_dir)

            # if exit is within 50 meters, the destination is the nearest exit
            # otherwise the destination is a mixed a nearest exit and the neighbors
            dir = neighbor_dir
            if np.linalg.norm(self.pos - self.dest) > self.vision:
                rand = np.random.random()
                if rand > 0.5:
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

    def nearest_exit(self) -> Exit:
        """Find the nearest exit relative to this agent"""
        closest = None
        smallest_dist = np.inf
        for exit in self.model.exits:
            dist = np.linalg.norm(exit.get_center() - self.pos)
            if dist < smallest_dist:
                closest = exit
                smallest_dist = dist
        return closest

    def least_crowded_exit(self) -> Exit:
        # define exit business as a dictionary
        busyness = {}
        for i, exit in enumerate(self.model.exits):
            # exit_name = f'exit{i}'
            busyness[i] = len(self.model.space.get_neighbors(exit.get_center(), 10, False))
        nb_exit = min(busyness, key=busyness.get)
        return self.model.exits[nb_exit]

    def neighbor_direction(self, origin_dir: np.ndarray) -> np.ndarray:
        # find the neighbors' direction
        neighbours = self.model.space.get_neighbors(self.pos, self.vision, False)
        # original direction is the same as the nearest exit
        sum_of_direction = origin_dir

        for other in neighbours:
            v = other.velocity
            sum_of_direction += v / np.linalg.norm(v)
            sum_of_direction /= np.linalg.norm(sum_of_direction)

        # TODO: check if it is desirable that the velocity may not be normalized if there are no neighbours
        return sum_of_direction

    def panic_index(self, desired_dir: np.ndarray = None):
        """Compute the panic index of agent using average speed"""
        # Compute average speed into desired direction for the agent
        if self.timestep == 0:
            self.v_bar = self.speed
        else:
            # progress can be either negative and positive
            if desired_dir is None:
                desired_dir = self.desired_dir()
            progress_t = np.dot(self.velocity, desired_dir)
            self.v_bar = (self.v_bar * (self.timestep-1) + progress_t) / self.timestep

        return 0 #1 - self.v_bar / self.init_desired_speed

    def desired_speed(self, panic_index: float = None):
        """ Compute the current desired speed of agent : v0_i(t)"""
        # eq 11 of baseline paper
        if panic_index is None:
            panic_index = self.panic_index()
        return (1-panic_index) * self.init_desired_speed + panic_index * self.max_speed

    def panic_noise_effect(self, panic_index: float = None):
        """Compute the force of noise scaled by individual's panic level"""
        # scale of noise : eq 10 from baseline paper
        if panic_index is None:
            panic_index = self.panic_index()
        noise_scale = (1 - panic_index) * Human.min_noise + panic_index * Human.max_noise
        # the random force is assumed to be random normal with scale = noise scale
        # return np.random.normal(loc=0.0, scale=noise_scale)
        return 0

    def acceleration_term(self, desired_dir: np.ndarray = None, panic_index: float = None) -> np.ndarray:
        """Compute the acceleration Term of agent"""
        if desired_dir is None:
            desired_dir = self.desired_dir()
        if panic_index is None:
            panic_index = self.panic_index(desired_dir)
        return (self.desired_speed(panic_index) * desired_dir - self.velocity) / self.tau

    def people_repulsive_effect(self, other: Human, d=None) -> np.ndarray:
        """Stub to split people_effect in more functions"""
        # Define some variables used in the equation defining the force
        d = d if d else np.linalg.norm(self.pos - other.pos)
        r = self.radius + other.radius - d
        n = (self.pos - other.pos) / d
        cosphi = np.dot(-n, self.velocity / np.linalg.norm(self.velocity))
        vision_term = (self.lam + (1 - self.lam) * (1 + cosphi) / 2)

        # the social repulsive (distancing) force: eq 3 in baseline
        social_force = Human.soc_strength * np.exp(r / Human.soc_range) * vision_term * n

        return social_force

    def leader_attractive_effect(self, leader: Human, d=None) -> np.ndarray:
        """Stub to split people_effect in more functions"""
        # TODO: Check if we agree on the formulation of leading force
        # This is f^{att}_{ik} as defined in page 11. Could also use f^{att}_{ij}.

        # Define some variables used in the equation defining the force
        d = d if d else np.linalg.norm(self.pos - leader.pos)
        r = self.radius + leader.radius - d
        n = (self.pos - leader.pos) / d
        cosphi = np.dot(-n, self.velocity / np.linalg.norm(self.velocity))
        vision_term = (self.lam + (1 - self.lam) * (1 + cosphi) / 2)

        # att_force is not explicitly given but hints of design is provided on page 11, paragraph 2
        # lead_strength is provided
        # for attraction force of leader we need to revert the direction
        # such that n_ki points from the agent i to the leader

        # TODO: According to the paper, the constant A_{ik} is usually small, negative and time-dependent.
        # Since this constant is positive in our model, a minus has been added in front of this calculation.
        att_force = - Human.lead_strength * np.exp(r / Human.lead_range) * vision_term * n

        return att_force

    def crash_effect(self, other: Human, d=None) -> np.ndarray:
        """Stub to split people_effect in more functions"""
        d = d if d else np.linalg.norm(self.pos - other.pos)
        r = self.radius + other.radius - d
        n = (self.pos - other.pos) / d
        t = np.flip(n) * np.array([-1, 1])

        # There is no crashing force if the agents are not touching, i.e. the distance
        # between them is bigger than the sum of their radii.
        if r < 0:
            return np.zeros(2)

        # Compute all total forces considered
        sliding_force = Human.sfc * r * np.dot(other.velocity - self.velocity, t) * t
        body_force = Human.bfc * r * n
        crashing_force = sliding_force + body_force

        # Consider energy loss from the crash
        crashing_strength = np.linalg.norm(crashing_force)
        deduction_param = 0.000001
        energy_lost = (crashing_strength / self.mass) * deduction_param
        # very big force can just kill people? seems not very realistic? but it's also not good to say maximum damage is a constant?
        if energy_lost > 0.25:
            energy_lost = 0.25
        self.energy -= energy_lost
        self.energy = np.clip(self.energy, 0, 1)
        # print(f'crashed with another guy! : energy lost {energy_lost}')

        return crashing_force

    def boundary_effect(self, obstacle: Obstacle, max_dist: float = None) -> np.ndarray:
        """Repulsive effect from an obstacle"""

        def theta(z: float) -> float:
            """The identifier function return z if z >= 0, otherwise 0"""
            return z if z > 0 else 0

        # Get the closest point of the obstacle w.r.t. the agent's current position
        obstacle_point = obstacle.get_closest_point(self.pos)
        # Compute the distance to the obstacle
        d = np.linalg.norm(self.pos - obstacle_point)

        # Ignore calculation if wall is far anyways
        if max_dist is not None and d > max_dist:
            return np.zeros(2)

        # Compute a unit vector towards the obstacle
        # TODO: fix divide by zero
        n = (self.pos - obstacle_point) / d
        # Compute a normal of n, which is therefore tangential to the obstacle
        t = np.flip(n) * np.array([-1, 1])

        # eq 7 in baseline
        obt_force = (Human.obs_strength * np.exp((self.radius - d)/Human.obs_range) + Human.bfc * theta(self.radius - d)) * n \
            - Human.sfc * theta(self.radius - d) * np.dot(self.velocity, t) * t

        # TODO: Check if energy should be added back in
        # crashing_force = Human.bfc * theta_val * n_ib_val - Human.sfc_wall * theta_val * (self.velocity * tib_val) * tib_val
        # crashing_strength = np.linalg.norm(crashing_force)
        # deduction_param = 0.000005
        # energy_lost = theta_val * ( crashing_strength / self.mass ) * deduction_param
        # self.energy -= energy_lost
        # self.energy = np.clip(self.energy,0,1)
        # print(f'crashed with the walls! : energy lost {energy_lost}')

        return obt_force

    def comfortness(self):
        """Compute the comfortness of agent by the time he escape"""
        # formula is given but it is optional for now
        raise NotImplementedError

    def step(self):
        """
        Compute all forces acting on this agent, update its velocity and move.
        """
        desired_dir = self.desired_dir()
        panic_index = self.panic_index(desired_dir)

        # Compute acceleration term of agent
        f_acc = self.acceleration_term(desired_dir, panic_index)

        # TODO: Is this still necessary?
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

        # Handle the repulsive effects from other people
        f_soc = np.array([0., 0.])
        
        neighbours = self.model.space.get_neighbors(self.pos, self.vision, False)
        for other in neighbours:
            distance = np.linalg.norm(self.pos - other.pos)
            # Compute repulsive effect from other people
            f_soc += self.people_repulsive_effect(other, d=distance) / self.mass

            # Follow the leader effect
            if other.is_leader:
                f_soc += self.leader_attractive_effect(other, d=distance) / self.mass

            # Crash effect
            f_soc += self.crash_effect(other, d=distance) / self.mass
# Type I
        # Handle the repulsive effects from obstacles
        f_obs = np.array([0.,0.])
        for obstacle in self.model.obstacles:
            f_obs += self.boundary_effect(obstacle, max_dist=2) / self.mass

        # for exit in self.model.exits:
        #     if np.linalg.norm(self.pos - exit.get_center()) < self.vision:
        #         self.velocity += self.leader_attractive_effect(exit) / self.mass

        # Compute random noise force
        f_noise = self.panic_noise_effect(panic_index)
        self.velocity += (f_acc + f_soc + f_obs + f_noise) * self.model.timestep

        # Update the movement, position features of the agent
        self.speed = np.clip(np.linalg.norm(self.velocity), 0, self.max_speed)
        # so speed is impacked by the remainly energy of individual, the minimum speed is applied to ensured badly injured agent still move out
        # uncommented line 343 - 350 and comment below line if we want energy = 0 agent to be dead and become an obstacle
        # self.speed = np.clip(self.speed * self.energy, self.min_speed, self.max_speed)
        self.velocity /= np.linalg.norm(self.velocity)
        self.velocity *= self.speed
        new_pos = self.pos + (self.velocity * self.model.timestep)

        # if out of bounds, put at bound
        if new_pos[0] > self.model.space.width:
            new_pos[0] = self.model.space.width - 0.00001
        elif new_pos[0] < 0:
            new_pos[0] = 0 + 0.00001

        if new_pos[1] > self.model.space.height:
            new_pos[1] = self.model.space.height - 0.00001
        elif new_pos[1] < 0:
            new_pos[1] = 0 + 0.00001
        self.model.space.move_agent(self, new_pos)

        self.timestep += 1

        # Remove the agent from the model if it has reached an exit
        for exit in self.model.exits:
            if exit.in_exit(self.pos, self.radius):
                self.model.exit_times.append(self.timestep*self.model.timestep)
                self.model.remove_agent(self)
                break
        
        # if self.unique_id == 0:
        #     plt.quiver(*self.pos, f_acc[0], f_acc[1], color=['r'], label='acc')
        #     plt.quiver(*self.pos, f_soc[0], f_soc[1], color=['b'], label='social')
        #     plt.quiver(*self.pos, f_obs[0], f_obs[1], color=['g'], label='obstable')
        #     plt.ylim(0,20)
        #     plt.xlim(0,20)
        #     plt.xlabel('width')
        #     plt.ylabel('height')
        #     plt.legend()
        #     plt.show()
        #     print("Social f:", f_soc)
        #     print("Acceleration f:", f_acc)
        #     print("Obstacle f:", f_obs)