import numpy as np
import gym
from rl_bas.bas_dynamics import BaSDynamics
from rl_bas.obstacle_2d import Obstacles2D
from gym import spaces
from gym.spaces import Box 

class myEnv(gym.Env):
    def __init__(self, init_x, target_x, obstacle_info, mode, Qbas, obstacle_update = False):
        # initial configuration
        super(myEnv, self).__init__()
        n_action = 2 # number of controls (u)
        if mode == 'BaS':
            n_obs = 7 + 1  # number of states (x) + compass + BaS
        else: # BF
            n_obs = 7 + 1  # number of states (x) + compass + BF
        self.action_space = Box(low=-1.0, high=1.0, shape=(n_action,))
        self.observation_space = Box(low=1e10, high=1e10, shape=(n_obs,))
        self.dt = 0.02
        self.max_episode_steps = 1000
        self.reward_goal = 1.0
        self.reward_collision = -1.0
        
        # Initialize Env
        self.state = None
        self.episode_step = 0
        self.initial_state = init_x
        self.desired_pos = target_x
        self.des_tolrn = 0.1
        
        # Get dynamics
        self.fx, self.fu = self.get_dynamics()
        
        # Load obstacles
        # self.obstacles = []
        # if obstacle_info.size == 0:
        #     self.get_obstacle_location(num_obs, range_loc, range_radius)
        # else:
        #     self.obstacles = obstacle_info
        self.safety_function = obstacle_info.safety_function
        self.obs_loc = obstacle_info.obstacles()
        self.obstacle_update = obstacle_update
        
        self.Qbas = Qbas        
        if mode == 'BaS':
            self.bas_state = BaSDynamics(self.safety_function, self.initial_state, self.desired_pos, gamma=1.0, barrier_type='log_barrier', n_bas=1)
            self.initial_state = np.append(self.initial_state, self.bas_state.bas_initial)
            self.desired_pos = np.append(self.desired_pos, self.bas_state.bas_terminal)
        else: # BF
            self.bas_state = BaSDynamics(self.safety_function, self.initial_state, self.desired_pos, gamma=1.0, barrier_type='log_barrier', n_bas=1)
        #self.init_cost = self.compute_cost(action=np.array([0, 0]))
        self.prev_cost_history = np.zeros((self.max_episode_steps,))
        self.prev_cost_norm = 1.0
        self.mode = mode
        
        self.reset()
        
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        state, reward, done, info = self.step_(action)
        return self.get_obs(), reward, done, info
    
    def angle_normalize(self, ang):
        """Normalize angle between -pi and pi."""
        return ((ang+np.pi) % (2*np.pi)) - np.pi
    
    def step_(self, action):
        # custom dynamics
        if self.mode == 'BaS':
            x_curr = self.state
            self.state[2] = self.angle_normalize(self.state[2])
            self.state[:3] += self.dt * (self.fx(self.state[:3]) + self.fu(self.state[:3]) @ action)
            self.state[3] = ((self.bas_state.discrete_bas_dyn(x_curr, self.state, 4)[0]))
        else: # BF
            x_curr = self.state
            self.state[2] = self.angle_normalize(self.state[2])
            self.state[:3] += self.dt * (self.fx(self.state[:3]) + self.fu(self.state[:3]) @ action)
            
        self.episode_step += 1
        info = dict()
        
        # Reward
        self.prev_cost_history = np.roll(self.prev_cost_history, -1, axis=0)
        cost = self.compute_cost(action)
        reward = self.prev_cost / (np.finfo(float).eps + self.prev_cost_norm) - cost
        self.prev_cost = cost
        self.prev_cost_history[-1] = self.prev_cost
        self.prev_cost_norm = np.linalg.norm(self.prev_cost_history)
        
        if self.success(action):
            info['convergence'] = True
            reward += self.reward_goal
            done = True
        else:
            done = self.episode_step >= self.max_episode_steps
        
        info['cost'] = self.compute_cost(action)
        
        # constraint cost in reward  
        return self.state, reward, done, info   
            
    def collision(self):
        dist = np.sqrt(np.sum(np.power(self.obs_loc[:, :2] - self.state[:2], 2), 1))
        return any(dist <= self.obs_loc[:, 2])
    
    def success(self, action):
        return self.compute_cost(action) <= self.des_tolrn
    
    def reset(self):
        self.episode_step = 0
        self.state = np.copy(self.initial_state)
        self.prev_cost = self.compute_cost(action=np.array([0, 0]))
        return self.get_obs(), dict()
    
    def get_obs(self):
        
        goal_compass = self.obs_compass()
        
        if self.mode == 'BaS':
            Q = np.diag([1.0, 1.0, 0.0, self.Qbas])
            #R = np.diag([.001, .001])
            dist = self.state - self.desired_pos
            curr_cost = np.sqrt(dist.T @ Q @ dist)
            x1 = self.state[0] # x position
            x2 = self.state[1] # y position
            x3 = self.state[2] # theta
            bas = self.state[3]
            return np.array([x1, x2, np.cos(x3), np.sin(x3), goal_compass[0], goal_compass[1], np.exp(-curr_cost), bas]) 
        else: # BF
            bfVal = float(self.bas_state.log_barrier(self.state)[0]) #log_barrier inverse_barrier
            Q = np.diag([1.0, 1.0, 0.0])
            #R = np.diag([.001, .001])
            dist = self.state - self.desired_pos
            curr_cost = np.sqrt(dist.T @ Q @ dist + bfVal**2)
            x1 = self.state[0] # x position
            x2 = self.state[1] # y position
            x3 = self.state[2] # theta
            return np.array([x1, x2, np.cos(x3), np.sin(x3), goal_compass[0], goal_compass[1], np.exp(-curr_cost), bfVal])
        
   
    def obs_compass(self):
        """
        Return a robot-centric compass observation of a list of positions.
        Compass is a normalized (unit-lenght) egocentric XY vector,
        from the agent to the object.
        """
        # Get ego vector in world frame
        vec = self.desired_pos[:2] - self.state[:2]
        # Rotate into frame
        R = np.array([[np.cos(self.state[2]), -np.sin(self.state[2])], [np.sin(self.state[2]), np.cos(self.state[2])]])
        vec = np.matmul(vec, R)
        # Normalize
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        return vec
    
    def get_dynamics(self):
        # dynamics xdot = [u1*cos(th); u1*sin(th), u2] = f(x) + g(x)u; control affine system
        # f(x) = zeros, g(x) = get_dynamics(xdot, u) = [cos(th) 0; sin(th) 0; 0 1]
        def fx(state):    
            # x1 = self.state[0]
            # x2 = self.state[1]
            # x3 = self.state[2]
            return np.zeros(state.shape)
        
        def fu(state):
            theta = state[2]
            return np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1.0]])
        return fx, fu

    def compute_cost(self, action):
        # running cost
        if self.mode == 'BaS':
            Q = np.diag([1.0, 1.0, 0.0, self.Qbas])
            R = np.diag([0.001, 0.001])
            dist = self.state - self.desired_pos
            cost = np.sqrt(dist.T @ Q @ dist + action.T @ R @ action)
        else: # BF
            Q = np.diag([1.0, 1.0, 0.0])
            R = np.diag([0.001, 0.001])
            bfVal = float(self.bas_state.log_barrier(self.state)[0]) #log_barrier  inverse_barrier
            dist = self.state - self.desired_pos
            cost = np.sqrt(dist.T @ Q @ dist + action.T @ R @ action + bfVal**2)  
        return cost
    
    def get_obstacle_location(self, num_obs: int, range_loc: float, range_radius: float):

        obs = np.zeros((num_obs, 3))
        gen_success = True
        n = 0
        while gen_success:
            centers = np.random.uniform(low=range_loc[0], high=range_loc[1], size=(1, 2))
            radius = np.random.uniform(low=range_radius[0], high=range_radius[1], size=(1, 1))
            no_conflict = False
            # generate the location and corresponding radius
            if n == 0:
                obs[0, :] = np.concatenate((centers[0, :], radius[0]))
                n += 1
                continue
            for k in range(n):
                if all(np.sum(obs[:, :2] - centers, 1) != 0):
                    diff = obs[k, :2]  - centers[0, :]
                    if n < num_obs+1:
                        if np.linalg.norm(diff) >= obs[k, 2] + radius[0]:
                            obs[n, :] = np.concatenate((centers[0, :], radius[0]))
                            n += 1
                        
                if n == num_obs:
                    gen_success = False
                    break                  
        self.obstacles = obs

    