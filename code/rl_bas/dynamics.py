import numpy as np
import torch
from copy import deepcopy
from rl_bas.bas_dynamics import BaSDynamics

class Model:
    def __init__(self, env, obstacle_info):
        self.env = env
        
        self.fx, self.fu = self.get_dyn()
        self.num_state = 3 + 1#self.env.observation_space.shape[0]
        self.num_action = self.env.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.x0 = env.initial_state
        self.xd = env.desired_pos
                
        self.safety_function = obstacle_info.safety_function
        self.bas_state = BaSDynamics(self.safety_function, self.env.initial_state, self.env.desired_pos, gamma=1.0, barrier_type='log_barrier', n_bas=1)
        
        
    def predict_next_state(self, state_batch, u_batch, t_batch=None):
        expand_dims = len(state_batch.shape) == 1
        if expand_dims:
            state_batch = np.expand_dims(state_batch, axis=0)
        
        # next_state_batch = state_batch + self.env.dt * (self.fx(state_batch, t_batch) + (self.fu(state_batch, t_batch) @ np.expand_dims(u_batch, -1)).squeeze(-1))
        # control affine system
        # next_state_batch = state_batch + self.env.dt * (self.get_dyn_safe(self.gamma, state_batch, u_batch))
        
        if self.env.mode == 'BaS':
            next_state_batch = deepcopy(state_batch)
            next_state_batch[:, :3] = state_batch[:, :3] + self.env.dt * (self.fx(state_batch[:, :3]) + (self.fu(state_batch[:, :3]) @ np.expand_dims(u_batch, -1)).squeeze(-1))
            next_state_batch[:, 3] = (self.bas_state.discrete_bas_dyn(state_batch, next_state_batch, 4)[0])#state_batch[:, 3] + self.env.dt * (self.bas_state.discrete_bas_dyn(state_batch, next_state_batch, 4)[0])
        else: # BF
            next_state_batch = deepcopy(state_batch)
            next_state_batch[:, :3] = state_batch[:, :3] + self.env.dt * (self.fx(state_batch[:, :3]) + (self.fu(state_batch[:, :3]) @ np.expand_dims(u_batch, -1)).squeeze(-1))
        

        pred_std = np.zeros(state_batch.shape)
        if expand_dims:
            next_state_batch = next_state_batch.squeeze(0)
            if pred_std is not None:
                pred_std = pred_std.squeeze(0)
                
        if t_batch is not None:
            next_t_batch = t_batch + self.env.dt
            return next_state_batch, self.env.dt * pred_std, next_t_batch
        return next_state_batch, self.env.dt * pred_std, t_batch
    
    def angle_normalize(self, ang):
        """Normalize angle between -pi and pi."""
        return ((ang+np.pi) % (2*np.pi)) - np.pi
      
    def predict_next_obs(self, x, u):
        next_x, _, _ = self.predict_next_state(x, u)
        next_obs = self.get_obs(next_x)
        return next_obs
    
        
    def get_dyn(self):
        # dynamics xdot = [u1*cos(th); u1*sin(th), u2] = f(x) + g(x)u 
        # f(x) = zeros, g(x) = jacobian(xdot, u) = [cos(th) 0; sin(th) 0; 0 1]
        
        def fx(state_batch, t_batch=None):    
            # x1 = self.state[0]
            # x2 = self.state[1]
            # x3 = self.state[2]
            return np.zeros(state_batch.shape)
        
        def fu(state_batch):
            theta = state_batch[:, 2]
            fu_mat = np.zeros((state_batch.shape[0], 3, 2))
            fu_mat[:, 0, 0] = np.cos(theta)
            fu_mat[:, 1, 0] = np.sin(theta)
            fu_mat[:, 2, 1] = 1.0
            return fu_mat
        return fx, fu
        
    def get_state(self, obs):
        
        expand_dims = len(obs.shape) == 1
        is_tensor = torch.is_tensor(obs)
        
        if is_tensor:
            dtype = obs.dtype
            device = obs.device
            obs = self.to_numpy(obs)
            
        if expand_dims:
            obs = np.expand_dims(obs, 0)
        
        if self.env.mode == 'BaS':
            theta = np.arctan2(obs[:, 3], obs[:, 2]) # atan2(sin(x2)/cos(x2))
            state_batch = np.zeros((obs.shape[0], 4))
            state_batch[:, 0] = obs[:, 0]
            state_batch[:, 1] = obs[:, 1]
            state_batch[:, 2] = theta #obs[2, :]
            state_batch[:, 3] = obs[:, -1]
        else: # BF
            theta = np.arctan2(obs[:, 3], obs[:, 2]) # atan2(sin(x2)/cos(x2))
            state_batch = np.zeros((obs.shape[0], 3))
            state_batch[:, 0] = obs[:, 0]
            state_batch[:, 1] = obs[:, 1]
            state_batch[:, 2] = theta #obs[2, :]                  
        
        return self.to_tensor(state_batch, dtype, device) if is_tensor else state_batch
    
    def get_obs(self, state_batch):
        if self.env.mode == 'BaS':
            obs = np.zeros((state_batch.shape[0], 5))
            obs[:, 0] = state_batch[:, 0]
            obs[:, 1] = state_batch[:, 1]
            obs[:, 2] = np.cos(state_batch[:, 2])    
            obs[:, 3] = np.sin(state_batch[:, 2])
            obs[:, 4] = state_batch[:, -1] 
        else: # BF
            obs = np.zeros((state_batch.shape[0], 5))
            obs[:, 0] = state_batch[:, 0]
            obs[:, 1] = state_batch[:, 1]
            obs[:, 2] = np.cos(state_batch[:, 2])    
            obs[:, 3] = np.sin(state_batch[:, 2])
            for i in range(len(state_batch)):
                obs[i, 4] = self.bas_state.log_barrier(state_batch[i])[0]  #log_barrier inverse_barrier
        return obs
    
    def seed(self, s):
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)

    def to_numpy(x):
        # convert torch tensor to numpy array
        return x.cpu().detach().double().numpy()

    def to_tensor(x, dtype, device, requires_grad=False):
        # convert numpy array to torch tensor
        if type(x).__module__ != 'numpy':
            return x
        return torch.from_numpy(x).type(dtype).to(device).requires_grad_(requires_grad)