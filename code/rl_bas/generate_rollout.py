import numpy as np
from copy import deepcopy
from rcbf_sac.utils import euler_to_mat_2d

def generate_model_rollout(env, memory_model, memory, agent, model, Qbas, horizon, batch_size=20):
    obs_batch, action_batch, reward_batch, next_obs_batch, mask_batch, t_batch, next_t_batch, _, _ = memory.sample(batch_size=batch_size)
    obs_batch_ = deepcopy(obs_batch)
    done_batch_ = [False for _ in range(batch_size)]
    t_batch_ = deepcopy(t_batch)
    if env.mode == 'BaS':
        Q = np.diag([1.0, 1.0, 0.0, Qbas])
    else: # BF
        Q = np.diag([1.0, 1.0, 0.0])
    R = np.diag([0.001, 0.001])
    
    for k in range(horizon):
        batch_size_ = obs_batch_.shape[0]
        #obs_batch_ = obs_batch_[:, :3]
        action_batch_ = agent.generate_action(obs_batch_, model)
        state_batch_ = model.get_state(obs_batch_)
        next_state_mu_, next_state_std_, next_t_batch_ = model.predict_next_state(state_batch_, action_batch_, t_batch_)
        next_state_batch_ = np.random.normal(next_state_mu_, next_state_std_)
        next_obs_batch_ = model.get_obs(next_state_batch_)
        
        # dubin's vehicle
        # terminal cost
        # Q = np.diag([1, 1, 0, 0.1]) 
        # R = np.diag([0.001, 0.001])
        prev_cost = -np.log(obs_batch_[:, -2])
        # next_obs_batch_[:, 4] -> bas_vec
        
        if env.mode == 'BaS':
            prev_cost = -np.log(obs_batch_[:, -2])
            error = env.unwrapped.desired_pos - next_obs_batch_[:, [0, 1, 2, 4]]
            bas_vec = (next_obs_batch_[:, -1].reshape((next_obs_batch[:, -1].shape[0], 1)))
            errorQ = np.sum(np.multiply(error @ Q, error), axis = 1)
            errorR = np.sum(np.multiply(action_batch_ @ R, action_batch_), axis = 1)#np.sqrt(np.sum(np.multiply(action_batch_ @ R, action_batch_), axis = 1))
            cost = np.sqrt(errorQ + errorR) #np.linalg.norm(error, axis=1) #.5*error @ Q @ error
        else: # BF
            prev_cost = -np.log(obs_batch_[:, -2])
            error = env.unwrapped.desired_pos - next_obs_batch_[:, [0, 1, 2]]
            bfVal = next_obs_batch_[:, -1]
            errorQ = np.sum(np.multiply(error @ Q, error), axis = 1)
            errorR = np.sum(np.multiply(action_batch_ @ R, action_batch_), axis = 1)#np.sqrt(np.sum(np.multiply(action_batch_ @ R, action_batch_), axis = 1))
            cost = np.sqrt(errorQ + errorR  + bfVal ** 2).reshape(len(errorQ), 1)
            bfVal = bfVal.reshape((next_obs_batch[:, -1].shape[0], 1))
            cost = cost.reshape(-1)
            
        # generate compass
        compass = np.matmul(np.expand_dims(error[:, :2], 1), euler_to_mat_2d(next_state_batch_[:, 2])).squeeze(1)
        compass /= np.sqrt(np.sum(np.square(compass), axis=1, keepdims=True)) + 0.001
        
        if env.mode == 'BaS':
            next_obs_batch_ = np.hstack((next_obs_batch_[:, :-1], compass, np.expand_dims(np.exp(-cost), axis=-1), bas_vec))
        else:
            next_obs_batch_ = np.hstack((next_obs_batch_[:, :-1], compass, np.expand_dims(np.exp(-cost), axis=-1), bfVal))
        # reward
        goal_size = 0.1
        prev_cost_norm = np.linalg.norm(prev_cost)
        reward_batch_ = prev_cost / (np.finfo(float).eps + prev_cost_norm) - cost
        prev_cost = cost
        reward_target = 1
        reward_distance = 1
        # done
        reached_goal = cost <= goal_size
        reached_goal = reached_goal
        reward_batch_ += reward_target * reached_goal
        
        done_batch_ = reached_goal
        mask_batch_ = np.invert(done_batch_)
        memory_model.batch_push(obs_batch_, action_batch_, reward_batch_, next_obs_batch_, mask_batch_,
                                t_batch_, next_t_batch_)
        
        t_batch_ = deepcopy(next_t_batch_)
        obs_batch_ = deepcopy(next_obs_batch_)
        
        # delete done trajectory
        if np.sum(done_batch_) > 0:
            obs_batch_ = np.delete(obs_batch_, done_batch_ > 0, axis=0)
        
    return memory_model