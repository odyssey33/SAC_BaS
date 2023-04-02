import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_bas.model import GaussianPolicy, QNetwork
from rcbf_sac.utils import hard_update, soft_update, to_tensor
import numpy as np 

# reference : https://spinningup.openai.com/en/latest/algorithms/sac.html
class Agent(nn.Module):
    def __init__(self, gamma, tau, alpha, action_space, learning_rate, num_inputs, num_hidden, target_update_interval = 1):
        super(Agent, self).__init__()

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.critic = QNetwork(num_inputs, action_space.shape[0], num_hidden)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], num_hidden).to(self.device)
        # copy & paste the target parameter (left) to critic (right)
        hard_update(self.critic_target, self.critic) 
        
        ## stochastic policy > gaussian policy update
        ## activate auto-grad (backpropagation)
        # return the products of all elements (tensor)
        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        
        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], num_hidden, action_space)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
    def generate_action(self, state, model):
        state = to_tensor(state, torch.FloatTensor, self.device)
        expand_dim = len(state.shape) == 1
        if expand_dim:
            state = state.unsqueeze(0) # dim (#, ) >> (1, #)  
        action, _, _ = self.policy.sample(state)
        final_action = action
        if expand_dim:
            return final_action.detach().cpu().numpy()[0]
        return final_action.detach().cpu().numpy()
    
    def update_parameters(self, memory, batch_size, updates, model, memory_model=None, ratio=None):
        
        if memory_model is not None and ratio < 1.0:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, t_batch, next_t_batch, cbf_info_batch, next_cbf_info_batch = memory.sample(
                batch_size=int(ratio * batch_size))
            state_batch_m, action_batch_m, reward_batch_m, next_state_batch_m, mask_batch_m, t_batch_m, next_t_batch_m, cbf_info_batch_m, next_cbf_info_batch_m = memory_model.sample(
                batch_size=int((1 - ratio) * batch_size))
            # mix state/action/reward from both regular RL and model-based RL memory
            state_batch = np.vstack((state_batch, state_batch_m))
            action_batch = np.vstack((action_batch, action_batch_m))
            reward_batch = np.hstack((reward_batch, reward_batch_m))
            next_state_batch = np.vstack((next_state_batch, next_state_batch_m))
            mask_batch = np.hstack((mask_batch, mask_batch_m))
        else:
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch, t_batch, next_t_batch, _, _ = memory.sample(batch_size=batch_size)
            
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch, action_batch)  
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf1_loss = F.mse_loss(qf1, next_q_value)  
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Compute Actions and log probabilities
        pi, log_pi, _ = self.policy.sample(state_batch)
        # Compute safe action using Differentiable CBF-QP
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() ]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
            
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone()  
        
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
    def save_model(self, output):
        print('Saving models in {}'.format(output))
        torch.save(
            self.policy.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
        if self.compensator:
            self.compensator.save_model(output)
    
    def load_weights(self, output):
        if output is None: return
        print('Loading models from {}'.format(output))

        self.policy.load_state_dict(
            torch.load('{}/actor.pkl'.format(output), map_location=self.device)
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output), map_location=self.device)
        )

        if self.compensator:
            self.compensator.load_weights(output)
