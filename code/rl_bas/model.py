import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

def weights_init(vec):
    if isinstance(vec, nn.Linear):
        torch.nn.init.xavier_uniform_(vec.weight, gain=1)
        torch.nn.init.constant_(vec.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, num_input, num_hidden):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_input, num_hidden)
        self.linear2 = nn.Linear(num_hidden, num_hidden)
        self.linear3 = nn.Linear(num_hidden, 1)
        
        self.apply(weights_init)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, num_hidden):
        super(QNetwork, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, num_hidden).to(self.device)
        self.linear2 = nn.Linear(num_hidden, num_hidden).to(self.device)
        self.linear3 = nn.Linear(num_hidden, 1).to(self.device)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, num_hidden).to(self.device)
        self.linear5 = nn.Linear(num_hidden, num_hidden).to(self.device)
        self.linear6 = nn.Linear(num_hidden, 1).to(self.device)

        self.apply(weights_init)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1).to(self.device)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2
    
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(num_inputs, hidden_dim).to(self.device)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim).to(self.device)

        self.mean_linear = nn.Linear(hidden_dim, num_actions).to(self.device)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions).to(self.device)
        self.epsilon = 1e-6
        self.apply(weights_init)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.).to(self.device)
            self.action_bias = torch.tensor(0.).to(self.device)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).to(self.device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Normal

# LOG_SIG_MAX = 2
# LOG_SIG_MIN = -20

# def weights_init(vec):
#     if isinstance(vec, nn.Linear):
#         torch.nn.init.xavier_uniform_(vec.weight, gain=1)
#         torch.nn.init.constant_(vec.bias, 0)

# class ValueNetwork(nn.Module):
#     def __init__(self, num_input, num_hidden):
#         super(ValueNetwork, self).__init__()
        
#         self.linear1 = nn.Linear(num_input, num_hidden)
#         self.linear2 = nn.Linear(num_hidden, num_hidden)
#         self.linear3 = nn.Linear(num_hidden, 1)
        
#         self.apply(weights_init)
        
#     def forward(self, state):
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#         x = self.linear3(x)
#         return x
    
# class QNetwork(nn.Module):
#     def __init__(self, num_inputs, num_actions, num_hidden):
#         super(QNetwork, self).__init__()
        
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         # Q1 architecture
#         self.linear1 = nn.Linear(num_inputs + num_actions, num_hidden).to(self.device)
#         self.linear2 = nn.Linear(num_hidden, num_hidden).to(self.device)
#         self.linear3 = nn.Linear(num_hidden, 1).to(self.device)

#         # Q2 architecture
#         self.linear4 = nn.Linear(num_inputs + num_actions, num_hidden).to(self.device)
#         self.linear5 = nn.Linear(num_hidden, num_hidden).to(self.device)
#         self.linear6 = nn.Linear(num_hidden, 1).to(self.device)

#         self.apply(weights_init)

#     def forward(self, state, action):
#         xu = torch.cat([state, action], 1).to(self.device)
        
#         x1 = F.relu(self.linear1(xu))
#         x1 = F.relu(self.linear2(x1))
#         x1 = self.linear3(x1)

#         x2 = F.relu(self.linear4(xu))
#         x2 = F.relu(self.linear5(x2))
#         x2 = self.linear6(x2)

#         return x1, x2
    
# class GaussianPolicy(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
#         super(GaussianPolicy, self).__init__()
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.linear1 = nn.Linear(num_inputs, hidden_dim).to(self.device)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim).to(self.device)

#         self.mean_linear = nn.Linear(hidden_dim, num_actions).to(self.device)
#         self.log_std_linear = nn.Linear(hidden_dim, num_actions).to(self.device)
#         self.epsilon = 1e-6
#         self.apply(weights_init)

#         # action rescaling
#         if action_space is None:
#             self.action_scale = torch.tensor(1.).to(self.device)
#             self.action_bias = torch.tensor(0.).to(self.device)
#         else:
#             self.action_scale = torch.FloatTensor(
#                 (action_space.high - action_space.low) / 2.).to(self.device)
#             self.action_bias = torch.FloatTensor(
#                 (action_space.high + action_space.low) / 2.).to(self.device)

#     def forward(self, state):
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#         mean = self.mean_linear(x)
#         log_std = self.log_std_linear(x)
#         log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
#         return mean, log_std

#     def sample(self, state):
#         mean, log_std = self.forward(state)
#         std = log_std.exp()
#         normal = Normal(mean, std)
#         x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         y_t = torch.tanh(x_t)
#         action = y_t * self.action_scale + self.action_bias
#         log_prob = normal.log_prob(x_t)
#         # Enforcing Action Bound
#         log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
#         log_prob = log_prob.sum(1, keepdim=True)
#         mean = torch.tanh(mean) * self.action_scale + self.action_bias
#         return action, log_prob, mean

#     def to(self, device):
#         self.action_scale = self.action_scale.to(device)
#         self.action_bias = self.action_bias.to(device)
#         return super(GaussianPolicy, self).to(device)

        