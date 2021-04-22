import torch
import torch.nn as nn
from torch.distributions import Normal

def mlp(input_size, hidden_sizes=(64, 64), activation="tanh"):
    if activation == "tanh":
        activation = nn.Tanh()
    elif activation == "relu":
        activation = nn.ReLU()
    elif activation == "sigmoid":
        activation = nn.Sigmoid()
    elif activation == "leaky_relu":
        activation = nn.LeakyReLU(0.001)
    else:
        raise NotImplementedError(f"Activation {activation} is not supported")
    
    layers = []
    sizes = (input_size, ) + hidden_sizes
    for i in range(len(hidden_sizes)):
        layers += [nn.Linear(sizes[i], sizes[i+1]), activation]
    return nn.Sequential(*layers)

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), activation="tanh", device="cpu", normalization=None):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        
        self.mlp_net = mlp(obs_dim, hidden_sizes, activation)
        self.mean_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.logstd_layer = nn.Linear(hidden_sizes[-1], act_dim)
        
        if normalization is not None:
            self.norm_mean, self.norm_std, self.act_mean = map(lambda t: torch.FloatTensor(t).to(self.device), normalization)
        else:
            self.norm_mean, self.norm_std, self.act_mean = torch.zeros(1, obs_dim), torch.ones(1, obs_dim), torch.zeros(1, act_dim)
        
#         self.mean_layer.weight.data.mul_(0.1)
#         self.mean_layer.bias.data.mul_(0.0)
        self.to(device)
    
    def forward(self, obs):
        obs = (obs - self.norm_mean) / self.norm_std
        out = self.mlp_net(obs)
        mean = self.mean_layer(out)
        mean += self.act_mean
        if len(mean.size()) == 1:
            mean = mean.view(1, -1)
        logstd = self.logstd_layer(out)
        std = torch.exp(logstd)
        return mean, logstd, std
    
    def get_act(self, obs, deterministic=False):
        mean, _, std = self.forward(obs)
        if deterministic:
            return mean
        else:
            z = torch.normal(torch.zeros_like(mean), torch.ones_like(mean))
            return mean + z * std
    
    # Agent interface
    def select_action(self, obs, eval=False):
        x = torch.FloatTensor(obs).to(self.device)
        out = self.get_act(x)
        action = out.cpu().detach().numpy()
        return action[0]
    
    def log_prob(self, obs, act):
        mean, _, std = self.forward(obs)
        normal = Normal(mean, std)
        return normal.log_prob(act).sum(-1, keepdim=True), mean, std