import torch
import torch.nn as nn


class GRU_Unit(nn.Module):
    def __init__(self, latent_dim, input_dim, n_units=100):
        super(GRU_Unit, self).__init__()

        # (20+1,100) tanh (100,10) sigmoid
        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.update_gate)

        # (20+1,100) tanh (100,10) sigmoid
        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.reset_gate)

        # (20+1,100) tanh (100,10) sigmoid
        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim * 2))
        init_network_weights(self.new_state_net)

    def forward(self, y_mean, y_std, x, mask):
        y_concat = torch.cat([y_mean, y_std, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)

        new_state, new_state_std = torch.chunk(self.new_state_net(concat), chunks=2, dim=-1)
        new_state_std = new_state_std.abs()

        output_y = (1 - update_gate) * new_state + update_gate * y_mean
        new_y_std = (1 - update_gate) * new_state_std + update_gate * y_std

        mask = (torch.sum(mask, -1, keepdim=True) > 0).float()

        new_y = mask * output_y + (1 - mask) * y_mean
        new_y_std = mask * new_y_std + (1 - mask) * y_std

        new_y_std = new_y_std.abs()
        return output_y, new_y, new_y_std


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)
