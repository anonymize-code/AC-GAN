import torch
import torch.nn as nn
import torch as tc
import math
from bptt import ode_func
from bptt.diffeq_solver import DiffeqSolver


def create_net(n_inputs, n_outputs, n_layers=1, n_units=100, nonlinear=nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


class NODE(nn.Module):
    """
    LSTM
    """

    def __init__(self, dim_x, dim_z, time_step, obs_model=None):
        super(NODE, self).__init__()
        self.d_x = dim_x
        self.d_z = dim_z
        self.time_step = time_step
        n = 1
        self.latent_step = nn.LSTM(input_size=self.d_x, hidden_size=self.d_z, num_layers=n, batch_first=True)
        self.obs_model = obs_model
        layer_norm = 0
        if layer_norm == 1:
            self.norm = nn.LayerNorm(self.d_z, elementwise_affine=False)
        else:
            self.norm = nn.Identity()
        ode_func_dim = 100
        ode_func_net = create_net(dim_z, dim_z,
                                  n_layers=n,
                                  n_units=ode_func_dim,
                                  nonlinear=nn.Tanh)
        init_network_weights(ode_func_net)

        rec_ode_func = ode_func.ODEFunc(ode_func_net=ode_func_net)

        self.ode_solver = DiffeqSolver(rec_ode_func, "euler", odeint_rtol=1e-3, odeint_atol=1e-4)

        self.sigma_fn = nn.Softplus()

    def forward(self, x, n, z0=None):
        '''creates forced trajectories'''
        # b batch size
        # T len
        # dx x_dim
        b, T, dx = x.size()
        B = self.obs_model.weight  # .detach()

        # no forcing if n is not specified
        if n is None:
            n = T + 1
        if z0 is None:
            z = (tc.pinverse(B) @ x[:, 0].float().T).T
        else:
            z = z0

        zs = tc.zeros(size=(b, T, self.d_z))
        # 每次ode后推n个时间点的数据
        times = tc.linspace(0., T, T * self.time_step)
        zt = 0
        prev_z = z.reshape((1, b, self.d_z))
        for t in range(0, int(T / n)):
            time_points = times[t * n * self.time_step:(t + 1) * n * self.time_step]
            ode_sol = self.ode_solver(prev_z, time_points)[0]
            for i in range(n):
                zs[:, zt] = ode_sol[:, (i + 1) * self.time_step - 1, :]
                zt += 1
            z = (tc.pinverse(B) @ x[:, t].float().T).T  # (*) copy this out
            prev_z = z.reshape((1, b, self.d_z))
        if zt < T:
            time_points_extra = times[int(T / n) * n * self.time_step:(int(T / n) + 1) * n * self.time_step]
            ode_sol = self.ode_solver(prev_z, time_points_extra)[0]
            for i in range(n):
                zs[:, zt] = ode_sol[:, (i + 1) * self.time_step - 1, :]
                zt += 1
                if zt == T:
                    break
        return zs

    def generate(self, T, data, z0=None, n_repeat=1):
        '''creates freely generated (unforced) trajectories'''

        Z = []
        len, dx = data.size()
        b = n_repeat
        step = int(len / n_repeat)
        x_ = data[::step]

        B = self.obs_model.weight  # .detach()
        # no interleaving obs. if n is not specified
        if z0 is None:
            # z = (tc.pinverse(B) @ x_.T).T
            z = (tc.pinverse(B) @ x_.float().T).T
        else:
            z = z0

        times = tc.linspace(0., T, T * self.time_step)
        prev_z = z.reshape((1, b, self.d_z))
        ode_sol = self.ode_solver(prev_z, times)[0]
        Z.append(z.unsqueeze(1))
        for t in range(T - 1):
            output = ode_sol[:, (t + 1) * self.time_step - 1, :]
            Z.append(output.unsqueeze(1))

        Z = tc.stack(Z, dim=1)
        shape = (n_repeat * T, self.d_z)
        Z = tc.reshape(Z, shape)
        return Z
