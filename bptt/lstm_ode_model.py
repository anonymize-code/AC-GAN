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


class LSTMODE(nn.Module):
    """
    LSTM
    """

    def __init__(self, dim_x, dim_z, time_step, obs_model=None):
        super(LSTMODE, self).__init__()
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
        ode_func_dim = dim_z * 4
        # ode_func_dim = 100
        ode_func_net = create_net(dim_z * 2, dim_z * 2,
                                  n_layers=n,
                                  n_units=ode_func_dim,
                                  nonlinear=nn.Tanh)
        init_network_weights(ode_func_net)

        rec_ode_func = ode_func.ODEFunc(ode_func_net=ode_func_net)

        self.ode_solver = DiffeqSolver(rec_ode_func, "rk4", odeint_rtol=1e-3, odeint_atol=1e-4)

        # 已有解码器
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, decoder_units),
        #     nn.Tanh(),
        #     nn.Linear(decoder_units, input_dim * 2))
        #
        # utils.init_network_weights(self.decoder)

        # (10,1,100)
        # 替换为latent_step
        # self.gru_unit = GRU_Unit(latent_dim, input_dim, n_units=decoder_units)

        # self.latent_dim = latent_dim

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
            c = tc.zeros((1, b, self.d_z))
            h = z.reshape((1, b, self.d_z))
            inp = tc.zeros(b, 1, self.d_x)
        else:
            z = z0
            c = tc.zeros((1, b, self.d_z))
            h = z.reshape((1, b, self.d_z))
            inp = tc.zeros(b, 1, self.d_x)

        zs = tc.zeros(size=(b, T, self.d_z))

        times = tc.linspace(0., T, T * self.time_step)
        zt = 0
        prev_h = h
        prev_c = c
        # for t in range(0, int(T / n)):
        #     time_points = times[t * n * self.time_step:(t + 1) * n * self.time_step]
        #     prev_ode = torch.cat((prev_h, prev_c), dim=2)
        #     ode_sol = self.ode_solver(prev_ode, time_points)[0]
        #     # hidden_ode = ode_sol[:, -1]
        #     ode_h = ode_sol[:, :, :self.d_z]
        #     ode_c = ode_sol[:, :, self.d_z:]
        #     for i in range(1, n):
        #         output = self.latent_step(inp, (ode_h[:, i * 5, :].unsqueeze(0), self.norm(ode_c[:, i * 5, :].unsqueeze(0))))[0]
        #         zs[:, zt] = output.squeeze(1)
        #         zt += 1
        #     z = (tc.pinverse(B) @ x[:, t].float().T).T  # (*) copy this out
        #     c = tc.zeros((1, b, self.d_z))  # (*) c.detach()
        #     h = z.reshape((1, b, self.d_z))  # (*)  h.detach()
        #     output, (h, c) = self.latent_step(inp, (h, self.norm(c)))
        #     zs[:, zt] = output.squeeze(1)
        #     zt += 1
        #     prev_h = h
        #     prev_c = c

        for t in range(0, T):

            # interleave observation every n time steps
            # for truncated BPTT copy in (*)
            time_points = times[t * self.time_step:(t + 1) * self.time_step]
            prev_ode = torch.cat((prev_h, prev_c), dim=2)
            ode_sol = self.ode_solver(prev_ode, time_points)[0]
            hidden_ode = ode_sol[:, -1]
            h = hidden_ode[:, :self.d_z].unsqueeze(0)
            c = hidden_ode[:, self.d_z:].unsqueeze(0)
            if t % n == 0 and t != 0:
                z = (tc.pinverse(B) @ x[:, t].float().T).T  # (*) copy this out
                c = tc.zeros((1, b, self.d_z))  # (*) c.detach()
                h = z.reshape((1, b, self.d_z))  # (*)  h.detach()
            output, (prev_h, prev_c) = self.latent_step(inp, (h, self.norm(c)))
            zs[:, t] = output.squeeze(1)

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
            if (B.dtype == torch.float64):
                z = (tc.pinverse(B) @ x_.T).T
            else:
                z = (tc.pinverse(B) @ x_.float().T).T
            c = tc.zeros((1, b, self.d_z))
            h = z.reshape((1, b, self.d_z))
            inp = tc.zeros(b, 1, self.d_x)
        else:
            z = z0
            c = tc.zeros((1, b, self.d_z))
            h = z.reshape((1, b, self.d_z))
            inp = tc.zeros(b, 1, self.d_x)

        times = tc.linspace(0., T, T * self.time_step)
        prev_h = h
        prev_c = c
        if prev_h.device != torch.device('cpu'):
            prev_c = prev_c.cuda()
            inp = inp.cuda()
            times = times.cuda()
        Z.append(z.unsqueeze(1))
        for t in range(T - 1):
            time_points = times[t * self.time_step:(t + 1) * self.time_step]
            prev_ode = torch.cat((prev_h, prev_c), dim=2)
            ode_sol = self.ode_solver(prev_ode, time_points)[0]
            hidden_ode = ode_sol[:, -1]
            h = hidden_ode[:, :self.d_z].unsqueeze(0)
            c = hidden_ode[:, self.d_z:].unsqueeze(0)
            output, (prev_h, prev_c) = self.latent_step(inp, (h, self.norm(c)))
            Z.append(output)

        Z = tc.stack(Z, dim=1)
        shape = (n_repeat * T, self.d_z)
        Z = tc.reshape(Z, shape)
        return Z

    def batch_generate(self, T, data, z0=None, n_repeat=1):
        '''creates freely generated (unforced) trajectories'''

        Z = []
        batch, len = data.size()
        b = n_repeat
        z_ = torch.zeros((batch, 1, self.d_z))
        c = tc.zeros((1, batch, self.d_z)).cuda()
        h = tc.zeros((1, batch, self.d_z)).cuda()
        inp = tc.zeros(batch, 1, self.d_x).cuda()
        B = self.obs_model.weight  # .detach()
        for i in range(batch):
            x_ = data[i, 0].reshape(1, 1)
            if B.dtype == torch.float64:
                z = (tc.pinverse(B) @ x_.T).T
            else:
                z = (tc.pinverse(B) @ x_.float().T).T
            z_[i] = z
            h[0, i] = z.reshape((b, self.d_z))

        times = tc.linspace(0., T, T * self.time_step)
        prev_h = h
        prev_c = c
        if prev_h.device != torch.device('cpu'):
            prev_c = prev_c.cuda()
            inp = inp.cuda()
            times = times.cuda()
            z_ = z_.cuda()
        Z.append(z_)
        for t in range(T - 1):
            time_points = times[t * self.time_step:(t + 1) * self.time_step]
            prev_ode = torch.cat((prev_h, prev_c), dim=2)
            # prev_ode1 = torch.cat((prev_h, prev_c), dim=2)
            # prev_ode = torch.cat((prev_ode, prev_ode1), dim=1)
            ode_sol = self.ode_solver(prev_ode, time_points)
            hidden_ode = ode_sol[:, :, -1]
            h = hidden_ode[:, :, :self.d_z].contiguous()
            c = hidden_ode[:, :, self.d_z:].contiguous()
            output, (prev_h, prev_c) = self.latent_step(inp, (h, self.norm(c)))
            Z.append(output)

        Z = tc.stack(Z, dim=1)
        shape = (batch*T, self.d_z)
        Z_out = tc.reshape(Z, shape)
        return Z_out,self.d_z
