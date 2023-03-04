import math
import torch
from torch import nn
from torch.nn import init

class RWKVCellV4(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.w_r = nn.Parameter(torch.empty((hidden_size, input_size)))
        self.w_k = nn.Parameter(torch.empty((hidden_size, input_size)))
        self.w_v = nn.Parameter(torch.empty((hidden_size, input_size)))
        self.w_o = nn.Parameter(torch.empty((input_size, input_size)))
        self.w_u = nn.Parameter(torch.empty((hidden_size)))
        self.w_v = nn.Parameter(torch.empty((hidden_size)))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            if weight.ndim == 2:
                init.uniform_(weight, -stdv, stdv)
            else:
                init.ones_(weight)

    def forward(self, inputs, frac_n, frac_d, scale):
        """
        args:
            inputs: (batch_size, input size).
            frac_n: (batch_size, hidden_size), numerator of softmax, must be a zeros at first step.
            frac_d: (batch_size, hidden_size), denominator of softmax, must be a zeros at first step.
            scale: (batch_size, hidden_size), scale value passing through time for overflow currection,
                must be a minimum value at first step, like -1e20.
        """
        # t indices time step.
        r_t = torch.sigmoid(torch.matmul(inputs, self.w_r.transpose(-1, -2))) # (batch_size, hidden_size)
        k_t = torch.matmul(inputs, self.w_k.transpose(-1, -2)) # (batch_size, hidden_size)
        v_t = torch.matmul(inputs, self.w_v.transpose(-1, -2)) # (batch_size, hidden_size)

        softmax_scale = torch.maximum(scale, self.w_u + k_t) # input scale for softmax to fix overflow
        cell_multiplier = torch.exp(scale - softmax_scale)
        attn_multiplier = torch.exp(self.w_u + k_t - softmax_scale)
        frac_n_t = cell_multiplier * frac_n + attn_multiplier * v_t
        frac_d_t = cell_multiplier * frac_d + attn_multiplier

        next_scale = torch.maximum(scale + self.w_w, k_t) # input scale for passing cell to fix overflow
        next_cell_multiplier = torch.exp(scale + self.w_w - next_scale)
        next_attn_multiplier = torch.exp(k_t - next_scale)
        next_frac_n = next_cell_multiplier * frac_n + next_attn_multiplier * v_t
        next_frac_d = next_cell_multiplier * frac_d + next_attn_multiplier

        output = torch.matmul(r_t * (frac_n_t / frac_d_t), self.w_o.transpose(-1, -2))

        return output, next_frac_n, next_frac_d, next_scale
