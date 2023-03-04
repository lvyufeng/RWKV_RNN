import math
import torch
from torch import nn
from torch.nn import init

class RWKVLayer(nn.Cell):
    def __init__(self, input_size, hidden_size):
        super().__init__(False)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_r = nn.Parameter(torch.empty((hidden_size, input_size)))
        self.w_k = nn.Parameter(torch.empty((hidden_size, input_size)))
        self.w_v = nn.Parameter(torch.empty((hidden_size, input_size)))
        self.w_u = nn.Parameter(torch.empty((hidden_size)))
        self.w_w = nn.Parameter(torch.empty((hidden_size)))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            if weight.ndim == 2:
                init.uniform_(weight, -stdv, stdv)
            else:
                init.ones_(weight)

    def forward(self, inputs, frac_n=None, frac_d=None, scale=None):
        seq_length, batch_size, _ = inputs.size()

        if frac_n is None:
            frac_n = torch.zeros((batch_size, self.hidden_size))
        if frac_d is None:
            frac_d = torch.zeros((batch_size, self.hidden_size))
        if scale is None:
            scale = torch.full((batch_size, self.hidden_size), -1e20)

        r = torch.sigmoid(torch.matmul(inputs, self.w_r, transpose_b=True)) # (seq_length, batch_size, hidden_size)
        k = torch.matmul(inputs, self.w_k, transpose_b=True) # (seq_length, batch_size, hidden_size)
        v = torch.matmul(inputs, self.w_v, transpose_b=True) # (seq_length, batch_size, hidden_size)

        outputs = []
        for t in range(seq_length):
            k_t = k[t]
            v_t = v[t]
            r_t = r[t]
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
            y = r_t * (frac_n_t / frac_d_t)

            outputs.append(y)

            scale = next_scale
            frac_n = next_frac_n
            frac_d = next_frac_d

        output = torch.concat(outputs)
        return output, y.unsqueeze(0)

class BiRWKV(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, bidirectional=False):
        super().__init__()
        self.fw = RWKVLayer(input_size, hidden_size)
        self.bidirecitonal = bidirectional
        self.batch_first = batch_first
        if bidirectional:
            self.bw = RWKVLayer(input_size, hidden_size)
        else:
            self.bw = None
    
    def forward(self, inputs, frac_n=None, frac_d=None, scale=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1)
        output_fw, hidden_fw = self.fw(inputs, frac_n, frac_d, scale)
        if self.bidirecitonal:
            inputs_fw = ops.reverse(inputs, (0,))
            output_bw, hidden_bw = self.bw(inputs_fw, frac_n, frac_d, scale)
            output_bw = ops.reverse(output_bw, (0,))
            output = ops.concat((output_fw, output_bw), 2)
            hidden = ops.concat((hidden_fw, hidden_bw), 0)
        else:
            output = output_fw
            hidden = hidden_fw.unsqueeze(0)
        if self.batch_first:
            output = output.swapaxes(0, 1)
        return output, hidden