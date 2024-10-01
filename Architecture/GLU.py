import torch
class GLU:
    def __init__(self,chan_in,chan_out,bias=True):
        self.weight = torch.randn(chan_in,chan_out)/(chan_in)*0.5
        self.bias = torch.zeros(chan_out) if bias else None
    def __call__(self,x):
        assert x.ndim >=2
        linear = x @ self.weight
        if self.bias is not None:
            linear += self.bias
        linear_out, gate = torch.chunk(linear, 2, dim=-1)
        gate = torch.sigmoid(gate)
        self.out = torch.tanh(linear_out) * gate
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])