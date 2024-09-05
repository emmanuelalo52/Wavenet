import torch
#Create linear
class Linear:
    def __init__(self,fan_in,fan_out,bias=True):
        self.weight = torch.randn(fan_in, fan_out)/ fan_out * 0.5
        self.bias = torch.zeros(fan_out) if bias else None
    def __call__(self,x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])