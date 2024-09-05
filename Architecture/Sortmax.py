import torch
class Softmax:
    def __call__(self,x):
        x = torch.max(x, dim=-1, keepdim=True)
        z = torch.exp(x)
        sum_z = z.sum()
        self.out = z/sum_z
        return self.out