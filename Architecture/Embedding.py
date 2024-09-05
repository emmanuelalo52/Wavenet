import torch
class Embedding:
  
    def __init__(self, num_embd, embd_dim):
        self.weight = torch.randn((num_embd, embd_dim))
    
    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out
  
    def parameters(self):
        return [self.weight]