import torch

a = torch.randn((3, 3, 5))
b = torch.randn((3, 3, 5))
c = abs(a-b)
print(c)