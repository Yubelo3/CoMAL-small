import torch


a=torch.randn(2,3,4)
a=a.sum(dim=[0,1])
print(a.shape)