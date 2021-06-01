import torch.nn as nn
import torch

x = torch.tensor([21, 3, 21, 21., 3., 23.], dtype=torch.float32)
print(torch.softmax(x, dim=0))
x = x - 1000
print(torch.softmax(x, dim=0))

