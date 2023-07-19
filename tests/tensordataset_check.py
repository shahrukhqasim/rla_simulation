import torch
from torch.utils.data import TensorDataset
import numpy as np


my_randoms = np.random.normal(0, 1, (20000, 3, 3))
dataset = TensorDataset(torch.Tensor(my_randoms))

print(dataset)