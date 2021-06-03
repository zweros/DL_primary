import torch

'''
    采用 gpu 加速
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()
print(use_gpu)

# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)
