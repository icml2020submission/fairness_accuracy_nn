import torch

disable_cuda = False
torch_device = None
if not disable_cuda and torch.cuda.is_available():
    TORCH_DEVICE = torch.device('cuda')
else:
    TORCH_DEVICE = torch.device('cpu')
