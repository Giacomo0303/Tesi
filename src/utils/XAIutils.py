import torch

# RELATIVE L2 WEIGHT CHANGE
def RWC(initial_weight: torch.Tensor, final_weight: torch.Tensor, eps=1e-7):
    if initial_weight.shape != final_weight.shape:
        return None
    return torch.norm(input=final_weight - initial_weight, p=2) / (torch.norm(input=initial_weight, p=2) + eps)
