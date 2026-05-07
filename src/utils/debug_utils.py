import torch

def grad_norm(model):

    total_norm = torch.norm(torch.stack([
        p.grad.detach().norm(2) for p in model.parameters()
        if p.grad is not None
    ]), 2).item()

    return total_norm

