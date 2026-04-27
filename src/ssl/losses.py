import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.07):
    B = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2*B, dtype=bool, device=z.device)
    sim = sim.masked_fill(mask, float('-inf'))
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)