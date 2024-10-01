import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDivergenceLoss(nn.Module):
    """KL Divergence Loss with optional l2 norm regularization for model weights."""
    def __init__(self, l2_reg_weight: float | None = 0.01, entropy_reg_weight: float | None = 0.0):
        """
        Initialize the class.

        Args:
            l2_reg_weight (float): Weight of the l2 regularization term in whole loss function. Default to `0.01`.
            entropy_reg_weight (float): Weight of the inverse entropy loss tern in the whole loss function. Default to `0.0`.
        """
        super(KLDivergenceLoss, self).__init__()
        self.l2_reg_weight = l2_reg_weight
        self.entropy_reg_weight = entropy_reg_weight

    def forward(self, outputs: torch.Tensor, targets:torch.Tensor, model_parameters):
        n_probs = torch.tensor(outputs.shape[-1], dtype=outputs.dtype, device=outputs.device)
        # KL divergence loss
        kl_loss = F.kl_div(torch.log(outputs + 1e-8), targets, reduction='batchmean')

        # L2 regularization
        l2_reg = torch.tensor(0., device=outputs.device)
        for param in model_parameters:
            l2_reg += torch.norm(param)

        # Entropy regularization
        entropy_reg = torch.tensor(0., device=outputs.device)
        if self.entropy_reg_weight > 0:
            entropy = -torch.sum(outputs * torch.log(outputs + 1e-8), dim=-1).mean()
            entropy_reg = torch.log(n_probs) - entropy

        # Combine all terms
        total_loss = kl_loss + self.l2_reg_weight*l2_reg + self.entropy_reg_weight*entropy_reg

        return total_loss
