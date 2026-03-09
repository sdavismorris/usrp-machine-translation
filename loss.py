import torch
import torch.nn as nn

class PairwiseRankingLoss(nn.Module):
    """
    Pairwise Ranking Loss for machine translation evaluation.

    For each pair of translations, if:
        score_correct > score_incorrect + margin,
    then the loss is 0, otherwise the loss is:
        loss = margin - (score_correct - score_incorrect)
    """
    def __init__(self, margin=0.1):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, scores_correct: torch.Tensor, scores_incorrect: torch.Tensor) -> torch.Tensor:
        """
        Compute the pairwise ranking loss.

        Args:
            scores_correct: Tensor of shape (batch_size,) containing scores for correct translations.
            scores_incorrect: Tensor of shape (batch_size,) containing scores for incorrect translations.

        Returns:
            A scalar tensor representing the loss.
        """
        # Ensure the inputs are of the same shape
        if scores_correct.shape != scores_incorrect.shape:
            raise ValueError("Input tensors must have the same shape")

        # margins represents the difference between the scores of correct and incorrect translations
        margins = scores_correct - scores_incorrect
        # Calculate the loss for each pair
        losses = torch.clamp(self.margin - margins, min=0)
        # Return the average loss over the batch
        return losses.mean()


class L2RegularizationLoss(nn.Module):
    """
    L2 Regularization Loss to penalize large weights in the model.

    The idea is to encourage the model to learn smaller (lower numerical value) weights to improve generalization.
    """
    def __init__(self, lambda_reg=0.01):
        super(L2RegularizationLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute the L2 regularization loss for the given model.

        Args:
            model: The neural network model whose parameters will be regularized.

        Returns:
            A scalar tensor representing the L2 regularization loss.
        """
        l2_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for param in model.parameters():
            l2_loss += torch.norm(param, p=2) ** 2
        return self.lambda_reg * l2_loss
