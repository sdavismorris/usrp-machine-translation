import torch
import torch.nn as nn
import numpy as np

class LinearRankingModel(nn.Module):
    """
    A linear model for ranking translations.

    score = w1 * bleu + w2 * chrf + w3 * ter + w4 * bertscore + w5 * bleurt + w6 * comet
    """
    def __init__(self, n_metrics=6):
        super(LinearRankingModel, self).__init__()
        # Initialize weights for each metric
        self.weights = nn.Parameter(torch.ones(n_metrics) / n_metrics)
        with torch.no_grad():
            self.weights.fill_(1.0 / n_metrics) # Start with equal weights for all metrics

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the score for a batch of translations.

        Args:
            features: Tensor of shape (batch_size, 6) containing the metric scores for each translation.

        Returns:
            Tensor of shape (batch_size,) containing the computed scores.
        """
        # Matrix multiplication which gives us a vector of shape (batch_size,)
        # where each element is the weighted sum of the metrics for one translation.
        return torch.matmul(features, self.weights)


# If you want to play around, here's a neural network model!
# MLP stands for Multi-Layer Perceptron. They invent these names to sound cool.
class MLPRankingModel(nn.Module):
    """
    A neural model for ranking translations.
    Our default architecture is a 2-layer network with ReLU activation.
    The advantage MLPs is in capturing non-linear relationships between the metrics and the translation quality.
    """
    def __init__(self, n_metrics=6, hidden_size=16):
        super(MLPRankingModel, self).__init__()
        self.fc1 = nn.Linear(n_metrics, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the score for a batch of translations.

        Args:
            features: Tensor of shape (batch_size, 6) containing the metric scores for each translation.

        Returns:
            Tensor of shape (batch_size,) containing the computed scores.
        """
        x = self.fc1(features)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze()
