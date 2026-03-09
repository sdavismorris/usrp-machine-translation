import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class TranslationRankingDataset(Dataset):
    def __init__(self, csv_file: str, label_column: str, metric_columns: str):
        """
        A PyTorch Dataset class for loading translation ranking data from a CSV file.
        Args:
            csv_file: Path to the CSV file containing the data.
            label_column: Name of the column in the CSV that contains the labels ('correct' or 'incorrect').
            metric_columns: List of column names that contain the metric scores (e.g., ['bleu-score', 'chrf-score', 'ter-score', 'bertscore', 'bleurt', 'comet']).
        """
        self.df = pd.read_csv(csv_file)
        correct_df = self.df[self.df[label_column] == 'correct'].reset_index(drop=True)
        incorrect_df = self.df[self.df[label_column] == 'incorrect'].reset_index(drop=True)

        # Convert the dataframe to a PyTorch tensor for the correct and incorrect translations
        # Each tensor is a matrix of shape (num_samples, num_metrics), where each row corresponds to a translation and each column corresponds to a metric score.
        self.X_correct = torch.tensor(correct_df[metric_columns].values, dtype=torch.float32)
        self.X_incorrect = torch.tensor(incorrect_df[metric_columns].values, dtype=torch.float32)

        self.normalize()

    def normalize(self):
        """
        Normalize the metric scores using z-score normalization
        This centers the data to have a mean of 0 and scales it to have a standard deviation of 1.
        We normalize the dataset to account for each metric having different scales and ranges
        (e.g. 'bleu-score' could be between 0 and 100, while 'bertscore' could be between 0 and 1).
        """
        all_metrics = torch.cat((self.X_correct, self.X_incorrect), dim=0)
        # Calculate the mean and standard deviation for each metric across all translations (both correct and incorrect)
        # e.g. mean is a vector with num_metrics elements,
        # where each element is the mean of that metric across all translations.
        mean = all_metrics.mean(dim=0)
        std = all_metrics.std(dim=0)

        self.X_correct = (self.X_correct - mean) / std
        self.X_incorrect = (self.X_incorrect - mean) / std

    def __len__(self):
        # The length of the dataset is the number of correct translations (= number of incorrect translations)
        return len(self.X_correct)

    def __getitem__(self, idx):
        # Return a pair of correct and incorrect translations for the given index
        return self.X_correct[idx], self.X_incorrect[idx]
