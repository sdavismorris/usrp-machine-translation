from sklearn.metrics import roc_auc_score
import pandas as pd
from scipy.stats import ttest_rel

def evaluate_metrics(data: pd.DataFrame, label_column: str, metrics: list[str]) -> dict[str, float]:
    """
    Evaluate each MT metric and meta-metric on:
    - the pairwise ranking accuracy (proportion of times the metric correctly ranks the correct translation higher than the incorrect translation)
    - the AUC-ROC score (measures the ability of the metric to distinguish between correct and incorrect translations across all possible thresholds)
    - mean margin (average difference in metric scores between correct and incorrect translations)
    - p-value from a paired t-test (statistical significance of the difference in scores between correct and incorrect translations)

    Args:
        data: A pandas DataFrame containing the test dataset with columns for labels ('correct' or 'incorrect') and metric scores.
        label_column: The name of the column in the DataFrame that contains the labels.
        metrics: A list of column names in the DataFrame that correspond to the MT metrics and meta-metric to be evaluated.

    Returns:
        A dictionary where keys are metric names and values are dictionaries containing the evaluation results for each metric.
    """
    results = {}
    for metric in metrics:
        correct_scores = data[data[label_column] == "correct"][metric].to_numpy()
        incorrect_scores = data[data[label_column] == "incorrect"][metric].to_numpy()

        # Pairwise ranking accuracy
        pairwise_accuracy = (correct_scores > incorrect_scores).mean()

        # AUC-ROC score
        auc_roc = roc_auc_score(data["label"].apply(lambda x: 1 if x == "correct" else 0), data[metric])

        # Mean margin
        mean_margin = (correct_scores - incorrect_scores).mean()

        # Paired t-test p-value
        t_stat, p_value = ttest_rel(correct_scores, incorrect_scores)

        results[metric] = {
            "pairwise_accuracy": pairwise_accuracy,
            "auc_roc": auc_roc,
            "mean_margin": mean_margin,
            "p_value": p_value
        }

    return results
