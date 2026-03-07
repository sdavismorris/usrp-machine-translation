import os
import pandas as pd
import evaluate
import torch
from bert_score import score as bert_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def evaluate_bertscore(references: list, hypotheses: list) -> list[float]:
    P, R, F1 = bert_score(
        hypotheses,
        references,
        model_type="microsoft/deberta-xlarge-mnli",
        device="cuda",
        batch_size=32
    )
    return F1.tolist()


def evaluate_bleurt(references: list, hypotheses: list) -> list[float]:
    metric = evaluate.load("bleurt", "BLEURT-20")
    results = metric.compute(
        predictions=hypotheses,
        references=references,
        batch_size=16
    )
    return results["scores"]


def evaluate_comet(references: list, hypotheses: list) -> list[float]:
    metric = evaluate.load("comet")
    results = metric.compute(
        predictions=hypotheses,
        references=references,
        model_name_or_path="Unbabel/wmt22-comet-da",
        batch_size=8,
        gpus=1
    )
    return results["scores"]


df = pd.read_csv("metric_data.csv")
references = df["reference"].tolist()
hypotheses = df["translation"].tolist()

bertscore_eval = evaluate_bertscore(references, hypotheses)
bleurt_eval = evaluate_bleurt(references, hypotheses)
comet_eval = evaluate_comet(references, hypotheses)

df["bertscore"] = bertscore_eval
df["bleurt"] = bleurt_eval
df["comet"] = comet_eval

df.to_csv("metric_data_2.csv", index=False)