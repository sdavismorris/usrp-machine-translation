import pandas as pd
import evaluate
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def evaluate_bertscore(references: list, hypotheses: list) -> list[float]:
    metric = evaluate.load("bertscore")
    results = metric.compute(
        predictions=hypotheses,
        references=references,
        model_type="xlm-roberta-large",
        num_layers=12,
        batch_size=16,
        device=device
    )
    return results["f1"]


def evaluate_bleurt(references: list, hypotheses: list) -> list[float]:
    metric = evaluate.load("bleurt", "BLEURT-20")
    results = metric.compute(
        predictions=hypotheses,
        references=references,
    )
    return results["scores"]


def evaluate_comet(sources: list, references: list, hypotheses: list) -> list[float]:
    metric = evaluate.load("comet")
    results = metric.compute(
        sources=sources,
        predictions=hypotheses,
        references=references,
    )
    return results["scores"]


df = pd.read_csv("metric_data.csv")
references = df["reference"].tolist()
hypotheses = df["translation"].tolist()
sources = df["source"].tolist()

bertscore_eval = evaluate_bertscore(references, hypotheses)
bleurt_eval = evaluate_bleurt(references, hypotheses)
comet_eval = evaluate_comet(sources, references, hypotheses)

df["bertscore"] = bertscore_eval
df["bleurt"] = bleurt_eval
df["comet"] = comet_eval

df.to_csv("metric_data_2.csv", index=False)
