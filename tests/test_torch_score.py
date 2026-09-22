"""Verifies AnomalyScorer.torch_score matches the sklearn-based .score()
exactly. This has to hold bit-for-bit (within float precision) before the
gradient-weighted rollout can be trusted, since a subtly wrong decision
function would silently produce a plausible-looking but meaningless heatmap.
"""
import numpy as np
import torch

from model.classifier import AnomalyScorer

CHECKPOINTS = ["checkpoints/UCSDped1_scorer.joblib", "checkpoints/UCSDped2_scorer.joblib"]


def test_torch_score_matches_sklearn():
    rng = np.random.default_rng(0)
    for path in CHECKPOINTS:
        scorer = AnomalyScorer.load(path)
        embeddings = rng.normal(size=(20, 768)).astype(np.float64)

        sklearn_scores = scorer.score(embeddings)
        torch_scores = np.array([
            scorer.torch_score(torch.tensor(e[None, :], dtype=torch.float64), torch.device("cpu")).item()
            for e in embeddings
        ])
        max_diff = np.abs(sklearn_scores - torch_scores).max()
        assert max_diff < 1e-6, f"{path}: max diff {max_diff} between sklearn and torch scores"
        print(f"{path}: OK, max diff {max_diff:.2e}")


if __name__ == "__main__":
    test_torch_score_matches_sklearn()
