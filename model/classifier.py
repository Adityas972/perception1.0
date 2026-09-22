"""Lightweight anomaly scorer trained on frozen VideoMAE embeddings.

UCSD Ped's Train split contains only normal frames, so this is a one-class
problem: we never see anomalies during training. A one-class SVM fits a
boundary around the normal embeddings; anomaly score = distance outside
that boundary (higher = more anomalous, i.e. the negative of sklearn's
`decision_function`, which is positive inside the boundary).
"""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


@dataclass
class AnomalyScorer:
    scaler: StandardScaler
    ocsvm: OneClassSVM

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """embeddings: (N, 768). Returns (N,) anomaly scores, higher = more anomalous."""
        x = self.scaler.transform(embeddings)
        return -self.ocsvm.decision_function(x)

    def save(self, path: Path):
        joblib.dump({"scaler": self.scaler, "ocsvm": self.ocsvm}, path)

    @classmethod
    def load(cls, path: Path) -> "AnomalyScorer":
        d = joblib.load(path)
        return cls(scaler=d["scaler"], ocsvm=d["ocsvm"])


def fit_scorer(embeddings: np.ndarray, nu: float = 0.1) -> AnomalyScorer:
    """nu: expected fraction of the *training* set treated as outliers — acts
    as a margin softness since even 'normal' training clips have some noisy
    frames (e.g. someone briefly entering/leaving frame)."""
    scaler = StandardScaler().fit(embeddings)
    x = scaler.transform(embeddings)
    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma="scale").fit(x)
    return AnomalyScorer(scaler=scaler, ocsvm=ocsvm)
