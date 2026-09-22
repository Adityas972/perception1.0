"""Evaluate anomaly scorers on the Test split: frame-level ROC-AUC against
the dataset's ground-truth abnormal-frame ranges.

A window's score is assigned to every frame it covers; a frame covered by
several overlapping windows gets the mean of their scores. This is the
standard frame-level protocol used in the UCSD Ped anomaly detection
literature (see Mahadevan et al., CVPR 2010).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from data.ucsd import frame_labels, list_clips, parse_gt_ranges
from model.classifier import AnomalyScorer
from model.feature_extractor import extract_embedding
from utils.video_io import load_frame_dir

DATA_ROOT = Path("data/raw/UCSD_Anomaly_Dataset.v1p2")
CHECKPOINTS_DIR = Path("checkpoints")
ASSETS_DIR = Path("assets")


def frame_scores_for_clip(frames: list, scorer: AnomalyScorer, num_frames: int = 16, stride: int = 4) -> np.ndarray:
    """Slide a window across the clip and average overlapping window scores
    back onto each frame. Smaller stride than training (4 vs 8) trades more
    compute for finer temporal localization at eval time."""
    n = len(frames)
    sums = np.zeros(n)
    counts = np.zeros(n)
    starts = range(0, max(n - num_frames + 1, 1), stride)
    for start in starts:
        end = min(start + num_frames, n)
        window = frames[start:end]
        if len(window) < num_frames:
            window = frames[-num_frames:]
            start = n - num_frames
            end = n
        score = scorer.score(extract_embedding(window)[None, :])[0]
        sums[start:end] += score
        counts[start:end] += 1
    counts[counts == 0] = 1
    return sums / counts


def evaluate_ped(ped: str) -> dict:
    scorer = AnomalyScorer.load(CHECKPOINTS_DIR / f"{ped}_scorer.joblib")
    clips = list_clips(DATA_ROOT / ped, "Test")
    gt_ranges = parse_gt_ranges(DATA_ROOT / ped)
    assert len(clips) == len(gt_ranges), f"{ped}: {len(clips)} clips vs {len(gt_ranges)} gt entries"

    all_scores, all_labels = [], []
    per_clip = []
    for clip_dir, ranges in tqdm(list(zip(clips, gt_ranges)), desc=f"{ped} test clips"):
        frames = load_frame_dir(clip_dir)
        scores = frame_scores_for_clip(frames, scorer)
        labels = frame_labels(len(frames), ranges)
        all_scores.append(scores)
        all_labels.append(labels)
        clip_auc = roc_auc_score(labels, scores) if labels.any() and not labels.all() else float("nan")
        per_clip.append((clip_dir.name, clip_auc))

    overall_auc = roc_auc_score(np.concatenate(all_labels), np.concatenate(all_scores))
    return {
        "ped": ped,
        "overall_auc": overall_auc,
        "per_clip": per_clip,
        "scores": all_scores,
        "labels": all_labels,
        "clip_names": [c.name for c in clips],
    }


def plot_clip(ped: str, clip_name: str, scores: np.ndarray, labels: np.ndarray):
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(scores, color="#d62728", label="anomaly score")
    ax.fill_between(range(len(labels)), 0, scores.max() if scores.max() > 0 else 1,
                     where=labels, color="gray", alpha=0.3, label="ground-truth anomaly")
    ax.set_title(f"{ped} / {clip_name}")
    ax.set_xlabel("frame")
    ax.set_ylabel("anomaly score")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / f"{ped}_{clip_name}_score.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peds", nargs="+", default=["UCSDped1", "UCSDped2"])
    parser.add_argument("--plot-worst", type=int, default=2, help="plot N lowest-AUC clips per scene")
    args = parser.parse_args()

    for ped in args.peds:
        result = evaluate_ped(ped)
        print(f"\n=== {ped}: overall frame-level AUC = {result['overall_auc']:.4f} ===")
        ranked = sorted(
            [(n, a) for n, a in result["per_clip"] if a == a],  # drop NaNs
            key=lambda x: x[1],
        )
        for name, auc in ranked[:5]:
            print(f"  worst: {name} AUC={auc:.4f}")
        for name, auc in ranked[-3:]:
            print(f"  best:  {name} AUC={auc:.4f}")

        for name, _ in ranked[:args.plot_worst]:
            idx = result["clip_names"].index(name)
            plot_clip(ped, name, result["scores"][idx], result["labels"][idx])
        print(f"  saved score plots for {min(args.plot_worst, len(ranked))} clips to {ASSETS_DIR}/")


if __name__ == "__main__":
    main()
