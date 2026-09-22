"""Extract frozen VideoMAE embeddings for the (all-normal) Train split of
each UCSD Ped scene and fit a one-class anomaly scorer on top.

Ped1 and Ped2 are different camera scenes with different normal-motion
statistics, so we train a separate scorer per scene rather than pooling them.
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data.ucsd import list_clips
from model.classifier import fit_scorer
from model.feature_extractor import extract_embedding
from utils.video_io import load_frame_dir, sliding_windows

DATA_ROOT = Path("data/raw/UCSD_Anomaly_Dataset.v1p2")
FEATURES_DIR = Path("data/features")
CHECKPOINTS_DIR = Path("checkpoints")


def extract_train_embeddings(ped: str, num_frames: int = 16, stride: int = 8) -> np.ndarray:
    cache_path = FEATURES_DIR / f"{ped}_train_embeddings.npy"
    if cache_path.exists():
        print(f"[{ped}] using cached embeddings at {cache_path}")
        return np.load(cache_path)

    clips = list_clips(DATA_ROOT / ped, "Train")
    embeddings = []
    for clip_dir in tqdm(clips, desc=f"{ped} train clips"):
        frames = load_frame_dir(clip_dir)
        windows = sliding_windows(frames, num_frames=num_frames, stride=stride)
        for window in windows:
            embeddings.append(extract_embedding(window))

    embeddings = np.stack(embeddings)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peds", nargs="+", default=["UCSDped1", "UCSDped2"])
    parser.add_argument("--nu", type=float, default=0.1)
    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    for ped in args.peds:
        print(f"=== {ped} ===")
        embeddings = extract_train_embeddings(ped)
        print(f"[{ped}] {embeddings.shape[0]} training windows, dim {embeddings.shape[1]}")
        scorer = fit_scorer(embeddings, nu=args.nu)
        out_path = CHECKPOINTS_DIR / f"{ped}_scorer.joblib"
        scorer.save(out_path)
        print(f"[{ped}] saved scorer to {out_path}")


if __name__ == "__main__":
    main()
