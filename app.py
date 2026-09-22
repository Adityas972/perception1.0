"""Gradio demo: upload a video clip, get a per-frame anomaly score curve and
a gradient-weighted attention-rollout saliency overlay on the most anomalous
window — i.e. which patches specifically pushed *this* window's score up,
not just what the transformer attends to in general (see
interpret/grad_rollout.py).

Scorers are pre-trained by train.py on UCSD Ped1 (uploaded clips should be
CCTV-style pedestrian-walkway footage for the scores to be meaningful — this
is a research-dataset demo, not a general-purpose anomaly detector).
"""
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from interpret.grad_rollout import explain
from interpret.visualize import overlay_saliency
from model.classifier import AnomalyScorer
from model.feature_extractor import extract_embedding, patch_grid_shape
from utils.video_io import load_video_file, sliding_windows

CHECKPOINTS_DIR = Path("checkpoints")
NUM_FRAMES = 16
STRIDE = 4

_scorers = {}


def get_scorer(ped: str) -> AnomalyScorer:
    if ped not in _scorers:
        _scorers[ped] = AnomalyScorer.load(CHECKPOINTS_DIR / f"{ped}_scorer.joblib")
    return _scorers[ped]


def analyze(video_path: str, scene: str):
    frames = load_video_file(video_path)
    if len(frames) < NUM_FRAMES:
        raise gr.Error(f"Clip too short: need at least {NUM_FRAMES} frames, got {len(frames)}")

    scorer = get_scorer(scene)
    windows = sliding_windows(frames, num_frames=NUM_FRAMES, stride=STRIDE)

    # cheap frozen forward pass (no backward) to score every window and find
    # the peak — only the peak window gets the more expensive gradient pass.
    window_scores = [scorer.score(extract_embedding(w)[None, :])[0] for w in windows]

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(window_scores, marker="o", color="#d62728")
    ax.set_xlabel("window index")
    ax.set_ylabel("anomaly score")
    ax.set_title(f"Anomaly score per {NUM_FRAMES}-frame window (stride {STRIDE})")
    fig.tight_layout()

    peak_idx = int(np.argmax(window_scores))
    grid_shape = patch_grid_shape(num_frames=NUM_FRAMES)
    peak_score, saliency = explain(windows[peak_idx], scorer, grid_shape)
    overlaid = overlay_saliency(windows[peak_idx], saliency)

    return fig, overlaid, f"Peak anomaly score: {peak_score:.3f} (window {peak_idx})"


with gr.Blocks(title="Explainable Video Anomaly Detection") as demo:
    gr.Markdown(
        "# Explainable Video Anomaly Detection\n"
        "Frozen VideoMAE embeddings + a one-class SVM trained only on *normal* "
        "footage (UCSD Ped1/Ped2). Upload a clip to get a per-window anomaly "
        "score and a gradient-weighted attention-rollout saliency overlay "
        "showing which patches specifically drove the score on the most "
        "anomalous window."
    )
    with gr.Row():
        video_in = gr.Video(label="Upload a clip")
        scene_dropdown = gr.Dropdown(["UCSDped1", "UCSDped2"], value="UCSDped1", label="Trained scene")
    run_btn = gr.Button("Analyze", variant="primary")
    score_plot = gr.Plot(label="Anomaly score over time")
    saliency_gallery = gr.Gallery(label="Saliency overlay (most anomalous window)", columns=4)
    peak_label = gr.Textbox(label="Result", interactive=False)

    run_btn.click(analyze, inputs=[video_in, scene_dropdown], outputs=[score_plot, saliency_gallery, peak_label])


if __name__ == "__main__":
    demo.launch()
