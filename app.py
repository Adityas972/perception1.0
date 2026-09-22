"""Gradio demo: upload a video clip, get a per-frame anomaly score curve and
an attention-rollout saliency overlay on the most anomalous window.

Scorers are pre-trained by train.py on UCSD Ped1 (uploaded clips should be
CCTV-style pedestrian-walkway footage for the scores to be meaningful — this
is a research-dataset demo, not a general-purpose anomaly detector).
"""
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from interpret.rollout import spatiotemporal_map
from interpret.visualize import overlay_saliency
from model.classifier import AnomalyScorer
from model.feature_extractor import extract_embedding_with_attention, patch_grid_shape
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

    window_scores = []
    window_attns = []
    for w in windows:
        emb, attn = extract_embedding_with_attention(w)
        window_scores.append(scorer.score(emb[None, :])[0])
        window_attns.append(attn)

    # score curve
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(window_scores, marker="o", color="#d62728")
    ax.set_xlabel("window index")
    ax.set_ylabel("anomaly score")
    ax.set_title(f"Anomaly score per {NUM_FRAMES}-frame window (stride {STRIDE})")
    fig.tight_layout()

    # saliency overlay on the most anomalous window
    peak_idx = int(np.argmax(window_scores))
    peak_window = windows[peak_idx]
    grid_shape = patch_grid_shape(num_frames=NUM_FRAMES)
    saliency = spatiotemporal_map(window_attns[peak_idx], grid_shape)[0].numpy()
    overlaid = overlay_saliency(peak_window, saliency)

    return fig, overlaid, f"Peak anomaly score: {window_scores[peak_idx]:.3f} (window {peak_idx})"


with gr.Blocks(title="Explainable Video Anomaly Detection") as demo:
    gr.Markdown(
        "# Explainable Video Anomaly Detection\n"
        "Frozen VideoMAE embeddings + a one-class SVM trained only on *normal* "
        "footage (UCSD Ped1/Ped2). Upload a clip to get a per-window anomaly "
        "score and a temporal attention-rollout saliency overlay on the most "
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
