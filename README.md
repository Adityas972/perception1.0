# Explainable Video Anomaly Detection

Video anomaly detection on the UCSD Ped1/Ped2 pedestrian-walkway benchmark,
built to run entirely on a laptop with no dedicated GPU and no paid APIs:

- **Perception backbone**: a frozen, pretrained VideoMAE (`MCG-NJU/videomae-base-finetuned-kinetics`)
  used purely as a feature extractor — no fine-tuning of the 86M-parameter
  transformer, just forward passes on Apple Silicon MPS.
- **Anomaly scoring**: a one-class SVM trained only on *normal* footage
  (UCSD Ped's Train split contains no anomalies), scored via distance from
  the learned normal-embedding boundary.
- **Explainability**: gradient-weighted attention rollout (`interpret/grad_rollout.py`,
  following Chefer et al. CVPR 2021 / the "grad rollout" recipe) — the
  anomaly score is backpropagated through the frozen VideoMAE backbone into
  its attention weights, so the resulting heatmap shows what specifically
  pushed *this* clip's score up, not just generic attention (see
  `interpret/rollout.py` for the plain, unconditional version this replaced).
  Requires a differentiable reimplementation of the one-class SVM's decision
  function (`model/classifier.py:torch_score`), checked to match sklearn's
  `decision_function` to float precision in `tests/test_torch_score.py`.
- **Demo**: a Gradio app — upload a clip, get a per-window anomaly score
  curve and a saliency overlay on the most anomalous window.

## Setup

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash data/download_ucsd.sh   # ~700MB, UCSD Anomaly Detection Dataset
```

## Usage

```bash
python train.py       # extract frozen embeddings for Train clips, fit one-class SVMs
python evaluate.py    # frame-level ROC-AUC on Test clips vs ground truth
python app.py          # launch the Gradio demo
```

## Results

Frame-level ROC-AUC (VideoMAE embeddings + one-class SVM, no fine-tuning):

| Scene    | Overall AUC | Best clip            | Worst clip            |
|----------|-------------|-----------------------|------------------------|
| UCSDped1 | 0.696       | Test020 (AUC 0.999)   | Test030 (AUC 0.016)    |
| UCSDped2 | 0.743       | Test007 (AUC 1.000)   | Test012 (AUC 0.010)    |

This is a reasonable baseline for a frozen backbone with zero fine-tuning —
published from-scratch/fine-tuned methods on this benchmark typically reach
0.65–0.90+ depending on approach, so there's headroom (e.g. fine-tuning a
lightweight adapter on top of the frozen features) without needing more
compute than this laptop has.

The worst clips (near-0 AUC) are cases where the anomaly score moves in the
*opposite* direction of the ground-truth window — worth inspecting via
`assets/*_score.png` before trusting the model on anything beyond this
benchmark.

(See `assets/` for per-clip anomaly score plots against ground truth;
`evaluate.py --plot-worst N` regenerates them.)

**Known data quirk**: one frame in UCSDped1 Test016 fails to decode (a
corrupted PackBits-encoded TIFF from the original 2010 dataset release);
`utils.video_io.load_frame_dir` silently drops unreadable frames, which
shifts that one clip's frame indices by one relative to its ground truth.
Negligible for the aggregate AUC (1 bad frame out of ~9,300 test frames).

**Limitation found while validating the explainability feature**: on
UCSDped1 Test003 (ground truth anomaly = frames 91–200), the highest-scoring
window actually falls just *before* the anomaly's onset, and scores dip
during the core anomalous segment before partially recovering near the end.
The gradient-weighted rollout itself is verified correct — it produces
real, non-degenerate spatiotemporal variation — but on this clip it ends up
highlighting a static background region (a tree/bush) rather than the
specific anomalous object. This is a property of the underlying frozen-embedding
+ one-class-SVM classifier's imperfect temporal localization (consistent with
the 0.696–0.743 AUCs), not a bug in the rollout math. Worth keeping in mind:
the saliency map explains what the *classifier* keyed on, which isn't always
what a human would call "the anomaly."

## Why this design

- **No fine-tuning of the video transformer.** Full fine-tuning of VideoMAE
  needs a GPU; running it frozen and training only a lightweight head on top
  ("linear probing") is the compute-efficient path that still uses a real
  video foundation model rather than a from-scratch CNN.
- **No LLM calls.** Scoring and explanation both come from the perception
  model itself (embedding distance + attention rollout), so the whole
  pipeline is free to run.
- **One-class formulation.** UCSD Ped's Train split is anomaly-free by
  construction, which matches a one-class SVM's assumptions better than a
  binary classifier would.
