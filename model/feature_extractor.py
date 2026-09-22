"""Frozen VideoMAE feature extraction.

We never fine-tune VideoMAE itself — a full fine-tune of an 86M-parameter
video transformer is impractical on a laptop with no dedicated GPU. Instead
we run it purely as a feature extractor (forward passes only, no gradients
into the backbone) and train a lightweight anomaly scorer on top of the
pooled embeddings. This is standard "linear probing" and is cheap enough to
run on Apple Silicon MPS.
"""
from functools import lru_cache

import numpy as np
import torch
from transformers import VideoMAEImageProcessor, VideoMAEModel

CHECKPOINT = "MCG-NJU/videomae-base-finetuned-kinetics"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def _load_model():
    device = get_device()
    processor = VideoMAEImageProcessor.from_pretrained(CHECKPOINT)
    # eager attention: SDPA (the default) doesn't materialize attention
    # weights even when output_attentions=True, which we need for rollout.
    model = VideoMAEModel.from_pretrained(CHECKPOINT, attn_implementation="eager")
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model, device


@torch.no_grad()
def extract_embedding(frames: list[np.ndarray]) -> np.ndarray:
    """frames: list of exactly 16 RGB (H, W, 3) uint8 arrays (one clip).
    Returns a (768,) mean-pooled embedding."""
    processor, model, device = _load_model()
    inputs = processor(list(frames), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    # last_hidden_state: (1, num_tokens, hidden_dim); VideoMAE has no CLS
    # token, so mean-pool over tokens the same way the classification head does.
    pooled = outputs.last_hidden_state.mean(dim=1)
    return pooled.squeeze(0).cpu().numpy()


@torch.no_grad()
def extract_embedding_with_attention(frames: list[np.ndarray]):
    """Same as extract_embedding, but also returns per-layer attention
    weights (for explainability via attention rollout)."""
    processor, model, device = _load_model()
    inputs = processor(list(frames), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_attentions=True)
    pooled = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
    attentions = [a.cpu() for a in outputs.attentions]  # list of (1, heads, T, T)
    return pooled, attentions


def forward_with_grad(frames: list[np.ndarray]):
    """Runs VideoMAE with gradients enabled, for gradient-weighted attention
    rollout (interpret/grad_rollout.py). The backbone's *parameters* stay
    frozen (requires_grad=False, set once in _load_model) — we only need
    gradients to flow through the *activations* so we can backprop an
    anomaly score into the attention maps, not to update any weights.

    Returns (pooled_embedding, attentions), both still attached to the
    autograd graph — the caller is responsible for calling .backward() and
    reading .grad off the attention tensors before they go out of scope.
    """
    processor, model, device = _load_model()
    inputs = processor(list(frames), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # every model parameter is frozen (requires_grad=False), so without this
    # the whole forward pass would have no tensor requiring grad at all and
    # `.backward()` would fail with "does not require grad and does not have
    # a grad_fn" — the input pixel values are what seeds the graph.
    inputs["pixel_values"].requires_grad_(True)
    outputs = model(**inputs, output_attentions=True)
    for attn in outputs.attentions:
        attn.retain_grad()  # non-leaf tensors don't keep .grad by default
    pooled = outputs.last_hidden_state.mean(dim=1)  # (1, 768)
    return pooled, outputs.attentions


def patch_grid_shape(num_frames: int = 16, tubelet_size: int = 2, image_size: int = 224, patch_size: int = 16):
    """VideoMAE's patch embedding grid: (temporal, height, width) in tokens."""
    t = num_frames // tubelet_size
    hw = image_size // patch_size
    return t, hw, hw
