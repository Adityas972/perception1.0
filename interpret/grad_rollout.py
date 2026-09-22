"""Gradient-weighted attention rollout: "why did the SVM call *this* clip
anomalous", not just "what does the transformer attend to in general".

interpret/rollout.py's plain rollout is unconditional — it describes the
backbone's attention regardless of any downstream task, so it's a poor guide
to which patches actually pushed the anomaly score up. This module instead
backprops the anomaly score through the frozen backbone into each layer's
attention weights, then follows Chefer et al. (CVPR 2021) / the "grad
rollout" recipe used in vit-explain: weight each layer's attention by
ReLU(attention * its own gradient) before doing the same identity-mixing
rollout composition as the unconditional version. The ReLU keeps only
patches that *positively* contributed to the score — a patch the model
attended to but that pushed the score toward "normal" is dropped rather than
shown as salient.
"""
import torch

from model.classifier import AnomalyScorer
from model.feature_extractor import forward_with_grad, get_device


def compute_grad_rollout(attn_list: list[torch.Tensor], grad_list: list[torch.Tensor]) -> torch.Tensor:
    """attn_list, grad_list: matching lists of (B, heads, T, T) tensors.
    Returns (B, T, T) grad-weighted rollout matrix."""
    B, _, T, _ = attn_list[0].shape
    eye = torch.eye(T, device=attn_list[0].device).unsqueeze(0).expand(B, -1, -1)

    rollout = None
    for A, G in zip(attn_list, grad_list):
        cam = (A * G).clamp(min=0).mean(dim=1)  # (B, T, T): positive contribution, averaged over heads
        # unlike raw attention, cam's rows don't sum to 1 — renormalize to a
        # distribution first, or the fixed 0.5/0.5 identity mix below is
        # dominated by the identity term and rollout collapses to ~uniform.
        cam = cam / cam.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        cam_hat = 0.5 * cam + 0.5 * eye
        cam_hat = cam_hat / cam_hat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        rollout = cam_hat if rollout is None else cam_hat @ rollout
    return rollout


def explain(frames: list, scorer: AnomalyScorer, grid_shape: tuple[int, int, int]):
    """Runs the full forward + backward + grad-rollout pipeline for one clip.
    Returns (anomaly_score: float, saliency: (t, h, w) numpy array)."""
    device = get_device()
    pooled, attentions = forward_with_grad(frames)
    score = scorer.torch_score(pooled, device)
    score.backward()

    attn_detached = [a.detach() for a in attentions]
    grad_detached = [a.grad.detach() for a in attentions]
    rollout = compute_grad_rollout(attn_detached, grad_detached)  # (1, T, T)

    importance = rollout.mean(dim=1)  # influence on the mean-pooled output, same reasoning as rollout.py
    t, h, w = grid_shape
    saliency = importance.view(t, h, w).cpu().numpy()
    return score.item(), saliency
