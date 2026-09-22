"""Attention rollout for VideoMAE (Abnar & Zuidema, 2020), adapted from the
CLS-token version in vit-from-scratch/interpret/rollout.py.

The core rollout math — composing each layer's attention with an identity
term to account for the residual stream, then chaining across depth — is
architecture-agnostic and reused as-is. The one real difference is how you
read a token-importance vector out of the resulting (T, T) matrix:

- ViT classifies from a dedicated [CLS] token, so "what mattered" is just
  that token's row: rollout[0, 1:].
- VideoMAE has no [CLS] token; VideoMAEModel classifies by mean-pooling
  every patch token's final representation. So no single row is "the"
  output — instead, token j's influence on the pooled prediction is its
  average influence across *all* output tokens: rollout[:, j].mean(). That's
  just a column-mean instead of a row lookup, which is exactly what
  mean-pooling over the rollout's output axis represents.
"""
import torch


def compute_rollout(attn_list: list[torch.Tensor]) -> torch.Tensor:
    """attn_list: list of (B, n_heads, T, T) attention weights, one per block
    (in order, block 0 first). Returns (B, T, T) rollout matrix."""
    layer_attns = [a.mean(dim=1) for a in attn_list]  # average over heads: (B, T, T)

    B, T, _ = layer_attns[0].shape
    eye = torch.eye(T, device=layer_attns[0].device).unsqueeze(0).expand(B, -1, -1)

    rollout = None
    for A in layer_attns:
        A_hat = 0.5 * A + 0.5 * eye
        A_hat = A_hat / A_hat.sum(dim=-1, keepdim=True)
        rollout = A_hat if rollout is None else A_hat @ rollout
    return rollout


def token_importance(attn_list: list[torch.Tensor]) -> torch.Tensor:
    """Returns (B, T): each input token's rolled-out influence on the
    mean-pooled final representation (see module docstring)."""
    rollout = compute_rollout(attn_list)  # (B, T, T)
    return rollout.mean(dim=1)  # average over output tokens = influence on the pooled output


def spatiotemporal_map(attn_list: list[torch.Tensor], grid_shape: tuple[int, int, int]) -> torch.Tensor:
    """grid_shape: (num_temporal_tokens, grid_h, grid_w) from
    feature_extractor.patch_grid_shape(). Returns (B, t, h, w) saliency."""
    importance = token_importance(attn_list)  # (B, T)
    t, h, w = grid_shape
    B = importance.shape[0]
    return importance.view(B, t, h, w)
