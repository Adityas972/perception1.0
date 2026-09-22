"""Overlay rollout saliency maps on video frames."""
import cv2
import numpy as np


def overlay_saliency(frames: list[np.ndarray], saliency: np.ndarray, tubelet_size: int = 2, alpha: float = 0.5) -> list[np.ndarray]:
    """frames: list of `tubelet_size * saliency.shape[0]` RGB frames (one clip).
    saliency: (t, h, w) array from interpret.rollout.spatiotemporal_map, one
    map per group of `tubelet_size` consecutive frames.
    Returns a new list of frames with a jet-colormap heatmap blended in."""
    sal = saliency - saliency.min()
    sal = sal / (sal.max() + 1e-8)

    out = [None] * len(frames)
    for t_idx, group_map in enumerate(sal):
        heat = (group_map * 255).astype(np.uint8)
        for k in range(tubelet_size):
            frame_idx = t_idx * tubelet_size + k
            if frame_idx >= len(frames):
                break
            frame = frames[frame_idx]
            heat_resized = cv2.resize(heat, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
            heat_color = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)
            heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
            out[frame_idx] = cv2.addWeighted(frame, 1 - alpha, heat_color, alpha, 0)
    return out
