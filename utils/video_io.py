"""Video/frame loading helpers.

UCSD Ped clips are shipped as directories of grayscale .tif frames, not
video files, so we read frame sequences directly rather than decoding
video containers. `sample_clip` also works on an in-memory list of
frames for the Gradio demo, where input is an uploaded .mp4.
"""
from pathlib import Path

import cv2
import numpy as np


def load_frame_dir(frame_dir: Path) -> list[np.ndarray]:
    """Load a sorted sequence of image frames from a directory (UCSD Ped clip)."""
    paths = sorted(Path(frame_dir).glob("*.tif")) or sorted(Path(frame_dir).glob("*.jpg"))
    frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames if f is not None]


def load_video_file(video_path: Path) -> list[np.ndarray]:
    """Decode an .mp4/.avi file into a list of RGB frames."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def sample_clip(frames: list[np.ndarray], num_frames: int = 16) -> list[np.ndarray]:
    """Uniformly sample `num_frames` frames from a longer sequence.

    VideoMAE was pretrained on fixed-length 16-frame clips; uniform sampling
    (rather than a sliding window) keeps a clip's temporal span consistent
    regardless of the source frame rate.
    """
    if len(frames) == 0:
        raise ValueError("no frames to sample from")
    if len(frames) <= num_frames:
        idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    else:
        idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    return [frames[i] for i in idx]


def sliding_windows(frames: list[np.ndarray], num_frames: int = 16, stride: int = 8) -> list[list[np.ndarray]]:
    """Split a long sequence into overlapping fixed-length clips for scoring
    a whole video (e.g. a full UCSD test sequence) rather than a single clip."""
    if len(frames) < num_frames:
        return [sample_clip(frames, num_frames)]
    clips = []
    for start in range(0, len(frames) - num_frames + 1, stride):
        clips.append(frames[start:start + num_frames])
    return clips
