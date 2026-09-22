"""UCSD Ped1/Ped2 dataset helpers: clip listing and frame-level ground truth.

Ground truth comes from the dataset's `UCSDped{1,2}.m` file under each Test
folder — a MATLAB cell array of `gt_frame = [start:end]` ranges, one entry
per test clip, in the same order as the sorted Test### directories. We parse
it with a regex rather than pulling in a MATLAB parser, since the format is
a single fixed pattern repeated per line.
"""
import re
from pathlib import Path

import numpy as np

GT_LINE_RE = re.compile(r"gt_frame\s*=\s*\[([^\]]+)\]")
PAIR_RE = re.compile(r"(\d+)\s*:\s*(\d+)")


def list_clips(ped_dir: Path, split: str) -> list[Path]:
    """Sorted clip directories for a split ('Train' or 'Test'), excluding
    the `*_gt` pixel-mask directories."""
    split_dir = Path(ped_dir) / split
    clips = [d for d in split_dir.iterdir() if d.is_dir() and not d.name.endswith("_gt")]
    return sorted(clips, key=lambda p: p.name)


def parse_gt_ranges(ped_dir: Path) -> list[list[tuple[int, int]]]:
    """Returns, per test clip (in Test001, Test002, ... order), a list of
    1-indexed inclusive (start, end) abnormal-frame ranges. A clip can have
    more than one disjoint range, e.g. `gt_frame = [5:90, 140:200]`."""
    ped_dir = Path(ped_dir)
    m_file = next(p for p in ped_dir.glob("Test/*.m") if not p.name.endswith("~"))
    ranges_per_clip = []
    for line in m_file.read_text().splitlines():
        bracket = GT_LINE_RE.search(line)
        if bracket:
            pairs = PAIR_RE.findall(bracket.group(1))
            ranges_per_clip.append([(int(s), int(e)) for s, e in pairs])
    return ranges_per_clip


def frame_labels(num_frames: int, ranges: list[tuple[int, int]]) -> np.ndarray:
    """1-indexed inclusive ranges -> 0-indexed boolean array of length num_frames."""
    labels = np.zeros(num_frames, dtype=bool)
    for start, end in ranges:
        labels[start - 1:end] = True
    return labels
