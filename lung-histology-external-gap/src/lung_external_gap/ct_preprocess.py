from __future__ import annotations

import numpy as np


def lung_window(volume_hu, low=-1000.0, high=400.0):
    x = np.asarray(volume_hu, dtype=np.float32)
    x = np.clip(x, low, high)
    return (x - low) / (high - low)


def fmcib_intensity_scale(volume_hu):
    """Published scaling used in the manuscript."""
    return (np.asarray(volume_hu, dtype=np.float32) + 1024.0) / 3072.0


def axis_aligned_cube_bounds(center_xyz_mm, size_mm=50.0):
    c = np.asarray(center_xyz_mm, dtype=float)
    h = size_mm / 2.0
    return c - h, c + h


def tri_planar_offsets(center_index, spacing_xyz_mm, offset_mm=2.0):
    """Return approximate +/- index offsets for tri-planar extraction.

    This helper only computes index locations; interpolation/resampling belongs
    to the dataset-specific loader.
    """
    center = np.asarray(center_index, dtype=float)
    spacing = np.asarray(spacing_xyz_mm, dtype=float)
    delta = offset_mm / spacing
    return {
        "x": (center - [delta[0], 0, 0], center, center + [delta[0], 0, 0]),
        "y": (center - [0, delta[1], 0], center, center + [0, delta[1], 0]),
        "z": (center - [0, 0, delta[2]], center, center + [0, 0, delta[2]]),
    }
