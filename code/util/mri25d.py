"""On-the-fly 2.5D sampling from T1w MRI (OASIS-3 session archives or NIfTI files)."""

from __future__ import annotations

import gzip
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from PIL import Image


View = Literal["axial", "coronal", "sagittal"]
VIEW_AXIS: dict[View, int] = {"sagittal": 0, "coronal": 1, "axial": 2}
VIEW_ORDER: tuple[View, ...] = ("axial", "coronal", "sagittal")


@dataclass(frozen=True)
class MRI25DConfig:
    output_size: int = 224
    axial_slices: int = 32
    coronal_slices: int = 8
    sagittal_slices: int = 8
    intensity_low_percentile: float = 0.5
    intensity_high_percentile: float = 99.5
    crop_margin_mm: float = 5.0
    min_foreground_fraction: float = 0.005

    def validate(self) -> None:
        if self.output_size <= 0 or self.axial_slices <= 0:
            raise ValueError("MRI output size and axial slice count must be positive")
        if self.coronal_slices < 0 or self.sagittal_slices < 0:
            raise ValueError("MRI auxiliary slice counts cannot be negative")
        if not 0 <= self.intensity_low_percentile < self.intensity_high_percentile <= 100:
            raise ValueError("Invalid MRI intensity percentile interval")
        if self.crop_margin_mm < 0:
            raise ValueError("MRI crop margin must be non-negative")
        if not 0 <= self.min_foreground_fraction < 1:
            raise ValueError("Invalid MRI minimum foreground fraction")


@dataclass
class PreparedMRI:
    volume: np.ndarray
    foreground: np.ndarray
    spacing: np.ndarray
    affine: np.ndarray
    intensity_lower: float
    intensity_upper: float
    foreground_threshold: float
    crop_lower: np.ndarray
    crop_upper: np.ndarray


def sample_positions(
    axis_length: int,
    count: int,
    *,
    training: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if axis_length <= 0 or count <= 0:
        raise ValueError(f"axis_length and count must be positive: {axis_length}, {count}")
    edges = np.linspace(0.0, float(axis_length), count + 1)
    indices = []
    valid = []
    seen: set[int] = set()
    for left, right in zip(edges[:-1], edges[1:]):
        low = min(axis_length - 1, max(0, int(np.floor(left))))
        high = min(axis_length - 1, max(low, int(np.ceil(right)) - 1))
        index = (
            int(rng.integers(low, high + 1))
            if training
            else int(np.clip(np.floor((left + right) / 2.0), low, high))
        )
        indices.append(index)
        valid.append(index not in seen)
        seen.add(index)
    index_array = np.asarray(indices, dtype=np.int64)
    coordinates = (
        np.zeros(count, dtype=np.float32)
        if axis_length == 1
        else (2.0 * index_array / float(axis_length - 1) - 1.0).astype(np.float32)
    )
    return index_array, coordinates, np.asarray(valid, dtype=bool)


def resize_physical_square(
    rgb: np.ndarray,
    row_spacing_mm: float,
    column_spacing_mm: float,
    size: int,
) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {rgb.shape}")
    if size <= 0 or row_spacing_mm <= 0 or column_spacing_mm <= 0:
        raise ValueError("Output size and plane spacings must be positive")
    physical_height = float(rgb.shape[0]) * row_spacing_mm
    physical_width = float(rgb.shape[1]) * column_spacing_mm
    scale = float(size) / max(physical_height, physical_width)
    new_height = max(1, min(size, int(round(physical_height * scale))))
    new_width = max(1, min(size, int(round(physical_width * scale))))
    uint8 = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    resized = Image.fromarray(uint8, mode="RGB").resize(
        (new_width, new_height),
        resample=Image.Resampling.BICUBIC,
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = np.asarray(resized)
    return canvas.astype(np.float32) / 255.0


def _is_nifti_file(path: str | Path) -> bool:
    name = str(path).lower()
    return name.endswith(".nii.gz") or name.endswith(".nii")


def t1w_nifti_members(path: str | Path) -> tuple[str, ...]:
    """List the T1w runs of a session.

    ``path`` is either an OASIS-3 session ZIP archive (every ``*T1w*.nii.gz``
    member is one run) or a single NIfTI file, which is treated as one run.
    """
    if _is_nifti_file(path):
        return (Path(path).name,)
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.endswith(".nii.gz") and "T1w" in name
        )
    if not members:
        raise ValueError(f"No T1w NIfTI members in archive: {path}")
    return tuple(members)


def _otsu_threshold(values: np.ndarray, lower: float, upper: float) -> float:
    clipped = np.clip(values, lower, upper)
    histogram, edges = np.histogram(clipped, bins=256, range=(lower, upper))
    centres = (edges[:-1] + edges[1:]) / 2.0
    weights_left = np.cumsum(histogram, dtype=np.float64)
    weights_right = np.cumsum(histogram[::-1], dtype=np.float64)[::-1]
    means_left = np.cumsum(histogram * centres, dtype=np.float64)
    means_right = np.cumsum((histogram * centres)[::-1], dtype=np.float64)[::-1]
    valid = (weights_left > 0) & (weights_right > 0)
    mean_left = np.divide(
        means_left,
        weights_left,
        out=np.zeros_like(means_left),
        where=weights_left > 0,
    )
    mean_right = np.divide(
        means_right,
        weights_right,
        out=np.zeros_like(means_right),
        where=weights_right > 0,
    )
    between = weights_left * weights_right * (mean_left - mean_right) ** 2
    between[~valid] = -1
    return float(centres[int(np.argmax(between))])


def _read_nifti_from_archive(path: Path, member: str) -> nib.spatialimages.SpatialImage:
    if _is_nifti_file(path):
        return nib.load(str(path))
    with zipfile.ZipFile(path) as archive:
        compressed = archive.read(member)
    uncompressed = gzip.decompress(compressed)
    return nib.Nifti1Image.from_bytes(uncompressed)


def load_and_prepare_mri(
    path: str | Path,
    member: str,
    config: MRI25DConfig,
) -> PreparedMRI:
    config.validate()
    image = nib.as_closest_canonical(_read_nifti_from_archive(Path(path), member))
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D MRI, got {image.shape}: {path}:{member}")
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"Invalid MRI voxel spacing {spacing}: {path}:{member}")
    volume = np.asarray(image.dataobj, dtype=np.float32)
    finite = np.isfinite(volume)
    if not finite.any():
        raise ValueError(f"MRI contains no finite voxels: {path}:{member}")
    finite_values = volume[finite]
    lower, upper = np.percentile(
        finite_values,
        [config.intensity_low_percentile, config.intensity_high_percentile],
    )
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError(
            f"Degenerate MRI intensity range {lower}, {upper}: {path}:{member}"
        )
    threshold = _otsu_threshold(finite_values, lower, upper)
    foreground = finite & (volume > threshold)
    indices = np.argwhere(foreground)
    if not len(indices):
        raise ValueError(f"MRI foreground is empty: {path}:{member}")

    margin = np.ceil(config.crop_margin_mm / spacing).astype(np.int64)
    crop_lower = np.maximum(0, indices.min(axis=0) - margin)
    crop_upper = np.minimum(np.asarray(volume.shape), indices.max(axis=0) + margin + 1)
    slices = tuple(
        slice(int(crop_lower[axis]), int(crop_upper[axis])) for axis in range(3)
    )
    cropped = volume[slices]
    cropped_finite = finite[slices]
    cropped_foreground = foreground[slices]
    normalized = np.clip((cropped - lower) / (upper - lower), 0.0, 1.0)
    normalized[~cropped_finite] = 0.0
    return PreparedMRI(
        volume=np.ascontiguousarray(normalized, dtype=np.float32),
        foreground=np.ascontiguousarray(cropped_foreground, dtype=bool),
        spacing=spacing,
        affine=np.asarray(image.affine, dtype=np.float64),
        intensity_lower=lower,
        intensity_upper=upper,
        foreground_threshold=threshold,
        crop_lower=crop_lower,
        crop_upper=crop_upper,
    )


def extract_view_slice(volume: np.ndarray, view: View, index: int) -> np.ndarray:
    if view == "axial":
        plane = volume[:, :, index]
    elif view == "coronal":
        plane = volume[:, index, :]
    elif view == "sagittal":
        plane = volume[index, :, :]
    else:  # pragma: no cover
        raise ValueError(f"Unknown MRI view: {view}")
    return np.ascontiguousarray(np.flipud(plane.T), dtype=np.float32)


def plane_spacing(view: View, spacing: np.ndarray) -> tuple[float, float]:
    if view == "axial":
        return float(spacing[1]), float(spacing[0])
    if view == "coronal":
        return float(spacing[2]), float(spacing[0])
    if view == "sagittal":
        return float(spacing[2]), float(spacing[1])
    raise ValueError(f"Unknown MRI view: {view}")


def sample_mri_multiplanar(
    prepared: PreparedMRI,
    config: MRI25DConfig,
    *,
    training: bool,
    rng: np.random.Generator,
) -> dict[View, dict[str, np.ndarray]]:
    config.validate()
    counts: dict[View, int] = {
        "axial": config.axial_slices,
        "coronal": config.coronal_slices,
        "sagittal": config.sagittal_slices,
    }
    output: dict[View, dict[str, np.ndarray]] = {}
    for view in VIEW_ORDER:
        count = counts[view]
        if count == 0:
            output[view] = {
                "images": np.empty(
                    (0, config.output_size, config.output_size, 3),
                    dtype=np.float32,
                ),
                "indices": np.empty(0, dtype=np.int64),
                "coordinates": np.empty(0, dtype=np.float32),
                "valid": np.empty(0, dtype=bool),
            }
            continue
        axis = VIEW_AXIS[view]
        indices, coordinates, valid = sample_positions(
            prepared.volume.shape[axis],
            count,
            training=training,
            rng=rng,
        )
        row_spacing, column_spacing = plane_spacing(view, prepared.spacing)
        images = []
        foreground_valid = []
        for index in indices:
            plane = extract_view_slice(prepared.volume, view, int(index))
            mask = extract_view_slice(
                prepared.foreground.astype(np.float32),
                view,
                int(index),
            )
            foreground_valid.append(
                float(np.mean(mask > 0.5)) >= config.min_foreground_fraction
            )
            rgb = np.repeat(plane[:, :, None], 3, axis=2)
            images.append(
                resize_physical_square(
                    rgb,
                    row_spacing_mm=row_spacing,
                    column_spacing_mm=column_spacing,
                    size=config.output_size,
                )
            )
        valid &= np.asarray(foreground_valid, dtype=bool)
        output[view] = {
            "images": np.stack(images).astype(np.float32, copy=False),
            "indices": indices,
            "coordinates": coordinates.astype(np.float32, copy=False),
            "valid": valid,
        }
    return output
