"""On-the-fly 2.5D CT preprocessing for Abdominal Age (Merlin abdominal CT).

The functions in this module never persist resampled volumes or extracted
slices. A NIfTI is canonicalized, body-cropped, resampled, and sampled in
memory; callers decide whether the resulting slice tensors are used for a
model batch or a small visual audit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output
from PIL import Image
from scipy import ndimage


View = Literal["axial", "coronal", "sagittal"]

VIEW_AXIS: dict[View, int] = {
    "sagittal": 0,  # RAS left-right
    "coronal": 1,   # RAS posterior-anterior
    "axial": 2,     # RAS inferior-superior
}

HU_WINDOWS = (
    (50.0, 400.0),    # soft tissue: [-150, 250]
    (-600.0, 1500.0), # lung / low density: [-1350, 150]
    (400.0, 1800.0),  # bone / high density: [-500, 1300]
)

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class CT25DConfig:
    target_spacing_mm: float = 1.5
    body_threshold_hu: float = -950.0
    body_margin_mm: float = 15.0
    output_size: int = 224
    axial_slices: int = 32
    coronal_slices: int = 8
    sagittal_slices: int = 8
    min_aux_aspect_ratio: float = 0.25
    min_body_component_fraction: float = 0.005
    min_body_component_slenderness: float = 0.12
    # Equal to the lowest HU-window bound so masked/padded background maps to 0.
    background_hu: float = -1350.0

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["hu_windows"] = [list(window) for window in HU_WINDOWS]
        result["orientation"] = "canonical RAS"
        return result


@dataclass
class PreparedVolume:
    volume_hu: np.ndarray
    native_mid_slice_hu: np.ndarray
    audit: dict[str, object]


def _finite_hu(data: np.ndarray, background_hu: float) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if not np.isfinite(data).all():
        data = data.copy()
        data[~np.isfinite(data)] = background_hu
    return data


def _coarse_largest_component_bbox(
    volume_hu: np.ndarray,
    threshold_hu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a foreground bbox using a bounded-size 3D component analysis."""
    shape = np.asarray(volume_hu.shape, dtype=int)
    stride = np.maximum(1, np.ceil(shape / 192.0).astype(int))
    coarse = volume_hu[
        :: int(stride[0]), :: int(stride[1]), :: int(stride[2])
    ]
    mask = coarse > threshold_hu
    if not mask.any():
        return np.zeros(3, dtype=int), shape

    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return np.zeros(3, dtype=int), shape
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    # Prefer a large component intersecting the central half of the axial FOV.
    centre = np.asarray(mask.shape[:2]) // 2
    half = np.maximum(1, np.asarray(mask.shape[:2]) // 4)
    central_labels = np.unique(
        labels[
            max(0, centre[0] - half[0]) : centre[0] + half[0] + 1,
            max(0, centre[1] - half[1]) : centre[1] + half[1] + 1,
            :,
        ]
    )
    central_labels = central_labels[central_labels > 0]
    if len(central_labels):
        chosen = int(central_labels[np.argmax(sizes[central_labels])])
    else:
        chosen = int(np.argmax(sizes))

    coordinates = np.argwhere(labels == chosen)
    lower_coarse = coordinates.min(axis=0)
    upper_coarse = coordinates.max(axis=0) + 1
    lower = lower_coarse * stride
    upper = np.minimum(shape, upper_coarse * stride)
    return lower.astype(int), upper.astype(int)


def body_bbox(
    volume_hu: np.ndarray,
    spacing_mm: np.ndarray,
    threshold_hu: float = -950.0,
    margin_mm: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate an axis-aligned body box and add a fixed physical margin."""
    lower, upper = _coarse_largest_component_bbox(volume_hu, threshold_hu)
    shape = np.asarray(volume_hu.shape, dtype=int)
    margin_voxels = np.ceil(margin_mm / np.asarray(spacing_mm)).astype(int)
    lower = np.maximum(0, lower - margin_voxels)
    upper = np.minimum(shape, upper + margin_voxels)
    if np.any(upper <= lower):
        raise ValueError(f"Invalid body bbox: lower={lower}, upper={upper}")
    return lower, upper


def crop_affine(affine: np.ndarray, lower: np.ndarray) -> np.ndarray:
    result = np.asarray(affine, dtype=float).copy()
    result[:3, 3] = nib.affines.apply_affine(affine, lower)
    return result


def mask_nonbody_components(
    volume_hu: np.ndarray,
    threshold_hu: float = -950.0,
    background_hu: float = -1350.0,
    dilation_voxels: int = 2,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Mask only disconnected table/hardware in each axial slice.

    The largest central component plus other sizeable, non-slender central
    components are retained, which preserves paired legs/arms. No erosion or
    opening is used: if table and body touch, both are conservatively retained
    rather than risking loss of anatomy. Hole filling keeps lung and bowel air
    within the body envelope, and dilation protects the skin edge.
    """
    foreground = np.asarray(volume_hu) > threshold_hu
    if not foreground.any():
        return np.asarray(volume_hu, dtype=np.float32), {
            "components": 0,
            "retained_mask_fraction": 1.0,
            "removed_foreground_fraction": 0.0,
        }
    body_mask = np.zeros_like(foreground)
    total_components = 0
    posterior_trimmed_slices = 0
    structure = ndimage.generate_binary_structure(2, 2)
    x_size, y_size = foreground.shape[:2]
    central = (
        slice(int(0.15 * x_size), max(int(0.85 * x_size), 1)),
        slice(int(0.15 * y_size), max(int(0.90 * y_size), 1)),
    )

    for axial_index in range(foreground.shape[2]):
        plane = foreground[:, :, axial_index].copy()
        # In canonical RAS, the scanner table is posterior (low y). Find the
        # last sustained near-empty band before the dense body peak and mask
        # only the region posterior to that band. This also handles a sparse
        # one-pixel table-to-body bridge without eroding anatomy.
        y_counts = plane.sum(axis=0)
        body_search_start = int(0.20 * y_size)
        body_search_stop = max(body_search_start + 1, int(0.92 * y_size))
        body_peak_y = body_search_start + int(
            np.argmax(y_counts[body_search_start:body_search_stop])
        )
        gap_threshold = max(2, int(round(0.02 * x_size)))
        gap = y_counts[:body_peak_y] <= gap_threshold
        gap_end = 0
        run_start: int | None = None
        for y_index, is_gap in enumerate(gap):
            if is_gap and run_start is None:
                run_start = y_index
            if (not is_gap or y_index == len(gap) - 1) and run_start is not None:
                run_stop = y_index if not is_gap else y_index + 1
                if run_stop - run_start >= 3:
                    gap_end = run_stop
                run_start = None
        if 0 < gap_end <= int(0.45 * y_size):
            plane[:, :gap_end] = False
            posterior_trimmed_slices += 1
        labels, count = ndimage.label(plane, structure=structure)
        total_components += int(count)
        if count == 0:
            continue
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        central_labels = np.unique(labels[central])
        central_labels = central_labels[central_labels > 0]
        if len(central_labels):
            primary = int(central_labels[np.argmax(sizes[central_labels])])
        else:
            primary = int(np.argmax(sizes))
        primary_size = int(sizes[primary])
        objects = ndimage.find_objects(labels)
        primary_slice = objects[primary - 1]
        if primary_slice is None:
            continue
        primary_x_extent = primary_slice[0].stop - primary_slice[0].start
        primary_y_extent = primary_slice[1].stop - primary_slice[1].start
        primary_slenderness = min(primary_x_extent, primary_y_extent) / max(
            primary_x_extent, primary_y_extent
        )
        primary_centre_y = (primary_slice[1].start + primary_slice[1].stop) / 2.0
        if primary_slenderness < 0.12 or not (
            0.20 * y_size <= primary_centre_y <= 0.92 * y_size
        ):
            continue
        retained = labels == primary

        for component, component_slice in enumerate(objects, start=1):
            if component == primary or component_slice is None:
                continue
            if sizes[component] < max(16, 0.08 * primary_size):
                continue
            x_extent = component_slice[0].stop - component_slice[0].start
            y_extent = component_slice[1].stop - component_slice[1].start
            slenderness = min(x_extent, y_extent) / max(x_extent, y_extent)
            centre_x = (component_slice[0].start + component_slice[0].stop) / 2.0
            centre_y = (component_slice[1].start + component_slice[1].stop) / 2.0
            if slenderness < 0.12:
                continue
            if not (0.05 * x_size <= centre_x <= 0.95 * x_size):
                continue
            if not (0.20 * y_size <= centre_y <= 0.92 * y_size):
                continue
            retained |= labels == component

        retained = ndimage.binary_fill_holes(retained)
        if dilation_voxels > 0:
            retained = ndimage.binary_dilation(
                retained,
                structure=structure,
                iterations=dilation_voxels,
            )
        body_mask[:, :, axial_index] = retained

    removed_foreground = foreground & ~body_mask
    output = np.asarray(volume_hu, dtype=np.float32).copy()
    output[~body_mask] = background_hu
    stats: dict[str, float | int] = {
        "components": total_components,
        "posterior_trimmed_slices": posterior_trimmed_slices,
        "retained_mask_fraction": round(float(body_mask.mean()), 6),
        "removed_foreground_fraction": round(
            float(removed_foreground.sum() / max(1, foreground.sum())), 6
        ),
    }
    return output, stats


def load_and_standardize(
    path: str | Path,
    config: CT25DConfig | None = None,
) -> PreparedVolume:
    """Load one NIfTI and return a canonical, isotropic, body-cropped volume."""
    config = config or CT25DConfig()
    path = Path(path)
    started = perf_counter()
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI, got shape {image.shape}: {path}")

    native_shape = tuple(int(value) for value in image.shape)
    native_spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    native_axcodes = tuple(str(value) for value in nib.aff2axcodes(image.affine))
    native_mid = _finite_hu(
        np.asanyarray(image.dataobj[:, :, native_shape[2] // 2]),
        config.background_hu,
    )

    canonical = nib.as_closest_canonical(image)
    canonical_spacing = np.asarray(
        nib.affines.voxel_sizes(canonical.affine)[:3], dtype=float
    )
    canonical_data = _finite_hu(
        canonical.get_fdata(dtype=np.float32, caching="unchanged"),
        config.background_hu,
    )
    canonical_shape = tuple(int(value) for value in canonical_data.shape)

    # A native-grid pre-crop bounds peak RAM and interpolation work. The body
    # box is recomputed after resampling, so the final result follows the locked
    # RAS -> isotropic -> fixed-margin body-box definition.
    pre_lower, pre_upper = body_bbox(
        canonical_data,
        canonical_spacing,
        config.body_threshold_hu,
        config.body_margin_mm + config.target_spacing_mm,
    )
    pre_slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(pre_lower, pre_upper))
    cropped_data = canonical_data[pre_slices]
    cropped = nib.Nifti1Image(
        cropped_data,
        crop_affine(canonical.affine, pre_lower),
    )
    del canonical_data

    isotropic = resample_to_output(
        cropped,
        voxel_sizes=(config.target_spacing_mm,) * 3,
        order=1,
        mode="constant",
        cval=config.background_hu,
    )
    isotropic_data = _finite_hu(
        isotropic.get_fdata(dtype=np.float32, caching="unchanged"),
        config.background_hu,
    )
    isotropic_spacing = np.asarray(
        nib.affines.voxel_sizes(isotropic.affine)[:3], dtype=float
    )
    post_lower, post_upper = body_bbox(
        isotropic_data,
        isotropic_spacing,
        config.body_threshold_hu,
        config.body_margin_mm,
    )
    post_slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(post_lower, post_upper))
    standardized = np.ascontiguousarray(isotropic_data[post_slices], dtype=np.float32)
    standardized, body_mask_audit = mask_nonbody_components(
        standardized,
        threshold_hu=config.body_threshold_hu,
        background_hu=config.background_hu,
    )

    if min(standardized.shape) < 2:
        raise ValueError(f"Degenerate standardized volume {standardized.shape}: {path}")

    finite = standardized[np.isfinite(standardized)]
    audit = {
        "image_path": str(path),
        "source_bytes": int(path.stat().st_size),
        "native_shape": list(native_shape),
        "native_spacing_mm": [round(value, 6) for value in native_spacing],
        "native_axcodes": list(native_axcodes),
        "canonical_shape": list(canonical_shape),
        "canonical_spacing_mm": [round(float(value), 6) for value in canonical_spacing],
        "pre_resample_bbox": [pre_lower.tolist(), pre_upper.tolist()],
        "isotropic_shape_before_final_crop": [int(value) for value in isotropic_data.shape],
        "isotropic_spacing_mm": [round(float(value), 6) for value in isotropic_spacing],
        "post_resample_bbox": [post_lower.tolist(), post_upper.tolist()],
        "standardized_shape": [int(value) for value in standardized.shape],
        "standardized_extent_mm": [
            round(float(size * spacing), 3)
            for size, spacing in zip(standardized.shape, isotropic_spacing)
        ],
        "hu_percentiles": {
            str(percentile): round(float(np.percentile(finite, percentile)), 3)
            for percentile in (0, 1, 25, 50, 75, 99, 100)
        },
        "body_component_mask": body_mask_audit,
        "seconds": round(perf_counter() - started, 3),
    }
    return PreparedVolume(standardized, native_mid, audit)


def sample_positions(
    axis_length: int,
    count: int,
    training: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one index per equal-width normalized-position bin."""
    if axis_length <= 0 or count <= 0:
        raise ValueError(f"axis_length and count must be positive: {axis_length}, {count}")
    rng = rng or np.random.default_rng()
    edges = np.linspace(0.0, float(axis_length), count + 1)
    indices: list[int] = []
    valid: list[bool] = []
    seen: set[int] = set()
    for left, right in zip(edges[:-1], edges[1:]):
        low = min(axis_length - 1, max(0, int(np.floor(left))))
        high = min(axis_length - 1, max(low, int(np.ceil(right)) - 1))
        if training:
            index = int(rng.integers(low, high + 1))
        else:
            index = int(np.clip(np.floor((left + right) / 2.0), low, high))
        indices.append(index)
        valid.append(index not in seen)
        seen.add(index)
    index_array = np.asarray(indices, dtype=np.int64)
    if axis_length == 1:
        coordinates = np.zeros(count, dtype=np.float32)
    else:
        coordinates = (2.0 * index_array / float(axis_length - 1) - 1.0).astype(
            np.float32
        )
    return index_array, coordinates, np.asarray(valid, dtype=bool)


def extract_view_slice(volume_hu: np.ndarray, view: View, index: int) -> np.ndarray:
    """Extract a consistently displayed 2D plane from a canonical RAS volume."""
    axis = VIEW_AXIS[view]
    index = int(np.clip(index, 0, volume_hu.shape[axis] - 1))
    if view == "axial":
        plane = volume_hu[:, :, index]
    elif view == "coronal":
        plane = volume_hu[:, index, :]
    elif view == "sagittal":
        plane = volume_hu[index, :, :]
    else:  # pragma: no cover - protected by the View type and lookup
        raise ValueError(f"Unknown view: {view}")
    return np.ascontiguousarray(np.flipud(plane.T), dtype=np.float32)


def hu_to_rgb(slice_hu: np.ndarray) -> np.ndarray:
    """Map one HU slice to the fixed three-window pseudo-RGB representation."""
    channels = []
    for level, width in HU_WINDOWS:
        lower = level - width / 2.0
        upper = level + width / 2.0
        channel = (np.asarray(slice_hu, dtype=np.float32) - lower) / (upper - lower)
        channels.append(np.clip(channel, 0.0, 1.0))
    return np.stack(channels, axis=-1).astype(np.float32, copy=False)


def resize_square(rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """Preserve aspect ratio, pad with zero-window background, and resize."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB array, got {rgb.shape}")
    height, width = rgb.shape[:2]
    side = max(height, width)
    padded = np.zeros((side, side, 3), dtype=np.float32)
    top = (side - height) // 2
    left = (side - width) // 2
    padded[top : top + height, left : left + width] = np.clip(rgb, 0.0, 1.0)
    uint8 = np.rint(padded * 255.0).astype(np.uint8)
    resized = Image.fromarray(uint8, mode="RGB").resize(
        (size, size), resample=Image.Resampling.BICUBIC
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def slice_to_model_input(slice_hu: np.ndarray, size: int = 224) -> np.ndarray:
    """Return a normalized CxHxW DINO input without constructing a torch tensor."""
    rgb = resize_square(hu_to_rgb(slice_hu), size)
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.moveaxis(normalized.astype(np.float32, copy=False), -1, 0)


def slice_contains_body(
    slice_hu: np.ndarray,
    threshold_hu: float = -950.0,
    min_component_fraction: float = 0.005,
    min_slenderness: float = 0.12,
) -> bool:
    """Reject empty or table-only planes while retaining substantive anatomy."""
    foreground = np.asarray(slice_hu) > threshold_hu
    labels, count = ndimage.label(
        foreground,
        structure=ndimage.generate_binary_structure(2, 2),
    )
    if count == 0:
        return False
    sizes = np.bincount(labels.ravel())
    objects = ndimage.find_objects(labels)
    height, width = foreground.shape
    minimum_size = max(32, int(round(min_component_fraction * foreground.size)))
    for component in np.argsort(sizes[1:])[::-1] + 1:
        if sizes[component] < minimum_size:
            break
        component_slice = objects[int(component) - 1]
        if component_slice is None:
            continue
        component_height = component_slice[0].stop - component_slice[0].start
        component_width = component_slice[1].stop - component_slice[1].start
        slenderness = min(component_height, component_width) / max(
            component_height, component_width
        )
        centre_row = (component_slice[0].start + component_slice[0].stop) / 2.0
        centre_column = (
            component_slice[1].start + component_slice[1].stop
        ) / 2.0
        if slenderness < min_slenderness:
            continue
        if not (0.05 * height <= centre_row <= 0.95 * height):
            continue
        if not (0.05 * width <= centre_column <= 0.95 * width):
            continue
        return True
    return False


def sample_multiplanar(
    volume_hu: np.ndarray,
    config: CT25DConfig | None = None,
    training: bool = False,
    rng: np.random.Generator | None = None,
    normalized: bool = True,
) -> dict[View, dict[str, np.ndarray]]:
    """Sample the locked 32/8/8 planes from a standardized HU volume."""
    config = config or CT25DConfig()
    counts: dict[View, int] = {
        "axial": config.axial_slices,
        "coronal": config.coronal_slices,
        "sagittal": config.sagittal_slices,
    }
    result: dict[View, dict[str, np.ndarray]] = {}
    for view in ("axial", "coronal", "sagittal"):
        if counts[view] == 0:
            image_shape = (
                (0, 3, config.output_size, config.output_size)
                if normalized
                else (0, config.output_size, config.output_size, 3)
            )
            result[view] = {
                "images": np.empty(image_shape, dtype=np.float32),
                "indices": np.empty(0, dtype=np.int64),
                "coordinates": np.empty(0, dtype=np.float32),
                "valid": np.empty(0, dtype=bool),
            }
            continue
        axis = VIEW_AXIS[view]
        indices, coordinates, valid = sample_positions(
            volume_hu.shape[axis], counts[view], training=training, rng=rng
        )
        images = []
        body_valid = []
        for index in indices:
            plane = extract_view_slice(volume_hu, view, int(index))
            body_valid.append(
                slice_contains_body(
                    plane,
                    threshold_hu=config.body_threshold_hu,
                    min_component_fraction=config.min_body_component_fraction,
                    min_slenderness=config.min_body_component_slenderness,
                )
            )
            if normalized:
                images.append(slice_to_model_input(plane, config.output_size))
            else:
                images.append(resize_square(hu_to_rgb(plane), config.output_size))
        valid &= np.asarray(body_valid, dtype=bool)
        if view == "coronal":
            aspect_ratio = volume_hu.shape[2] / volume_hu.shape[0]
            if aspect_ratio < config.min_aux_aspect_ratio:
                valid[:] = False
        elif view == "sagittal":
            aspect_ratio = volume_hu.shape[2] / volume_hu.shape[1]
            if aspect_ratio < config.min_aux_aspect_ratio:
                valid[:] = False
        result[view] = {
            "images": np.stack(images),
            "indices": indices,
            "coordinates": coordinates,
            "valid": valid,
        }
    return result


def display_native_slice(slice_hu: np.ndarray, size: int = 224) -> np.ndarray:
    """Create a soft-tissue grayscale RGB image for a native-array audit view."""
    displayed = np.flipud(np.asarray(slice_hu, dtype=np.float32).T)
    level, width = HU_WINDOWS[0]
    gray = np.clip((displayed - (level - width / 2.0)) / width, 0.0, 1.0)
    return resize_square(np.repeat(gray[..., None], 3, axis=2), size)
