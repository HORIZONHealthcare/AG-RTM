"""On-the-fly NIfTI dataset for 2.5D CT age regression (Abdominal Age)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from util.ct25d import (
    CT25DConfig,
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_and_standardize,
    sample_multiplanar,
)


VIEW_IDS = {"axial": 0, "coronal": 1, "sagittal": 2}
VIEW_ORDER = tuple(VIEW_IDS)


class CT25DDataset(Dataset):
    """Construct 2.5D slice tokens from a NIfTI CT volume on the fly.

    Nothing derived from the volume (resampled volumes, slices, features) is
    written to disk.
    """

    def __init__(self, csv_path: str | Path, args, is_train: bool = False):
        self.csv_path = Path(csv_path)
        manifest = pd.read_csv(self.csv_path)
        required = {"subject_id", "image_path", "biomarker_value"}
        if not required.issubset(manifest.columns):
            raise ValueError(
                f"CSV must contain {sorted(required)}; got {list(manifest.columns)}"
            )
        self.is_train = bool(is_train)
        self.df = manifest.reset_index(drop=True)
        self.biomarker_mean = float(args.biomarker_mean)
        self.biomarker_std = float(args.biomarker_std)
        if not np.isfinite(self.biomarker_std) or self.biomarker_std <= 0:
            raise ValueError(f"biomarker_std must be positive: {self.biomarker_std}")

        self.config = CT25DConfig(
            target_spacing_mm=float(getattr(args, "ct_target_spacing_mm", 1.5)),
            body_threshold_hu=float(getattr(args, "ct_body_threshold_hu", -950.0)),
            body_margin_mm=float(getattr(args, "ct_body_margin_mm", 15.0)),
            output_size=int(getattr(args, "backbone_img_size", 224)),
            axial_slices=int(getattr(args, "ct_axial_slices", 32)),
            coronal_slices=int(getattr(args, "ct_coronal_slices", 8)),
            sagittal_slices=int(getattr(args, "ct_sagittal_slices", 8)),
            min_aux_aspect_ratio=float(
                getattr(args, "ct_min_aux_aspect_ratio", 0.25)
            ),
            min_body_component_fraction=float(
                getattr(args, "ct_min_body_component_fraction", 0.005)
            ),
            min_body_component_slenderness=float(
                getattr(args, "ct_min_body_component_slenderness", 0.12)
            ),
            background_hu=float(getattr(args, "ct_background_hu", -1350.0)),
        )
        if self.config.axial_slices <= 0:
            raise ValueError("ct_axial_slices must be positive")
        if self.config.coronal_slices < 0 or self.config.sagittal_slices < 0:
            raise ValueError("Auxiliary slice counts cannot be negative")

        self.affine_degrees = float(getattr(args, "ct_affine_degrees", 0.0))
        self.affine_translate = float(getattr(args, "ct_affine_translate", 0.0))
        self.affine_scale_min = float(getattr(args, "ct_affine_scale_min", 1.0))
        self.affine_scale_max = float(getattr(args, "ct_affine_scale_max", 1.0))
        if self.affine_degrees < 0 or not 0 <= self.affine_translate < 1:
            raise ValueError("Invalid CT affine rotation/translation range")
        if not 0 < self.affine_scale_min <= self.affine_scale_max:
            raise ValueError("Invalid CT affine scale range")

        print(
            f"Loaded {len(self.df)} CT scans "
            f"from {self.csv_path} ("
            f"train={self.is_train}, slices="
            f"{self.config.axial_slices}/"
            f"{self.config.coronal_slices}/"
            f"{self.config.sagittal_slices})"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _rng(self) -> np.random.Generator:
        # PyTorch seeds NumPy separately in each dataloader worker. Drawing a
        # fresh seed here gives deterministic runs while varying jitter across
        # samples and newly created epoch workers.
        seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        return np.random.default_rng(seed)

    def _augment(self, images: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        if not self.is_train:
            return images
        if (
            self.affine_degrees == 0
            and self.affine_translate == 0
            and self.affine_scale_min == 1
            and self.affine_scale_max == 1
        ):
            return images

        size = int(images.shape[-1])
        max_shift = int(round(self.affine_translate * size))
        translate = [
            int(rng.integers(-max_shift, max_shift + 1)) if max_shift else 0,
            int(rng.integers(-max_shift, max_shift + 1)) if max_shift else 0,
        ]
        angle = float(rng.uniform(-self.affine_degrees, self.affine_degrees))
        scale = float(rng.uniform(self.affine_scale_min, self.affine_scale_max))
        return transform_functional.affine(
            images,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        subject_id = str(row["subject_id"])
        value = float(row["biomarker_value"])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite target for {subject_id}: {value}")

        rng = self._rng()
        image_path = Path(str(row["image_path"]))
        prepared = load_and_standardize(image_path, config=self.config)
        sampled = sample_multiplanar(
            prepared.volume_hu,
            config=self.config,
            training=self.is_train,
            rng=rng,
            normalized=False,
        )

        image_parts = []
        valid_parts = []
        coordinate_parts = []
        view_parts = []
        for view in VIEW_ORDER:
            item = sampled[view]
            images = item["images"]
            if len(images):
                image_parts.append(torch.from_numpy(np.moveaxis(images, -1, 1)))
                valid_parts.append(torch.from_numpy(item["valid"]))
                coordinate_parts.append(torch.from_numpy(item["coordinates"]))
                view_parts.append(
                    torch.full(
                        (len(images),), VIEW_IDS[view], dtype=torch.long
                    )
                )

        images = torch.cat(image_parts, dim=0).float()
        valid = torch.cat(valid_parts, dim=0).bool()
        coordinates = torch.cat(coordinate_parts, dim=0).float()
        view_ids = torch.cat(view_parts, dim=0).long()
        if not torch.any(valid & (view_ids == VIEW_IDS["axial"])):
            raise ValueError(f"No valid axial slice after preprocessing: {subject_id}")

        images = self._augment(images, rng)
        mean = torch.as_tensor(IMAGENET_MEAN, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.as_tensor(IMAGENET_STD, dtype=images.dtype).view(1, 3, 1, 1)
        images = (images - mean) / std

        target = torch.tensor(
            (value - self.biomarker_mean) / self.biomarker_std,
            dtype=torch.float32,
        )
        samples = {
            "images": images.contiguous(),
            "valid": valid,
            "coordinates": coordinates,
            "view_ids": view_ids,
        }
        return subject_id, samples, target


def build_ct25d_dataset(split: str, args, csv_path: str | Path | None = None):
    if csv_path is None:
        default_name = f"{split}.csv"
        csv_name = getattr(args, f"{split}_csv_name", default_name) or default_name
        csv_path = os.path.join(args.splits_dir, csv_name)
    return CT25DDataset(csv_path, args=args, is_train=(split == "train"))
