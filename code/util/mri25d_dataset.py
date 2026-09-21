"""2.5D T1w MRI age-regression dataset (Brain Age)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from util.mri25d import (
    MRI25DConfig,
    VIEW_ORDER,
    load_and_prepare_mri,
    sample_mri_multiplanar,
    t1w_nifti_members,
)


VIEW_IDS = {"axial": 0, "coronal": 1, "sagittal": 2}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class MRI25DDataset(Dataset):
    """Use one random repeat per training session and all repeats for evaluation."""

    def __init__(self, csv_path: str | Path, args, is_train: bool = False):
        self.csv_path = Path(csv_path)
        manifest = pd.read_csv(self.csv_path)
        required = {
            "subject_id",
            "session_id",
            "archive_path",
            "biomarker_value",
        }
        if not required.issubset(manifest.columns):
            raise ValueError(
                f"CSV must contain {sorted(required)}; got {list(manifest.columns)}"
            )
        if manifest["session_id"].duplicated().any():
            raise ValueError(f"Duplicate MRI sessions in {self.csv_path}")
        self.is_train = bool(is_train)
        self.eval_group_separator = str(
            getattr(args, "eval_group_separator", "") or ""
        )
        if not self.is_train and not self.eval_group_separator:
            raise ValueError("MRI repeat-averaged evaluation needs eval_group_separator")

        rows = []
        for _, source_row in manifest.sort_values("session_id").iterrows():
            archive_path = Path(str(source_row["archive_path"]))
            if not archive_path.is_file():
                raise FileNotFoundError(f"Missing MRI archive: {archive_path}")
            members = t1w_nifti_members(archive_path)
            if self.is_train:
                row = source_row.copy()
                row["_members"] = members
                rows.append(row)
            else:
                for member in members:
                    row = source_row.copy()
                    row["_member"] = member
                    row["_run_id"] = (
                        Path(member).name.removesuffix(".nii.gz").removesuffix(".nii")
                    )
                    rows.append(row)
        self.df = pd.DataFrame(rows).reset_index(drop=True)
        self.biomarker_mean = float(args.biomarker_mean)
        self.biomarker_std = float(args.biomarker_std)
        if not np.isfinite(self.biomarker_std) or self.biomarker_std <= 0:
            raise ValueError(f"biomarker_std must be positive: {self.biomarker_std}")

        self.config = MRI25DConfig(
            output_size=int(getattr(args, "backbone_img_size", 224)),
            axial_slices=int(getattr(args, "mri_axial_slices", 32)),
            coronal_slices=int(getattr(args, "mri_coronal_slices", 8)),
            sagittal_slices=int(getattr(args, "mri_sagittal_slices", 8)),
            intensity_low_percentile=float(
                getattr(args, "mri_intensity_low_percentile", 0.5)
            ),
            intensity_high_percentile=float(
                getattr(args, "mri_intensity_high_percentile", 99.5)
            ),
            crop_margin_mm=float(getattr(args, "mri_crop_margin_mm", 5.0)),
            min_foreground_fraction=float(
                getattr(args, "mri_min_foreground_fraction", 0.005)
            ),
        )
        self.config.validate()
        self.affine_degrees = float(getattr(args, "mri_affine_degrees", 0.0))
        self.affine_translate = float(getattr(args, "mri_affine_translate", 0.0))
        self.affine_scale_min = float(getattr(args, "mri_affine_scale_min", 1.0))
        self.affine_scale_max = float(getattr(args, "mri_affine_scale_max", 1.0))
        if self.affine_degrees < 0 or not 0 <= self.affine_translate < 1:
            raise ValueError("Invalid MRI affine rotation/translation range")
        if not 0 < self.affine_scale_min <= self.affine_scale_max:
            raise ValueError("Invalid MRI affine scale range")

        print(
            f"Loaded {len(self.df)} MRI "
            f"{'sessions' if self.is_train else 'T1w repeats'} from {self.csv_path} "
            f"(train={self.is_train}, slices={self.config.axial_slices}/"
            f"{self.config.coronal_slices}/{self.config.sagittal_slices})"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _rng(self) -> np.random.Generator:
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
        return transform_functional.affine(
            images,
            angle=float(rng.uniform(-self.affine_degrees, self.affine_degrees)),
            translate=translate,
            scale=float(rng.uniform(self.affine_scale_min, self.affine_scale_max)),
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        session_id = str(row["session_id"])
        value = float(row["biomarker_value"])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite target for MRI session {session_id}: {value}")
        rng = self._rng()
        if self.is_train:
            members = row["_members"]
            member = members[int(rng.integers(0, len(members)))]
        else:
            member = str(row["_member"])

        prepared = load_and_prepare_mri(
            Path(str(row["archive_path"])),
            member,
            self.config,
        )
        sampled = sample_mri_multiplanar(
            prepared,
            self.config,
            training=self.is_train,
            rng=rng,
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
                        (len(images),),
                        VIEW_IDS[view],
                        dtype=torch.long,
                    )
                )
        images = torch.cat(image_parts, dim=0).float()
        valid = torch.cat(valid_parts, dim=0).bool()
        coordinates = torch.cat(coordinate_parts, dim=0).float()
        view_ids = torch.cat(view_parts, dim=0).long()
        if not torch.any(valid & view_ids.eq(VIEW_IDS["axial"])):
            raise ValueError(f"No valid axial MRI slice: {session_id}:{member}")

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
        identifier = session_id
        if not self.is_train:
            identifier = (
                f"{session_id}{self.eval_group_separator}{row['_run_id']}"
            )
        return identifier, samples, target


def build_mri25d_dataset(split: str, args, csv_path: str | Path | None = None):
    if csv_path is None:
        default_name = f"{split}.csv"
        csv_name = getattr(args, f"{split}_csv_name", default_name) or default_name
        csv_path = os.path.join(args.splits_dir, csv_name)
    return MRI25DDataset(csv_path, args=args, is_train=(split == "train"))
