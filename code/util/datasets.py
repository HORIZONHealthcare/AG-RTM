import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class BiomarkerDataset(Dataset):
    def __init__(self, csv_path, transform, biomarker_mean=50, biomarker_std=50):
        self.df = pd.read_csv(csv_path)
        assert {'subject_id', 'image_path', 'biomarker_value'}.issubset(self.df.columns), \
            f"CSV must have columns: subject_id, image_path, biomarker_value. Got: {list(self.df.columns)}"
        self.transform = transform
        self.biomarker_mean = biomarker_mean
        self.biomarker_std = biomarker_std
        print(f"Loaded {len(self.df)} samples from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        subject_id = str(row['subject_id'])
        image_path = row['image_path']
        value = float(row['biomarker_value'])

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        value = torch.tensor(value, dtype=torch.float)
        value = (value - self.biomarker_mean) / self.biomarker_std
        return subject_id, image.float(), value


def build_dataset(split, args):
    transform = build_transform(split == 'train', args)
    default_name = f'{split}.csv'
    csv_name = getattr(args, f'{split}_csv_name', default_name) or default_name
    csv_path = os.path.join(args.splits_dir, csv_name)
    return BiomarkerDataset(csv_path, transform, args.biomarker_mean, args.biomarker_std)


def denormalize_biomarker(values, mean, std):
    return values * std + mean


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    if is_train:
        transform = create_transform(
            input_size=args.backbone_img_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    t = []
    if args.backbone_img_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.backbone_img_size / crop_pct)
    t.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC))
    t.append(transforms.CenterCrop(args.backbone_img_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
