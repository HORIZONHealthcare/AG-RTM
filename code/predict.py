"""Estimate age with a released AG-RTM checkpoint.

Example:
    python predict.py --checkpoint ../weights/chest_age.pth \
        --input my_images.csv --output predictions.csv

The input CSV uses the same columns as training (see data/README.md). The
``biomarker_value`` column (chronological age) is optional; when it is given,
the age gap and MAE / RMSE / Pearson r are reported as well.
"""

import argparse
import os
import tempfile

import numpy as np
import pandas as pd
import torch

from engine import move_to_device
from train import build_input_dataset, build_model
from util.datasets import denormalize_biomarker


def get_args():
    parser = argparse.ArgumentParser('AG-RTM age estimation')
    parser.add_argument('--checkpoint', required=True, help='Released .pth file')
    parser.add_argument('--input', required=True, help='Input CSV')
    parser.add_argument('--output', required=True, help='Output CSV')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Default: 32 for 2D images, 4 for volumes')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_release_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    missing = {'model', 'config', 'marker'} - set(checkpoint)
    if missing:
        raise ValueError(f'{path} is not a released AG-RTM checkpoint (missing {missing})')
    return checkpoint


def main():
    cli = get_args()
    checkpoint = load_release_checkpoint(cli.checkpoint)
    config = dict(checkpoint['config'])
    # The fine-tuned weights replace the DINOv3 initialisation, so there is no
    # need to download the pretrained backbone.
    config['backbone_pretrained'] = False
    args = argparse.Namespace(**config)
    input_mode = args.input_mode
    print(f"{checkpoint['marker']} ({input_mode}), trained for {checkpoint.get('epoch', '?')} epochs")

    frame = pd.read_csv(cli.input)
    if input_mode == 'mri25d':
        # Brain Age accepts one NIfTI file per session as well as OASIS-3 archives.
        if 'archive_path' not in frame.columns and 'image_path' in frame.columns:
            frame['archive_path'] = frame['image_path']
        if 'session_id' not in frame.columns:
            frame['session_id'] = frame['subject_id']
    has_age = 'biomarker_value' in frame.columns and frame['biomarker_value'].notna().all()
    if not has_age:
        # The dataset classes need a finite target; it is not used for prediction.
        frame['biomarker_value'] = args.biomarker_mean
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, 'input.csv')
        frame.to_csv(csv_path, index=False)
        dataset = build_input_dataset(split='test', args=args, csv_path=csv_path)

        device = torch.device(cli.device)
        model = build_model(args)
        model.load_state_dict(checkpoint['model'], strict=True)
        model.to(device).eval()

        batch_size = cli.batch_size or (32 if input_mode == 'image2d' else 4)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=cli.num_workers, pin_memory=device.type == 'cuda',
        )
        ids, preds = [], []
        with torch.no_grad():
            for step, (sample_ids, samples, _) in enumerate(loader):
                samples = move_to_device(samples, device)
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    outputs = model(samples).squeeze(1)
                preds.extend(outputs.float().cpu().numpy())
                ids.extend(sample_ids)
                if step % 20 == 0:
                    print(f'batch {step + 1}/{len(loader)}')

    predicted_age = denormalize_biomarker(
        np.asarray(preds, dtype=np.float64), args.biomarker_mean, args.biomarker_std
    )
    if input_mode == 'mri25d':
        # Brain Age: one prediction per T1w run, averaged within the session.
        separator = str(args.eval_group_separator)
        runs = pd.DataFrame({
            'session_id': [str(value).split(separator, 1)[0] for value in ids],
            'predicted_age': predicted_age,
        })
        output = runs.groupby('session_id', sort=False, as_index=False).agg(
            predicted_age=('predicted_age', 'mean'),
            run_count=('predicted_age', 'size'),
        )
        sessions = frame.assign(session_id=frame['session_id'].astype(str)).set_index('session_id')
        output.insert(0, 'subject_id', output['session_id'].map(sessions['subject_id']))
        if has_age:
            output['age'] = output['session_id'].map(sessions['biomarker_value']).astype(float)
    else:
        # 2D images and CT volumes: one prediction per input row, in input order.
        if [str(value) for value in ids] != frame['subject_id'].astype(str).tolist():
            raise RuntimeError('Prediction order does not match the input CSV')
        output = frame[['subject_id', 'image_path']].copy()
        output['predicted_age'] = predicted_age
        if has_age:
            output['age'] = frame['biomarker_value'].astype(float).to_numpy()

    if has_age:
        output['age_gap'] = output['predicted_age'] - output['age']
        error = output['age_gap'].to_numpy()
        print(
            f"n={len(output)}  MAE={np.mean(np.abs(error)):.3f}  "
            f"RMSE={np.sqrt(np.mean(error ** 2)):.3f}  "
            f"mean age gap={np.mean(error):.3f}  "
            f"Pearson r={np.corrcoef(output['predicted_age'], output['age'])[0, 1]:.3f}"
        )
    os.makedirs(os.path.dirname(os.path.abspath(cli.output)), exist_ok=True)
    output.to_csv(cli.output, index=False)
    print(f'Wrote {len(output)} predictions to {cli.output}')


if __name__ == '__main__':
    main()
