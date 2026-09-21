import os
import csv
import re
import torch
import numpy as np
import pandas as pd
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from util.datasets import denormalize_biomarker


def move_to_device(value, device):
    """Move tensors in either a 2D image batch or a 2.5D sample mapping."""
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_to_device(item, device) for item in value)
    return value


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    log_writer=None,
    args=None
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq, accum_iter = 20, args.accum_iter
    optimizer.zero_grad()

    if log_writer:
        print(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (_, samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, f'Epoch: [{epoch}]')):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = move_to_device(samples, device)
        targets = targets.to(device, non_blocking=True).float()

        with torch.cuda.amp.autocast():
            outputs = model(samples).squeeze(1)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        loss /= accum_iter

        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(),
                    create_graph=False, update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        min_lr, max_lr = float('inf'), 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/train', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, epoch, mode, log_writer):
    criterion = torch.nn.MSELoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    model.eval()
    preds, trues, id_list = [], [], []

    for batch in metric_logger.log_every(data_loader, 10, f'{mode}:'):
        subject_id, images, targets = batch
        images = move_to_device(images, device)
        targets = targets.to(device, non_blocking=True).float()

        with torch.cuda.amp.autocast():
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)

        metric_logger.update(loss=loss.item())
        preds.extend(outputs.detach().cpu().numpy())
        trues.extend(targets.detach().cpu().numpy())
        id_list.extend(subject_id)

    preds = denormalize_biomarker(np.array(preds), args.biomarker_mean, args.biomarker_std)
    trues = denormalize_biomarker(np.array(trues), args.biomarker_mean, args.biomarker_std)

    # Brain Age evaluation expands every T1w run of a session.  IDs encode
    # ``session_id<separator>run_id``; average predictions within the session
    # before computing metrics.  The default empty separator (2D images and
    # CT) keeps one prediction per row.
    reconstruction_output = None
    group_separator = str(getattr(args, 'eval_group_separator', '') or '')
    if group_separator:
        encoded_ids = [str(value) for value in id_list]
        if any(group_separator not in value for value in encoded_ids):
            bad = [value for value in encoded_ids if group_separator not in value][:10]
            raise ValueError(
                f'Evaluation IDs lack group separator {group_separator!r}: {bad}'
            )
        group_ids, sample_ids = zip(
            *(value.split(group_separator, 1) for value in encoded_ids)
        )
        reconstruction_output = pd.DataFrame({
            'subject_id': group_ids,
            'sample_id': sample_ids,
            'true_value': trues,
            'predicted_value': preds,
        })
        target_ranges = reconstruction_output.groupby('subject_id')['true_value'].agg(
            lambda values: float(values.max() - values.min())
        )
        if (target_ranges > 1e-5).any():
            bad = target_ranges[target_ranges > 1e-5].head(10).to_dict()
            raise ValueError(f'Inconsistent targets within reconstruction groups: {bad}')
        grouped = reconstruction_output.groupby('subject_id', sort=False).agg(
            true_value=('true_value', 'first'),
            predicted_value=('predicted_value', 'mean'),
            reconstruction_count=('sample_id', 'size'),
        ).reset_index()
        id_list = grouped['subject_id'].astype(str).tolist()
        trues = grouped['true_value'].to_numpy(dtype=float)
        preds = grouped['predicted_value'].to_numpy(dtype=float)
        print(
            f'Aggregated {len(reconstruction_output)} reconstruction predictions '
            f'to {len(grouped)} study predictions'
        )

    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    me = np.mean(preds - trues)
    denominator = np.sum((trues - np.mean(trues)) ** 2)
    r2 = 1.0 - np.sum((preds - trues) ** 2) / denominator if denominator > 0 else float('nan')
    pearson = np.corrcoef(preds, trues)[0, 1] if len(preds) > 1 else float('nan')

    if log_writer:
        log_writer.add_scalar(f'perf/mae', mae, epoch)
        log_writer.add_scalar(f'perf/rmse', rmse, epoch)

    print(f'ME: {me:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}, Pearson: {pearson:.3f}')

    metric_logger.synchronize_between_processes()

    suffix = getattr(args, 'eval_output_suffix', '') or ''
    suffix = re.sub(r'[^A-Za-z0-9._+-]+', '_', str(suffix).strip().strip('_'))
    file_stem = mode if not suffix else f'{mode}_{suffix}'

    results_path = os.path.join(exp_dir, f'metrics_{file_stem}.csv')
    file_exists = os.path.isfile(results_path)
    with open(results_path, 'a', newline='', encoding='utf8') as f:
        wf = csv.writer(f)
        if not file_exists:
            wf.writerow(['epoch', 'ME', 'MAE', 'RMSE', 'R2', 'Pearson'])
        wf.writerow([epoch, me, mae, rmse, r2, pearson])

    df_output = pd.DataFrame({
        'subject_id': id_list,
        'true_value': trues,
        'predicted_value': preds
    })
    if reconstruction_output is not None:
        counts = reconstruction_output.groupby('subject_id').size()
        df_output['reconstruction_count'] = df_output['subject_id'].map(counts).astype(int)
        reconstruction_output.to_csv(
            os.path.join(exp_dir, f'predictions_{file_stem}_reconstructions.csv'),
            index=False,
        )
    output_csv_path = os.path.join(exp_dir, f'predictions_{file_stem}.csv')
    df_output.to_csv(output_csv_path, index=False)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats.update({'mae': mae, 'rmse': rmse, 'me': me, 'r2': r2, 'pearson': pearson})
    return stats, mae
