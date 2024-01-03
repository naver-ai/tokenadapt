# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------------------------------------
# Additionally modified by NAVER Cloud Corp. for TokenAdapt
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
from token_transform import *
import utils


def train_one_epoch(model: torch.nn.Module, codebook: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, mixup_ta_fn: Optional[Mixup] = None,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    transform_color = build_color_transform(True, args)
    transform_base = build_pixel_transform(False, args)
    transform_tokenadapt = build_pixel_transform(True, args)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            samples = codebook(samples).detach()
            samples = transform_color(samples)

            samples, samples_ta = samples.chunk(2)
            targets, targets_ta = targets.chunk(2)

            samples_ta = model.module.forward_conversion(samples_ta)

            samples_ta = transform_tokenadapt(samples_ta)

            if mixup_ta_fn is not None:
                samples_ta, targets_ta = mixup_ta_fn(samples_ta, targets_ta)

            samples_ta = model.module.forward_inversion(samples_ta)
            samples_ta = codebook(samples_ta).detach()

            samples = transform_base(samples)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)

            samples = torch.cat((samples, samples_ta), dim=0)
            targets = torch.cat((targets, targets_ta), dim=0)

            if args.bce_loss:
                targets = targets.gt(0.0).type(targets.dtype)

            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, codebook, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for samples, target in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            samples = codebook(samples)
            output = model(samples)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def tokenize_and_evaluate(data_loader, model, transform, codebook, tokenizer, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    tokenizer.eval()
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            encoded = tokenizer(images)
            tokens = utils.convert_to_tokens(encoded, codebook).to(device, non_blocking=True)
            if transform is not None:
                tokens = transform(tokens)

            output = model(tokens)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
