import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from os.path import join

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from timm.models import create_model, model_parameters, resume_checkpoint
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils.checkpoint_saver import CheckpointSaver
from timm.utils.clip_grad import dispatch_clip_grad
from timm.utils.cuda import ApexScaler, NativeScaler
from timm.utils.log import setup_default_logging
from timm.utils.metrics import AverageMeter
from timm.utils.random import random_seed
from timm.utils.summary import update_summary
from torchvision import transforms
import wandb

from dataset import create_dataset
from tort import DataAugmentationTort, MultiCropWrapper, TortHead, CESoftmaxLoss
from tort_utils import cosine_scheduler
from train_ssl_parser import parse_ssl_args

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')


def main():
    setup_default_logging()
    args, args_text = parse_ssl_args()
    
    if args.log_wandb:
        wandb.init(name=args.version, project=args.experiment, 
                   entity=args.entity, config=args)
    
    args.device = 'cuda'

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    random_seed(args.seed)

    student = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=0,
        drop_path_rate=args.drop_path,
        scriptable=args.torchscript,
    )
    embed_dim = student.embed_dim
    teacher = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=0,
        scriptable=args.torchscript,
    )

    student = MultiCropWrapper(student, TortHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head, 
                                                 norm_last_layer=args.norm_last_layer))
    teacher = MultiCropWrapper(teacher, TortHead(embed_dim, args.out_dim, args.use_bn_in_head))


    student = student.to(args.device)
    teacher = teacher.to(args.device)
    if args.channels_last:
        student = student.to(memory_format=torch.channels_last)
        teacher = teacher.to(memory_format=torch.channels_last)
    
    for p in teacher.parameters():
        p.requires_grad = False

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        student = torch.jit.script(student)
        teacher = torch.jit.script(teacher)

    optimizer = create_optimizer_v2(student, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        student, optimizer = amp.initialize(student, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            student, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            # log_info=args.local_rank == 0)
            log_info=True)

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    _logger.info('Scheduled epochs: {}'.format(num_epochs))

    dataset_train = create_dataset(args.dataset, args.data_dir, is_training=True)
    dataset_valid = create_dataset(args.dataset, args.data_dir, is_training=False)
    dataset_train.transform = DataAugmentationTort(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset_valid.transform = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size_per_gpu,
        num_workers=args.workers, shuffle=True, pin_memory=True, drop_last=True)
    loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size_per_gpu,
        num_workers=args.workers, shuffle=False, pin_memory=True, drop_last=False)

    train_loss_fn = CESoftmaxLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        num_epochs,
    ).to(args.device)

    momentum_scheduler = cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(loader_train))

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    exp_name = '-'.join([args.version, datetime.now().strftime("%Y%m%d-%H%M%S"),])
    output_dir = join(args.output, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    decreasing = True if eval_metric == 'loss' else False
    saver = CheckpointSaver(
        model=student, optimizer=optimizer, args=args, amp_scaler=loss_scaler,
        checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
    with open(join(output_dir, 'args_ssl.yaml'), 'w') as f:
        f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            train_metrics = train_one_epoch_ssl(
                epoch, student, teacher, loader_train, optimizer, train_loss_fn, args, 
                momentum_scheduler, lr_scheduler=lr_scheduler, saver=saver, 
                amp_autocast=amp_autocast, loss_scaler=loss_scaler)

            valid_metrics = valid_one_epoch_ssl(epoch, student, loader_valid, args, amp_autocast)

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1, train_metrics[eval_metric])

            if output_dir is not None:
                update_summary(epoch, train_metrics, valid_metrics, join(output_dir, 'summary.csv'),
                               write_header=best_metric is None, log_wandb=args.log_wandb)

            if saver is not None:
                save_metric = train_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info(f'*** Best metric: {best_metric} (epoch {best_epoch})')


def train_one_epoch_ssl(
        epoch, student, teacher, loader, optimizer, loss_fn, args, momentum_scheduler,
        lr_scheduler=None, saver=None, amp_autocast=suppress, loss_scaler=None):

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    student.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, _) in enumerate(loader):
        last_batch = batch_idx == last_idx
        input = [el.to(args.device) for el in input]
        if args.channels_last:
            input = [el.contiguous(memory_format=torch.channels_last) for el in input]

        with amp_autocast():
            teacher_output = teacher(input[:2])  # only the 2 global views pass through the teacher
            student_output = student(input)
            loss = loss_fn(student_output, teacher_output, epoch)

        losses_m.update(loss.item())

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(student, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(student, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_scheduler[batch_idx]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            _logger.info(
                'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'LR: {lr:.3e}  '.format(
                    epoch,
                    batch_idx, len(loader),
                    100. * batch_idx / last_idx,
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=input[0].size(0) / batch_time_m.val,
                    rate_avg=input[0].size(0) / batch_time_m.avg,
                    lr=lr))

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for
    _logger.info('Train: {} Time: {batch_time.sum:.3f}s'.format(epoch, batch_time=batch_time_m))

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])

def valid_one_epoch_ssl(epoch, student, loader, args, amp_autocast=suppress):
    start = time.time()
    student.eval()

    features, labels = None, None
    with torch.no_grad():
        for batch_idx, (input, batch_labels) in enumerate(loader):
            input = input.to(args.device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                student_output = student(input).cpu()

            if features is None:
                features = torch.zeros(len(loader.dataset), student_output.shape[-1])
            if labels is None:
                labels = torch.zeros(len(loader.dataset))

            start_idx = args.batch_size_per_gpu * batch_idx
            end_idx = args.batch_size_per_gpu * (batch_idx+1)
            features[start_idx:end_idx] = student_output
            labels[start_idx:end_idx] = batch_labels

            torch.cuda.synchronize()
            # end for
        # end no_grad

    features = features.view(-1, features.shape[-1]).numpy()
    labels = labels.view(-1).numpy()
    cls = KNeighborsClassifier(n_neighbors=20, metric='cosine').fit(features, labels)
    acc1 = 100 * np.mean(cross_val_score(cls, features, labels))

    _logger.info(f'Valid: {epoch} Acc@1: {acc1:>7.4f} Time: {time.time() - start:.3f}s')

    return OrderedDict([('top1', acc1)])


if __name__ == '__main__':
    main()
