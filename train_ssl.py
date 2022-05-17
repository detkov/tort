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
from tort_head import MultiCropWrapper, TortHead
from tort_aug import TortAugmenter, rand_rot
from tort_loss import TortLoss
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
                                                 norm_last_layer=args.norm_last_layer, 
                                                 num_classes=args.num_classes, is_rot_head=args.use_rot_loss))
    teacher = MultiCropWrapper(teacher, TortHead(embed_dim, args.out_dim, args.use_bn_in_head,
                                                 num_classes=args.num_classes, is_rot_head=args.use_rot_loss))


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

    masked_crops_number = int(args.apply_masking)
    global_crops_number = 2

    dataset_train = create_dataset(args.dataset, args.data_dir, is_training=True)
    dataset_train.transform = TortAugmenter(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args.apply_masking,
        args.masked_crop_scale,
    )
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size_per_gpu,
        num_workers=args.workers, shuffle=True, pin_memory=True, drop_last=True)

    dataset_valid = create_dataset(args.dataset, args.data_dir, is_training=False)
    dataset_valid.transform = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size_per_gpu,
        num_workers=args.workers, shuffle=False, pin_memory=True, drop_last=False)

    train_loss_fn = TortLoss(args.use_sl_loss, args.use_rot_loss, 
        args.device, args.rep_w, args.sl_w, args.rot_w, 
        args.out_dim, global_crops_number, masked_crops_number, args.local_crops_number, args.warmup_teacher_temp,
        args.teacher_temp, args.warmup_teacher_temp_epochs, num_epochs, args.student_temp,
        args.center_momentum, args.sl_smoothing)

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
                epoch, student, teacher, loader_train, global_crops_number,
                optimizer, train_loss_fn, args, 
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
        epoch, student, teacher, loader, global_crops_number, 
        optimizer, loss_fn, args, momentum_scheduler,
        lr_scheduler=None, saver=None, amp_autocast=suppress, loss_scaler=None):

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    use_rep_losses_m = args.use_sl_loss or args.use_rot_loss
    if use_rep_losses_m:
        rep_losses_m = AverageMeter()

    student.train()

    n_gm_crops = global_crops_number + int(args.apply_masking)
    use_sl_for, use_rot_for = None, None
    if args.use_sl_loss:
        use_sl_for = n_gm_crops * args.batch_size_per_gpu
        sl_losses_m = AverageMeter()
    if args.use_rot_loss:
        use_rot_for = n_gm_crops * args.batch_size_per_gpu
        rot_losses_m = AverageMeter()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, sl_labels) in enumerate(loader):
        last_batch = batch_idx == last_idx
        input = [el.to(args.device) for el in input]

        if args.use_sl_loss:
            if args.apply_masking:
                sl_labels = torch.cat([sl_labels, sl_labels, sl_labels])
            else:
                sl_labels = torch.cat([sl_labels, sl_labels])
            sl_labels = sl_labels.long().to(args.device)

        rot_labels = None
        if args.use_rot_loss:
            input[0], rots_g1 = rand_rot(input[0], args.rot_prob)
            input[1], rots_g2 = rand_rot(input[1], args.rot_prob)
            rot_labels = [*rots_g1, *rots_g2]
            if args.apply_masking:
                input[2], rots_m = rand_rot(input[2], args.rot_prob)
                rot_labels.extend(rots_m)
            rot_labels = torch.tensor(rot_labels).long().to(args.device)
        
        if args.channels_last:
            input = [el.contiguous(memory_format=torch.channels_last) for el in input]

        with amp_autocast():
            teacher_output, _, _ = teacher(input[:n_gm_crops], use_sl_for=None, use_rot_for=None)
            student_output, sl_pred, rot_pred = student(input, use_sl_for=use_sl_for, use_rot_for=use_rot_for)
            loss, rep_loss, sl_loss, rot_loss = loss_fn(student_output, teacher_output, epoch, 
                                                        sl_pred, sl_labels, 
                                                        rot_pred, rot_labels)

        losses_m.update(loss.item())
        if use_rep_losses_m:
            rep_losses_m.update(rep_loss.item())
        if args.use_sl_loss:
            sl_losses_m.update(sl_loss.item())
        if args.use_rot_loss:
            rot_losses_m.update(rot_loss.item())

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

        # torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            log_str = (f'Train: {epoch} [{batch_idx:>4d}/{len(loader)} ({100. * batch_idx / last_idx:>3.0f}%)]  ' +
                       f'Loss: {losses_m.val:#.4g} ({losses_m.avg:#.3g})  ' + 
                      (f'L_rep: {rep_losses_m.val:#.4g}  ' if use_rep_losses_m else '') +
                      (f'L_sl: {sl_losses_m.val:#.4g}  ' if args.use_sl_loss else '') +
                      (f'L_rot: {rot_losses_m.val:#.4g}  ' if args.use_rot_loss else '') +
                       f'Time: {batch_time_m.val:.3f}s, {input[0].size(0) / batch_time_m.val:>7.2f}/s  ' +
                       f'LR: {lr:.3e}  '
                )
            _logger.info(log_str)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for
    _logger.info('Train: {}  Time: {batch_time.sum:.3f}s'.format(epoch, batch_time=batch_time_m))

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    metrics = [('loss', losses_m.avg)]
    if use_rep_losses_m:
        metrics.append(('loss_rep', rep_losses_m.avg))
    if args.use_sl_loss:
        metrics.append(('loss_sl', sl_losses_m.avg))
    if args.use_rot_loss:
        metrics.append(('loss_rot', rot_losses_m.avg))
    return OrderedDict(metrics)

def valid_one_epoch_ssl(epoch, student, loader, args, amp_autocast=suppress):
    start = time.time()
    student.eval()

    features, labels = None, None
    if args.use_sl_loss:
        prediction = None

    with torch.no_grad():
        for batch_idx, (input, sl_labels) in enumerate(loader):
            input = input.to(args.device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                student_output, sl_pred, _ = student(input, use_sl_for=args.batch_size_per_gpu, use_rot_for=False)
                student_output = student_output.cpu()
                sl_pred = torch.softmax(sl_pred, 1).argmax(1).cpu()

            if features is None:
                features = torch.zeros(len(loader.dataset), student_output.shape[-1])
            if labels is None:
                labels = torch.zeros(len(loader.dataset))
            if args.use_sl_loss:
                if prediction is None:
                    prediction = torch.zeros(len(loader.dataset))

            start_idx = args.batch_size_per_gpu * batch_idx
            end_idx = args.batch_size_per_gpu * (batch_idx+1)
            features[start_idx:end_idx] = student_output
            labels[start_idx:end_idx] = sl_labels
            if args.use_sl_loss:
                prediction[start_idx:end_idx] = sl_pred

            # torch.cuda.synchronize()
            # end for
        # end no_grad

    features = features.view(-1, features.shape[-1]).numpy()
    labels = labels.view(-1).numpy()
    cls = KNeighborsClassifier(n_neighbors=20, metric='cosine').fit(features, labels)
    acc1_knn = 100 * np.mean(cross_val_score(cls, features, labels))

    from sklearn.metrics import accuracy_score
    if args.use_sl_loss:
        prediction = prediction.view(-1).numpy()
        acc1_sl = 100 * accuracy_score(labels, prediction)

    _logger.info(f'Valid: {epoch}  Time: {time.time() - start:.3f}s  k-NN Acc@1: {acc1_knn:>7.4f} {f" SL Acc@1: {acc1_sl:>7.4f} " if args.use_sl_loss else ""}')

    metrics = [('top1_knn', acc1_knn)]
    if args.use_sl_loss:
        metrics.append(('top1', acc1_sl))
    return OrderedDict(metrics)


if __name__ == '__main__':
    main()
