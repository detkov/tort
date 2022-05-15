import argparse
from os.path import basename, splitext

import yaml


def parse_sl_args():
    config_parser = parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='./configs/test_sl.yaml', type=str, 
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument('--data_dir', default='./', help='path to dataset')
    parser.add_argument('--dataset', default='test', help='dataset name')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str,
                        help='Name of model to train (default: "vit_base_patch16_224"')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--initial_checkpoint', default='', type=str, 
                        help='Initialize model from this checkpoint (default: none)')
    parser.add_argument('--resume', default='', type=str, 
                        help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--no_resume_opt', action='store_true', default=False,
                        help='prevent resume of optimizer state when resuming model')
    parser.add_argument('--num_classes', type=int, default=None, 
                        help='number of label classes (Model default if None)')
    parser.add_argument('--gp', default=None, type=str, 
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--img_size', type=int, default=None, 
                        help='Image patch size (default: None => model default)')
    parser.add_argument('--input_size', default=None, nargs=3, type=int,
                        help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    parser.add_argument('--crop_pct', default=None, type=float,
                        help='Input image center crop percent (for validation only)')
    parser.add_argument('--mean', type=float, nargs='+', default=None, 
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, 
                        help='Override std deviation of dataset')
    parser.add_argument('--interpolation', default='', type=str, 
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('-b', '--batch_size', type=int, default=128, 
                        help='Input batch size for training (default: 128)')
    parser.add_argument('-vb', '--validation_batch_size', type=int, default=None, 
                        help='Validation batch size override (default: None)')
    parser.add_argument('--channels_last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='torch.jit.script the full model')
    parser.add_argument('--grad_checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing through model blocks/stages')

    # Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, 
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--opt_eps', default=None, type=float, 
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', 
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=2e-5,
                        help='weight decay (default: 2e-5)')
    parser.add_argument('--clip_grad', type=float, default=None, 
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip_mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--layer_decay', type=float, default=None,
                        help='layer-wise learning rate decay (default: None)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, 
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=0.05, 
                        help='learning rate (default: 0.05)')
    parser.add_argument('--lr_noise', type=float, nargs='+', default=None, 
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr_noise_pct', type=float, default=0.67, 
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr_noise_std', type=float, default=1.0, 
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr_cycle_mul', type=float, default=1.0, 
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr_cycle_decay', type=float, default=0.5, 
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr_cycle_limit', type=int, default=1, 
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr_k_decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup_lr', type=float, default=0.0001, 
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min_lr', type=float, default=1e-6, 
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--epoch_repeats', type=float, default=0., 
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start_epoch', default=None, type=int, 
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay_epochs', type=float, default=100, 
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup_epochs', type=int, default=3, 
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown_epochs', type=int, default=3, 
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience_epochs', type=int, default=10, 
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, 
                        help='LR decay rate (default: 0.1)')

    # Augmentation & regularization parameters
    parser.add_argument('--no_aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], 
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], 
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color_jitter', type=float, default=0.4, 
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default=None, 
                        help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    parser.add_argument('--aug_splits', type=int, default=0,
                        help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    parser.add_argument('--jsd_loss', action='store_true', default=False,
                        help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    parser.add_argument('--bce_loss', action='store_true', default=False,
                        help='Enable BCE loss w/ Mixup/CutMix use.')
    parser.add_argument('--bce_target-thresh', type=float, default=None,
                        help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    parser.add_argument('--reprob', type=float, default=0., 
                        help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup_off_epoch', default=0, type=int, 
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    parser.add_argument('--drop', type=float, default=0.0, 
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_connect', type=float, default=None, 
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop_path', type=float, default=None, 
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop_block', type=float, default=None, 
                        help='Drop block rate (default: None)')

    # Model Exponential Moving Average
    parser.add_argument('--model_ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model_ema_decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')

    # Misc
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed (default: 42)')
    parser.add_argument('--worker_seeding', type=str, default='all',
                        help='worker seed mode (default: all)')
    parser.add_argument('--log_interval', type=int, default=100, 
                        help='how many batches to wait before logging training status')
    parser.add_argument('--recovery_interval', type=int, default=0, 
                        help='how many batches to wait before writing recovery checkpoint')
    parser.add_argument('--checkpoint_hist', type=int, default=5, 
                        help='number of checkpoints to keep (default: 5)')
    parser.add_argument('-j', '--workers', type=int, default=8, 
                        help='how many training processes to use (default: 4)')
    parser.add_argument('--save_images', action='store_true', default=False,
                        help='save images of input bathes every log interval for debugging')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex_amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native_amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--no_ddp_bb', action='store_true', default=False,
                        help='Force broadcast buffers for native DDP to off.')
    parser.add_argument('--pin_mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--output', default='./results', type=str, 
                        help='path to output folder (default: none, current dir)')
    parser.add_argument('--eval_metric', default='top1', type=str, 
                        help='Best metric (default: "top1"')
    parser.add_argument('--tta', type=int, default=0, 
                        help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--use_multi_epochs_loader', action='store_true', default=False,
                        help='use the multi-epochs-loader to save time at the beginning of every epoch')
    
    #Logging
    parser.add_argument('--log_wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--experiment', default='tort', type=str, 
                        help='name of train experiment')
    parser.add_argument('--entity', default='detkov', type=str, 
                        help='wandb account')

    args_config, remaining = config_parser.parse_known_args()

    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    args.version = splitext(basename(args_config.config))[0]

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text
