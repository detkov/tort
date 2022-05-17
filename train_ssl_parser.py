import argparse
from os.path import basename, splitext

import yaml


def parse_ssl_args():
    config_parser = parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='./configs/test_ssl.yaml', type=str, 
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
    parser.add_argument('--resume', default='', type=str, 
                        help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--channels_last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='torch.jit.script the full model')
    
    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=6, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--masked_crop_scale', type=float, nargs='+', default=(0.2, 0.6),
        help="""Scale range of the mask to be erased from the corresponding crop (0.2, 0.6)""")

    # Tort's models parameters
    parser.add_argument('--apply_masking', default=False, type=bool)
    parser.add_argument('--use_sl_loss', default=False, type=bool)
    parser.add_argument('--use_rot_loss', default=False, type=bool)
    parser.add_argument('--rep_w', default=1, type=float)
    parser.add_argument('--sl_w', default=0, type=float)
    parser.add_argument('--rot_w', default=0, type=float)
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=bool,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=bool,
        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    parser.add_argument('--student_temp', default=0.1, type=float)
    parser.add_argument('--center_momentum', default=0.9, type=float)
    parser.add_argument('--sl_smoothing', default=None, type=float)
    parser.add_argument('--rot_prob', default=0.5, type=float)

    # Training/Optimization parameters
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, 
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=None, type=float, 
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', 
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.04,
                        help='weight decay (default: 2e-5)')
    parser.add_argument('--clip_grad', type=float, default=None, 
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip_mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--layer_decay', type=float, default=None,
                        help='layer-wise learning rate decay (default: None)')    
    parser.add_argument('--decay_epochs', type=float, default=None, 
                        help='epoch interval to decay LR')
    parser.add_argument('--patience_epochs', type=int, default=None, 
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay_rate', '--dr', type=float, default=None, 
                        help='LR decay rate (default: 0.1)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, 
                        help='LR scheduler (default: "cosine"')
    parser.add_argument("--lr", default=0.0005, type=float, 
                        help='learning rate (default: 0.0005 for bs=256)')
    parser.add_argument('--warmup_lr', type=float, default=0.001,
                        help='warmup learning rate (default: 0.001)')
    parser.add_argument('--min_lr', type=float, default=1e-6, 
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--start_epoch', default=None, type=int, 
                        help='manual epoch number (useful on restarts)')
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown_epochs', type=int, default=3, 
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--drop_path', type=float, default=0.1, help="stochastic depth rate")

    # Misc
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--log_interval', type=int, default=1, 
                        help='how many batches to wait before logging training status')
    parser.add_argument('--recovery_interval', type=int, default=0, 
                        help='how many batches to wait before writing recovery checkpoint')
    parser.add_argument('--checkpoint_hist', type=int, default=1, 
                        help='number of checkpoints to keep (default: 5)')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex_amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native_amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--output', default="./results", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--eval_metric', default='loss', type=str, 
                        help='Best metric (default: "top1"')
    
    #Logging
    parser.add_argument('--log_wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--experiment', default='tort', type=str, 
                        help='name of train experiment')
    parser.add_argument('--entity', default='detkov', type=str, 
                        help='wandb account')
    parser.add_argument('--experiments_group', default='test', type=str, 
                        help='experiments group for easy filtering')

    args_config, remaining = config_parser.parse_known_args()

    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    args.version = splitext(basename(args_config.config))[0]

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text
