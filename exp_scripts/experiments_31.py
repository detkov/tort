import yaml
import os
from os.path import join
from itertools import product


if __name__ == '__main__':
    series = os.path.basename(__file__).split('.')[0].split('_')[-1]
    configs_dir = join('configs', series)
    os.makedirs(configs_dir, exist_ok=True)
    
    common_params = {
        'model': 'vit_base_patch16_224',
        'pretrained': False,
        'amp': True,
        'experiments_group': series,

        'momentum_teacher': 0.9998,
        'out_dim': 1024,
        'warmup_teacher_temp': 0.04,
        'teacher_temp': 0.04,
        'warmup_teacher_temp_epochs': 0,
        'local_crops_number': 6,

        'apply_masking': True,
        'use_sl_loss': True,
        'use_rot_loss': True,
        'rep_w': 1,
        'sl_w': 2.5,
        'rot_w': 1,
        'rot_prob': 0.5,
        'sl_smoothing': None,

        'batch_size_per_gpu': 32,
        'channels_last': True,
        'pin_mem': True,
        'norm_last_layer': True,

        'sched': 'cosine',
        'lr': 0.0005,
        'min_lr': 0.000001,
        'warmup_lr': 0.00000001,
        'opt': 'lookahead_adamw',
        'clip_mode': 'norm',
        'clip_grad': 3.0,
        # 'weight_decay': 0.04,

        'epochs': 100,
        'warmup_epochs': 10,
        'cooldown_epochs': 3,

        'dataset': 'Flowers102',
        'num_classes': 102,
        'data_dir': './',

        'log_wandb': True,
    }
    
    par_weight_decay = [0, 0.1, 0.2, 0.3, 0.4]

    configs = []
    for weight_decay in par_weight_decay:
        configs.append({'weight_decay': weight_decay, **common_params})

    for i, config in enumerate(configs):
        with open(join(configs_dir, f'{series}_{i:03}.yaml'), 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
