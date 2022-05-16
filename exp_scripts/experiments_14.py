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
        # 'out_dim': 2048,
        'warmup_teacher_temp': 0.04,
        'teacher_temp': 0.04,
        'warmup_teacher_temp_epochs': 0,
        'local_crops_number': 6,

        'batch_size_per_gpu': 32,
        'channels_last': True,
        'pin_mem': True,
        'norm_last_layer': True,

        'sched': 'cosine',
        'lr': 0.0005,
        'min_lr': 0.000001,
        'warmup_lr': 0,
        'opt': 'lookahead_adamw',
        'clip_mode': 'norm',
        'clip_grad': 3.0,
        'weight_decay': 0.0,

        'epochs': 300,
        'warmup_epochs': 5,
        'cooldown_epochs': 3,

        'dataset': 'Flowers102',
        'num_classes': 0,
        'data_dir': './',

        'log_wandb': True,
    }
    
    par_out_dim = [128, 256, 512, 1024, 2048]

    configs = []
    for out_dim in par_out_dim:
        configs.append({'out_dim': out_dim, **common_params})

    for i, config in enumerate(configs):
        with open(join(configs_dir, f'{series}_{i:03}.yaml'), 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
