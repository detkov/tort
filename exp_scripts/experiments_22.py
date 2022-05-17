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
        'use_rot_loss': True,

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
        'weight_decay': 0.04,

        'epochs': 100,
        'warmup_epochs': 10,
        'cooldown_epochs': 3,

        'dataset': 'Flowers102',
        'num_classes': 102,
        'data_dir': './',

        'log_wandb': True,
    }
    
    par_rot_w = [0.25, 0.5, 1, 2.5, 5, 10]
    par_rot_prob = [0.5, 1.0]
    
    par_combinations = list(product(par_rot_w, par_rot_prob))

    print(len(par_combinations))

    configs = []
    for combination in par_combinations:
        rot_w, rot_prob = combination
        configs.append({'rot_w': rot_w, 
                        'rot_prob': rot_prob, 
                        **common_params})

    for i, config in enumerate(configs):
        with open(join(configs_dir, f'{series}_{i:03}.yaml'), 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
