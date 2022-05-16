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
        'out_dim': 2048,
        'warmup_teacher_temp': 0.04,
        'teacher_temp': 0.04,
        'warmup_teacher_temp_epochs': 0,
        'local_crops_number': 6,

        'batch_size_per_gpu': 32,
        'channels_last': True,
        'pin_mem': True,
        'norm_last_layer': True,

        # 'sched': 'cosine',
        'lr': 0.0005,
        'min_lr': 0.0000001,
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
    
    configs = []
    configs.append({'sched': 'cosine', **common_params})
    configs.append({'sched': 'tanh', **common_params})
    configs.append({'sched': 'step',
                    'decay_epochs': common_params['epochs'] // 2,
                    'decay_rate': 0.1,
                    **common_params})
    configs.append({'sched': 'plateau',
                    'patience_epochs': 10,
                    'decay_rate': 0.5,
                    **common_params})
    configs.append({'sched': 'poly',
                    'decay_rate': 0.5,
                    **common_params})

    for i, config in enumerate(configs):
        with open(join(configs_dir, f'{series}_{i:03}.yaml'), 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
