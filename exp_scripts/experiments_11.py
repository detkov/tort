import yaml
import os
from os.path import join
from itertools import product


if __name__ == '__main__':
    configs_dir = join('configs', '11')
    os.makedirs(configs_dir, exist_ok=True)
    
    common_params = {
        'model': 'vit_base_patch16_224',
        # 'pretrained': False,
        'amp': True,

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

        'sched': 'cosine',
        # 'lr': 0.0005,
        'min_lr': 0.000001,
        'warmup_lr': 0,
        # 'opt': 'lookahead_adamw',
        'clip_mode': 'norm',
        'clip_grad': 3.0,
        # 'weight_decay': 0.04,

        'epochs': 100,
        'warmup_epochs': 10,
        'cooldown_epochs': 3,

        'dataset': 'Flowers102',
        'num_classes': 0,
        'data_dir': './',

        'log_wandb': True,
    }

    par_pretrained = [False, True]
    par_lr = [0.001, 0.0005, 0.0001, 0.00001]
    par_opt = ['lookahead_adamw', 'lookahead_sgd']
    par_weight_decay = [0.0, 0.1]
    
    par_combinations = list(product(par_pretrained, par_lr, par_opt, par_weight_decay))

    print(len(par_combinations))

    configs = []
    for combination in par_combinations:
        pretrained, lr, opt, weight_decay = combination
        configs.append({'pretrained': pretrained, 
                        'lr': lr, 
                        'opt': opt, 
                        'weight_decay': weight_decay, 
                        **common_params})
    
    for i, config in enumerate(configs):
        with open(join(configs_dir, f'{i:03}.yaml'), 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
