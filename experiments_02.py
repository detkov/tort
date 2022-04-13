import yaml
import os
from os.path import join


if __name__ == '__main__':
    configs_dir = 'configs'
    os.makedirs(configs_dir, exist_ok=True)

    dataset_names = ['FGVCAircraft', 'Flowers102', 'Food101', 'StanfordCars', 'OxfordIIITPet']
    dataset_num_classes = [102, 103, 101, 196, 37]

    common_params = {
        'model': 'vit_base_patch16_224_dino',
        'sched': 'cosine',
        'smoothing': 0,
        'drop': 0.1,
        'warmup_epochs': 3,
        'warmup_lr': 1e-6,
        'batch_size': 64,
        'workers': 8,
        'pin_mem': True,
        'channels_last': True,
        'log_wandb': True,
    }

    finetune_params = {
        'pretrained': True,
        'opt': 'sgd',
        'momentum': 0.9,
        'weight_decay': 0.,
        'lr': 0.003,
    }

    configs = []
    configs_names = []
    for dataset_name, num_classes in zip(dataset_names, dataset_num_classes):
        configs_names.append(f'{dataset_name}-dino-finetune.yaml')
        configs.append({'dataset': dataset_name, 'num_classes': num_classes, **common_params, **finetune_params})
    
    for config, config_name in zip(configs, configs_names):
        with open(join(configs_dir, config_name), 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
