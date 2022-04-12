import yaml


if __name__ == '__main__':
    dataset_names = ['FGVCAircraft', 'Flowers102', 'Food101', 'StanfordCars', 'OxfordIIITPet']
    dataset_num_classes = [102, 102, 101, 196, 37]

    common_params = {
        'model': 'vit_base_patch16_224',

        'opt': 'sgd',
        'momentum': 0.9,
        'weight_decay': 0,
        'sched': 'cosine',
        'smoothing': 0,

        'batch_size': 32,
        'workers': 8,
        'pin_mem': True,

        'log_wandb': True,
    }

    train_params = {
        'pretrained': False,
        'opt': 'adamw',
        'opt-betas': [0.9, 0.999],
        'weight_decay': 0.1,
        'lr': 0.003,
        'warmup_lr': 1e-6,
        'warmup-epochs': 1,
        'drop': 0.1,
    }

    finetune_params = {
        'pretrained': True,
        'opt': 'sgd',
        'momentum': 0.9,
        'weight_decay': 0,
        'lr': 0.003,
        'warmup_lr': 1e-6,
        'warmup-epochs': 1,
        'drop': 0.1,
    }    
    