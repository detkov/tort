from torchvision.datasets import Flowers102, FGVCAircraft, StanfordCars, OxfordIIITPet, FakeData


dataset_config = {
    'Flowers102': [Flowers102, ['train', 'val']],
    'FGVCAircraft': [FGVCAircraft, ['train', 'val']],
    'StanfordCars': [StanfordCars, ['train', 'test']],
    'OxfordIIITPet': [OxfordIIITPet, ['trainval', 'test']],
}

def create_dataset(name, root, is_training):
    if name == 'test':
        return FakeData(64, (3, 224, 224), 10)
    if not name in dataset_config.keys():
        raise Exception('Unsupported dataset name.')
    dataset = dataset_config[name][0]
    split = dataset_config[name][1][0] if is_training else dataset_config[name][1][1]

    return dataset(root=root, split=split, download=True)
