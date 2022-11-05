import numpy as np
import jax.numpy as jnp
import torchvision
from torchvision import transforms
from .utilities import Cast, FlattenAndCast

def CIFAR100(dataroot='data', train_aug=False):
    task1 = {'beaver', 'aquarium_fish', 'orchid', 'bottle', 'apple', 'clock', 'bed', 'bee', 'bear', 'bridge', 'cloud', 'camel', 'fox', 'crab', 'baby', 'crocodile', 'hamster', 'maple_tree', 'bicycle', 'lawn_mower'}
    task2 = {'dolphin', 'flatfish', 'poppy', 'bowl', 'mushroom', 'keyboard', 'chair', 'beetle', 'leopard', 'castle', 'forest', 'cattle', 'porcupine', 'lobster', 'boy', 'dinosaur', 'mouse', 'oak_tree', 'bus', 'rocket'}
    task3 = {'otter', 'ray', 'rose', 'can', 'orange', 'lamp', 'couch', 'butterfly', 'lion', 'house', 'mountain', 'chimpanzee', 'possum', 'snail', 'girl', 'lizard', 'rabbit', 'palm_tree', 'motorcycle', 'streetcar'}
    task4 = {'seal', 'shark', 'sunflower', 'cup', 'pear', 'telephone', 'table', 'caterpillar', 'tiger', 'road', 'plain', 'elephant', 'raccoon', 'spider', 'man', 'snake', 'shrew', 'pine_tree', 'pickup_truck', 'tank'}
    task5 = {'whale', 'trout', 'tulip', 'plate', 'sweet_pepper', 'television', 'wardrobe', 'cockroach', 'wolf', 'skyscraper', 'sea', 'kangaroo', 'skunk', 'worm', 'woman', 'turtle', 'squirrel', 'willow_tree', 'train', 'tractor'}
    tasks = [task1, task2, task3, task4, task5]

    stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
        Cast()
        # FlattenAndCast()
    ])
    train_transform = val_transform

    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
            Cast()
            # FlattenAndCast()
        ])

    train_data = torchvision.datasets.CIFAR100(
        root=dataroot,
        download=True,
        transform=train_transform
    )
    train_images = [[] for _ in range(len(tasks))]
    train_labels = [[] for _ in range(len(tasks))]
    for image, label in train_data:
        class_name = train_data.classes[label]
        for i, task in enumerate(tasks):
            if class_name in task:
                train_images[i].append(image)
                train_labels[i].append(sparse2coarse(label))

    val_data = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        transform=val_transform
    )
    val_images = [[] for _ in range(len(tasks))]
    val_labels = [[] for _ in range(len(tasks))]
    for image, label in val_data:
        class_name = val_data.classes[label]
        for i, task in enumerate(tasks):
            if class_name in task:
                val_images[i].append(image)
                val_labels[i].append(sparse2coarse(label))

    return jnp.array(train_images), jnp.array(train_labels), jnp.array(val_images), jnp.array(val_labels) 

def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5, 
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17, 
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13, 
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    return coarse_labels[targets]