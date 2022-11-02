import torchvision
from torchvision import transforms

from .utilities import CacheClassLabel, FlattenAndCast, AppendName, Subclass


def MNIST(dataroot, train_aug=False):
    """
    Load MNIST dataset (from https://github.com/GT-RIPL/Continual-Learning-Benchmark/blob/master/dataloaders/base.py)
    """
    # Add padding to make 32x32
    # normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for 28x28
    normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    val_transform = transforms.Compose([
        transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        normalize,
        FlattenAndCast()
    ])
    train_transform = val_transform

    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
            FlattenAndCast()
        ])

    train_dataset = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def SplitGen(train_dataset, val_dataset, first_split_sz=2, other_split_sz=2, rand_split=False, remap_class=False):
    """
    Split dataset (from https://github.com/GT-RIPL/Continual-Learning-Benchmark/blob/master/dataloaders/datasetGen.py)
    Generate the dataset splits based on the labels.

    train_dataset: (torch.utils.data.dataset)
    val_dataset: (torch.utils.data.dataset)
    first_split_sz: (int)
    other_split_sz: (int)
    rand_split: (bool) Randomize the set of label in each split
    remap_class: (bool) Ex: remap classes in a split from [2,4,6 ...] to [0,1,2 ...]

    Returns train_loaders {task_name:loader}, val_loaders {task_name:loader}, out_dim {task_name:num_classes}
    """
    assert train_dataset.number_classes==val_dataset.number_classes,'Train/Val has different number of classes'
    num_classes =  train_dataset.number_classes

    # Calculate the boundary index of classes for splits
    # Ex: [0,2,4,6,8,10] or [0,50,60,70,80,90,100]
    split_boundaries = [0, first_split_sz]
    while split_boundaries[-1]<num_classes:
        split_boundaries.append(split_boundaries[-1]+other_split_sz)
    assert split_boundaries[-1]==num_classes,'Invalid split size'

    # Assign classes to each splits
    # Create the dict: {split_name1:[2,6,7], split_name2:[0,3,9], ...}
    if not rand_split:
        class_lists = {str(i):list(range(split_boundaries[i-1],split_boundaries[i])) for i in range(1,len(split_boundaries))}
    else:
        randseq = torch.randperm(num_classes)
        class_lists = {str(i):randseq[list(range(split_boundaries[i-1],split_boundaries[i]))].tolist() for i in range(1,len(split_boundaries))}

    # Generate the dicts of splits
    # Ex: {split_name1:dataset_split1, split_name2:dataset_split2, ...}
    train_dataset_splits = {}
    val_dataset_splits = {}
    task_output_space = {}
    for name,class_list in class_lists.items():
        train_dataset_splits[name] = AppendName(Subclass(train_dataset, class_list, remap_class), name)
        val_dataset_splits[name] = AppendName(Subclass(val_dataset, class_list, remap_class), name)
        task_output_space[name] = len(class_list)

    return train_dataset_splits, val_dataset_splits, task_output_space


def get_task_names_and_splits():
    train_dataset, val_dataset = MNIST(dataroot='data', train_aug=True)

    train_dataset_splits, val_dataset_splits, task_output_space = SplitGen(train_dataset,
                                                                        val_dataset,
                                                                        first_split_sz=2,
                                                                        other_split_sz=2,
                                                                        rand_split=False,
                                                                        remap_class=False)
    task_names = sorted(list(task_output_space.keys()), key=int)
    return train_dataset_splits, val_dataset_splits, task_names