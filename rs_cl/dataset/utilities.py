import torch
import torch.utils.data as data
from os import path
import numpy as np
import jax.numpy as jnp


# Dataset utilities (from https://github.com/GT-RIPL/Continual-Learning-Benchmark/blob/master/dataloaders/wrapper.py)


class CacheClassLabel(data.Dataset):
    """
    A dataset wrapper that has a quick access to all labels of data.
    """
    def __init__(self, dataset):
        super(CacheClassLabel, self).__init__()
        self.dataset = dataset
        self.labels = torch.LongTensor(len(dataset)).fill_(-1)
        label_cache_filename = path.join(dataset.root, str(type(dataset)) + '_' + str(len(dataset)) + '.pth')
        if path.exists(label_cache_filename):
            self.labels = torch.load(label_cache_filename)
        else:
            for i, data in enumerate(dataset):
                self.labels[i] = data[1]
            torch.save(self.labels, label_cache_filename)
        self.number_classes = len(torch.unique(self.labels))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img,target = self.dataset[index]
        return img, target


class AppendName(data.Dataset):
    """
    A dataset wrapper that also return the name of the dataset/task
    """
    def __init__(self, dataset, name, first_class_ind=0):
        super(AppendName,self).__init__()
        self.dataset = dataset
        self.name = name
        self.first_class_ind = first_class_ind  # For remapping the class index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img,target = self.dataset[index]
        target = target + self.first_class_ind
        return img, target, self.name


class Subclass(data.Dataset):
    """
    A dataset wrapper that return the task name and remove the offset of labels (Let the labels start from 0)
    """
    def __init__(self, dataset, class_list, remap=True):
        '''
        :param dataset: (CacheClassLabel)
        :param class_list: (list) A list of integers
        :param remap: (bool) Ex: remap class [2,4,6 ...] to [0,1,2 ...]
        '''
        super(Subclass,self).__init__()
        assert isinstance(dataset, CacheClassLabel), 'dataset must be wrapped by CacheClassLabel'
        self.dataset = dataset
        self.class_list = class_list
        self.remap = remap
        self.indices = []
        for c in class_list:
            self.indices.extend((dataset.labels==c).nonzero().flatten().tolist())
        if remap:
            self.class_mapping = {c: i for i, c in enumerate(class_list)}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img,target = self.dataset[self.indices[index]]
        if self.remap:
            raw_target = target.item() if isinstance(target,torch.Tensor) else target
            target = self.class_mapping[raw_target]
        return img, target

class FlattenAndCast(object):
    """
    Transformation to flatten an image and cast it to jnp float type.
    """
    def __call__(self, image):
        return np.ravel(np.array(image, dtype=jnp.float32))


class Cast(object):
    """
    Transformation to cast an image to jnp float type.
    """
    def __call__(self, image):
        return np.array(image, dtype=jnp.float32)
