from dataset.dataset import get_task_names_and_splits
import torch
import numpy as np
import jax.numpy as jnp

def MNIST():
    offsets = [0, 2, 4, 6, 8]
    train_dataset = []
    val_dataset = []
    train_dataset_splits, val_dataset_splits, task_names = get_task_names_and_splits()
    for i in range(len(task_names)):
        train_name = task_names[i]
        train_loader = torch.utils.data.DataLoader(train_dataset_splits[train_name],
                                                        batch_size=1,
                                                        shuffle=True,
                                                        collate_fn=numpy_collate,
                                                        num_workers=3)
        val_loader = torch.utils.data.DataLoader(val_dataset_splits[train_name],
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=3)

        # Process images and labels
        train_images, train_labels = get_images_and_labels(train_loader.dataset,
                                                        offsets[i])
        val_images, val_labels = get_images_and_labels(val_loader.dataset,
                                                    offsets[i])

        # Add images and labels to respective dataset lists
        train_dataset.append([train_images, train_labels])
        val_dataset.append([val_images, val_labels])

    return train_dataset, val_dataset


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def get_images_and_labels(dataset, label_offset):
    images = []
    labels = []
    for j in range(len(dataset)):
        images.append(dataset[j][0])
        labels.append(dataset[j][1]- label_offset)
    images = jnp.array(images)
    labels = jnp.array(labels)
    return images, labels