import numpy as np
import jax.numpy as jnp


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