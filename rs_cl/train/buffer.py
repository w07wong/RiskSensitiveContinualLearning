import numpy as np


class Buffer():
    '''Adapted from https://github.com/aimagelab/mammoth.'''
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_seen_examples = 0
        # self.images = np.array([None for _ in range(buffer_size)])
        # self.labels = np.array([None for _ in range(buffer_size)])
        # self.task_labels = np.array([None for _ in range(buffer_size)])
        self.images = []
        self.labels = []
        self.task_labels = []

    def len(self):
        return min(self.num_seen_examples, self.buffer_size)

    def _reservoir(self, num_seen_examples, buffer_size):
        '''Reservoir sampling algorithm.'''
        if num_seen_examples < buffer_size:
            return num_seen_examples

        rand = np.random.randint(0, num_seen_examples + 1)
        if rand < buffer_size:
            return rand
        else:
            return -1

    def add_data(self, images, labels=None, task_labels=None):
        '''Adds data to the memory buffer according to the reservoir strategy.'''
        for i in range(len(images)):
            index = self._reservoir(self.num_seen_examples, self.buffer_size)
            if index >= 0 and index == self.num_seen_examples:
                self.images.append(np.asarray(images[i]))
                if labels is not None:
                    self.labels.append(np.asarray(labels[i]))
                if task_labels is not None:
                    self.task_labels.append(task_labels[i])
            elif index >= 0:
                self.images[i] = np.asarray(images[i])
                if labels is not None:
                    self.labels[i] = np.asarray(labels[i])
                if task_labels is not None:
                    self.task_labels.append(task_labels[i])
            self.num_seen_examples += 1

    def get_data(self, size):
        '''Randomly samples a batch of size items.'''
        if size > min(self.num_seen_examples, len(self.images)):
            size = min(self.num_seen_examples, len(self.images))

        choice = np.random.choice(min(self.num_seen_examples, len(self.images)),
                                  size=size, replace=False)

        # return self.images[choice], self.labels[choice], self.task_labels[choice]
        return [self.images[i] for i in choice], [self.labels[i] for i in choice], [self.task_labels[i] for i in choice]