import numpy as np


class Buffer():
    def __init__(self, buffer_max_size):
        '''
        Args:
            buffer_size: maximum number of elements buffer can hold
        '''
        self.buffer_max_size = buffer_max_size
        self.buffer_size = 0
        self.num_seen_examples = 0
        self.buffer_task_ids = []
        self.buffer_data_ids = []

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

    def add_data(self, indices, task_id):
        '''Add data to the buffer with reservoir sampling.'''
        # Calculate available space in buffer
        available_space = self.buffer_max_size - self.buffer_size
        if available_space > 0:
            # Fill buffer to brim with available indices
            self.buffer_task_ids.extend([task_id for _ in range(available_space)])
            self.buffer_data_ids.extend([indices[i] for i in range(available_space)])
            self.buffer_size += available_space
            self.num_seen_examples += available_space

        # Only called when the buffer is full and we have elements in indices not yet added.
        for elem_idx in range(available_space, len(indices)):
            candidate_index = self._reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if candidate_index > -1:
                self.buffer_task_ids[candidate_index] = task_id
                self.buffer_data_ids[candidate_index] = indices[elem_idx]

    def get_data(self, size):
        '''Randomly sample a batch of size items.'''
        if size > self.buffer_size:
            size = self.buffer_size

        random_indices = np.random.choice(self.buffer_size, size=size, replace=False)
        return np.array(self.buffer_task_ids)[random_indices], np.array(self.buffer_data_ids)[random_indices]