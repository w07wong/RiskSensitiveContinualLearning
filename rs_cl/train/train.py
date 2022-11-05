import argparse
import jax
from jax import jit
from functools import partial
from flax.training import train_state
import numpy as np
import tqdm.notebook as tq
import pickle
import jax.numpy as jnp
import optax
from absl import logging
import multiprocessing as mp
import os
import platform
import math
from jax_resnet import ResNet18

from experiment_functionals import risk_functionals
from models.mlp import MLP
from models.mlp_simple import MLPSimple
from models.cnn import CNN
from dataset.mnist import MNIST
from dataset.cifar100 import CIFAR100
from buffer import Buffer


logging.set_verbosity(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2)
parser.add_argument('--momentum', type=float, default=0)
parser.add_argument('--batch_size', '-bsz', type=int, default=128)
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--training_seeds', type=int, default=5)
parser.add_argument('--task_permutations', type=int, default=10)
parser.add_argument('--dataset', type=str, default='MNIST', help='MNIST|CIFAR100')
parser.add_argument('--processes', type=int, default=1, help='Number of multiprocessing processes.')
parser.add_argument('--model', type=str, default='MLP', help='MLP|MLPSimple|CNN|ResNet18')
parser.add_argument('--replay', action='store_true')
parser.add_argument('--save_dir', type=str, default='runs', help='Location to save models and training stats.')


@partial(jit, static_argnums=(2, 6))
def apply_model(
    state, batch_stats, risk_functional, images, labels, task_labels, out_features, eval=False):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params, out_features):
        if batch_stats is not None:
            variables = {'params': params, 'batch_stats': batch_stats}
            mutable = [] if eval else ['batch_stats']
            logits, new_batch_stats = state.apply_fn(variables, images, mutable=mutable)
        else:
            logits = state.apply_fn({'params': params}, images)
            new_batch_stats = None
        one_hot = jax.nn.one_hot(labels, out_features)
        loss = risk_functionals[risk_functional](
            optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        '''
        if reduction is None, then apply aggregate losses per task
        and apply risk functional on each aggregation
        '''
        # if task_labels is None:
        #     loss = jnp.mean(loss)
        # else:
        #     task_losses = {}
        #     for i in range(len(loss)):
        #         if task_labels[i] not in task_losses:
        #             task_losses[task_labels[i]] = []
        #         task_losses[task_labels[i]].append(loss[i])
        #     if len(task_losses.keys()) > 1:
        #         losses = jnp.array([jnp.mean(jnp.array(task_losses[task])) for task in task_losses.keys()])
        #         '''Task loss is final loss'''
        #         # loss = jnp.mean(risk_functionals[risk_functional](losses))
        #         '''Or, task loss is auxiliary loss'''
        #         loss = 0.5 * (jnp.mean(loss) + jnp.mean(risk_functionals[risk_functional](losses)))
        #     else:
        #         loss = jnp.mean(loss)
        return loss, (logits, new_batch_stats)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, aux_data), grads = grad_fn(state.params, out_features)
    logits, new_batch_stats = aux_data
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy, new_batch_stats


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def create_train_state(model, params, config):
    """Creates initial `TrainState`."""
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)


@partial(jit, static_argnums=(2, 6))
def train_epoch(state, batch_stats, risk_functional, images, labels, task_labels, out_features):
    """Train for a single epoch."""
    grads, loss, accuracy, batch_stats = apply_model(
        state, batch_stats, risk_functional, images, labels, task_labels, out_features)
    state = update_model(state, grads)
    return state, grads, loss, accuracy, batch_stats


def train_and_evaluate(config: argparse.Namespace,
                       train_images,
                       train_labels,
                       val_images,
                       val_labels,
                       risk_functional,
                       state,
                       task_order,
                       out_features: int,
                       random_seed,
                       batch_stats=None) -> train_state.TrainState:
    """Execute model training and evaluation loop.
    Args:
        config: Hyperparameter configuration for training and evaluation.
        train_images: NumPy array of training images separated by task
        train_labels: NumPy array of training labels separated by task
        val_images: NumPy array of validation images separated by task
        val_labels: NumPy array of validation labels separated by task
        risk_functional: Risk functional to transform losses.
        state: Model state.
        task_order: ordering of tasks by id. e.g. [1, 2, 3, 4] or [3, 1, 4, 2]
        out_features: number of classes
        random_seed: number to seed pseudorandom number generator.
        batch_stats: Initial batch_stats collection to contain all running statistics for all the BatchNorm layers.

    Returns:
        The train state (which includes the `.params`).
    """

    all_train_accs = []
    all_val_accs = []
    first_task_losses = []
    first_task_accs = []
    last_layer_grads = dict()

    # Check to see if nan is encountered during training. If so, end training early.
    nan_encountered = False

    rng = jax.random.PRNGKey(random_seed)

    # Create replay buffer
    if config.replay:
        buffer = Buffer(5000) 

    for task_id in task_order:
        logging.info('==================Task {}==================='.format(task_id + 1))

        task_train_images, task_train_labels = train_images[task_id], train_labels[task_id]
        task_val_images, task_val_labels = val_images[task_id], val_labels[task_id]

        # Shuffle dataset
        train_ds_size = len(task_train_images)
        batches_per_epoch = int(math.ceil(train_ds_size / config.batch_size))
        permutations = jax.random.permutation(rng, train_ds_size)

        epoch = 0
        last_layer_grads[task_id] = []

        while epoch < config.epochs and not nan_encountered:
            batch_losses = []
            batch_accs = []
            for b in range(batches_per_epoch):
                # Get batch data
                start_idx = b * config.batch_size
                end_idx = min(start_idx + config.batch_size, len(permutations))
                indices = permutations[start_idx:end_idx]
                batch_images = task_train_images[indices, ...]
                batch_labels = task_train_labels[indices, ...]
                batch_task_labels = [task_id for _ in range(len(batch_images))]

                if config.replay and task_id != task_order[0]:
                    replay_task_ids, replay_data_ids = buffer.get_data(batch_images.shape[0])
                    replay_images = train_images[replay_task_ids, replay_data_ids]
                    replay_labels = train_labels[replay_task_ids, replay_data_ids]
                    state, train_grads, train_loss, train_accuracy, batch_stats = train_epoch(
                        state,
                        batch_stats,
                        risk_functional,
                        jnp.concatenate((batch_images, replay_images)),
                        jnp.concatenate((batch_labels, replay_labels)),
                        batch_task_labels + replay_task_ids,
                        out_features)
                else:
                    state, train_grads, train_loss, train_accuracy, batch_stats = train_epoch(
                        state,
                        batch_stats,
                        risk_functional,
                        batch_images,
                        batch_labels,
                        batch_task_labels,
                        out_features)

                if math.isnan(train_loss) or math.isnan(train_accuracy):
                    nan_encountered = True
                    break

                batch_losses.append(train_loss)
                batch_accs.append(train_accuracy)

                if config.replay:
                    buffer.add_data(indices, task_id)

                # Save last layer gradients w.r.t. loss of first task
                # first_task_val_images, first_task_val_labels = val_images[task_order[0]], val_labels[task_order[0]]
                # first_task_val_grads, _, _, _ = apply_model(
                #     state, batch_stats, 'Expected Value', first_task_val_images, first_task_val_labels, eval=True)
                # last_layer_grads[task_id].append(
                #     (train_grads['Dense_1'], first_task_val_grads['Dense_1']))

            _, val_loss, val_accuracy, _ = apply_model(
                state, batch_stats, 'Expected Value', task_val_images, task_val_labels, None, out_features, eval=True)
            all_val_accs.append(val_accuracy)

            logging.info(
                'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, val_loss: %.4f, val_accuracy: %.2f'
                % (epoch, np.mean(batch_losses), np.mean(batch_accs) * 100,
                    val_loss, val_accuracy * 100))
            all_train_accs.append(np.mean(batch_accs))

            if task_id == 0:
                first_task_losses.append(val_loss)
                first_task_accs.append(val_accuracy * 100)
            if task_id > 0:
                first_task_val_images, first_task_val_labels = val_images[task_order[0]], val_labels[task_order[0]]
                _, first_task_val_loss, first_task_val_accuracy, _ = apply_model(
                    state, batch_stats, 'Expected Value', first_task_val_images,
                    first_task_val_labels, None, out_features, eval=True)

                first_task_losses.append(first_task_val_loss)
                first_task_accs.append(first_task_val_accuracy * 100)

                logging.info(
                    'val_loss on first task: %.4f, val_accuracy on first task: %.2f'
                    % (first_task_val_loss, first_task_val_accuracy * 100))

            # Increment epoch counter
            epoch += 1

        if nan_encountered:
            break

    return state, first_task_losses, first_task_accs, last_layer_grads, all_train_accs, all_val_accs


def train_with_task_order(config, train_images, train_labels, val_images, val_labels, out_features, task_order):
    print('Task order: {}'.format(task_order))
    models = dict()
    all_last_node_grads = dict()
    all_first_task_losses = []
    all_first_task_accs = []
    for risk_functional in risk_functionals:
        models[risk_functional] = dict()
        print('Risk Functional: {}'.format(risk_functional))
        first_task_losses = []
        first_task_accs = []
        last_node_grads = []

        for i in tq.tqdm(range(config.training_seeds), desc='Random seed'):
            # Create model
            rng = jax.random.PRNGKey(i)
            rng, init_rng = jax.random.split(rng)
            batch_stats = None
            if config.model == 'MLP':
                model = MLP(out_features=out_features)
                first_image_shape = train_images[0][0].shape
                params = model.init(rng, jnp.ones([1, np.prod(first_image_shape)]))['params']
            elif config.model == 'MLPSimple':
                model = MLPSimple(out_features=out_features)
                first_image_shape = train_images[0][0].shape
                params = model.init(rng, jnp.ones([1, np.prod(first_image_shape)]))['params']
            elif config.model == 'CNN':
                model = CNN(train=True, out_features=out_features)
                first_image_shape = train_images[0][0].shape
                vars_initialized = model.init(rng, jnp.ones(
                    [1, first_image_shape[0], first_image_shape[1], first_image_shape[2]]))
                params = vars_initialized['params']
                # batch_stats = vars_initialized['batch_stats']
            elif config.model == 'ResNet18':
                model = ResNet18(n_classes=20)
                first_image_shape = train_images[0][0].shape
                vars_initialized = model.init(rng, jnp.ones(
                    [1, first_image_shape[0], first_image_shape[1], first_image_shape[2]]))
                params = vars_initialized['params']
                batch_stats = vars_initialized['batch_stats']
            else:
                raise Exception('Model type does not exist.')
            state = create_train_state(model, params, config)

            state, losses, accs, grads, train_accs, val_accs = train_and_evaluate(
                                                                config,
                                                                train_images,
                                                                train_labels,
                                                                val_images,
                                                                val_labels,
                                                                risk_functional,
                                                                state,
                                                                task_order,
                                                                out_features,
                                                                i,
                                                                batch_stats=batch_stats)
            first_task_losses.append(losses)
            first_task_accs.append(accs)
            last_node_grads.append(grads)
            models[risk_functional][i] = (state.params, losses, accs, grads, train_accs, val_accs)

        all_last_node_grads[risk_functional] = list(last_node_grads)
        all_first_task_losses.append([
            risk_functional,
            np.mean(first_task_losses, axis=0),
            np.max(first_task_losses, axis=0),
            np.min(first_task_losses, axis=0)])
        all_first_task_accs.append([
            risk_functional,
            np.mean(first_task_accs, axis=0),
            np.max(first_task_accs, axis=0),
            np.min(first_task_accs, axis=0)])
    
        all_results = dict()
        all_results['models'] = models
        all_results['all_last_node_grads'] = all_last_node_grads
        all_results['all_first_task_losses'] = all_first_task_losses
        all_results['all_first_task_accs'] = all_first_task_accs

        # Save data periodically
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        with open(config.save_dir + '/' + str(task_order) + '.pickle', 'wb') as handle:
            pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = parser.parse_args()

    # Create datasets
    if args.dataset == 'MNIST':
        train_images, train_labels, val_images, val_labels = MNIST()
        out_features = 2
    elif args.dataset == 'CIFAR100':
        train_images, train_labels, val_images, val_labels = CIFAR100(train_aug=True)
        out_features = 20
    else:
        raise Exception('Dataset does not exist.')

    # Create task orderings
    np.random.seed(0)
    task_orders = [[i for i in range(len(train_images))]]
    for _ in range(args.task_permutations - 1):
        new_order = [i for i in range(len(train_images))]
        np.random.shuffle(new_order)
        task_orders.append(new_order)
    
    for task_order in task_orders:
        train_with_task_order(
            args, train_images, train_labels, val_images, val_labels, out_features, task_order)

    # # Spawn training process
    # pool = mp.Pool(processes=args.processes)
    # func = partial(
    #         train_with_task_order, args, train_images, train_labels, val_images, val_labels, out_features)
    # pool.map(func, task_orders)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    # cpu_cores = [i for i in range(0, 1)] # Cores (numbered 0-11)
    # os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))

    main()