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
parser.add_argument('--model', type=str, default='MLP', help='MLP|MLPSimple|CNN')
parser.add_argument('--replay', action='store_true')


# @partial(jit, static_argnums=(1,5))
def apply_model(state, risk_functional, images, labels, task_labels, out_features):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params, out_features):
        logits = state.apply_fn({'params': params}, images)
        one_hot = jax.nn.one_hot(labels, out_features)
        loss = risk_functionals[risk_functional](
            optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        '''
        if reduction is None, then apply aggregate losses per task
        and apply risk functional on each aggregation
        '''
        if task_labels is None:
            loss = jnp.mean(loss)
        else:
            task_losses = {}
            for i in range(len(loss)):
                if task_labels[i] not in task_losses:
                    task_losses[task_labels[i]] = []
                task_losses[task_labels[i]].append(loss[i])
            if len(task_losses.keys()) > 1:
                losses = jnp.array([jnp.mean(jnp.array(task_losses[task])) for task in task_losses.keys()])
                '''Task loss is final loss'''
                # loss = jnp.mean(risk_functionals[risk_functional](losses))
                '''Or, task loss is auxiliary loss'''
                loss = 0.5 * (jnp.mean(loss) + jnp.mean(risk_functionals[risk_functional](losses)))
            else:
                loss = jnp.mean(loss)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params, out_features)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def create_train_state(model, params, rng, config):
    """Creates initial `TrainState`."""
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)


def train_epoch(state, risk_functional, images, labels, task_labels, out_features):
    """Train for a single epoch."""
    epoch_loss = []
    epoch_accuracy = []

    grads, loss, accuracy = apply_model(state, risk_functional, images, labels, task_labels, out_features)
    state = update_model(state, grads)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy, grads


def train_and_evaluate(config: argparse.Namespace,
                       train_dataset,
                       val_dataset,
                       risk_functional,
                       state,
                       task_order,
                       out_features: int,
                       random_seed) -> train_state.TrainState:
    """Execute model training and evaluation loop.
    Args:
    config: Hyperparameter configuration for training and evaluation.
    train_dataset: list of lists containing training images and training labels
    val_dataset: list of lists containing validation images and validation labels
    risk_functional: Risk functional to transform losses.
    state: Model state.
    task_order: ordering of tasks by id. e.g. [1, 2, 3, 4] or [3, 1, 4, 2]
    out_features: number of classes
    random_seed: number to seed pseudorandom number generator.
    Returns:
    The train state (which includes the `.params`).
    """

    all_train_accs = []
    all_val_accs = []
    first_task_losses = []
    first_task_accs = []
    last_layer_grads = dict()

    rng = jax.random.PRNGKey(random_seed)

    if config.replay:
        buffer = Buffer(5000) 

    for task_id in task_order:
        logging.info('==================Task {}==================='.format(task_id + 1))

        train_images, train_labels = train_dataset[task_id]
        val_images, val_labels = val_dataset[task_id]

        train_ds_size = len(train_images)
        steps_per_epoch = train_ds_size // config.batch_size
        perms = jax.random.permutation(rng, len(train_images))
        perms = perms[:steps_per_epoch * config.batch_size]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, config.batch_size))

        epoch = 0
        last_layer_grads[task_id] = []

        while epoch < config.epochs:
            batch_losses = []
            batch_accs = []
            for perm in perms:
                batch_images = train_images[perm, ...]
                batch_labels = train_labels[perm, ...]
                batch_task_labels = [task_id for _ in range(len(batch_images))]

                if config.replay and task_id > 0:
                    replay_images, replay_labels, replay_task_labels = buffer.get_data(config.batch_size)
                    replay_images = np.stack(replay_images, axis=0)
                    replay_labels = np.stack(replay_labels, axis=0)
                    replay_task_labels = replay_task_labels.tolist() + batch_task_labels
                    state, train_loss, train_accuracy, train_grads = train_epoch(state,
                                                                risk_functional,
                                                                jnp.concatenate((batch_images, replay_images)),
                                                                jnp.concatenate((batch_labels, replay_labels)),
                                                                replay_task_labels,
                                                                out_features)
                else:
                    state, train_loss, train_accuracy, train_grads = train_epoch(state,
                                                                risk_functional,
                                                                batch_images,
                                                                batch_labels,
                                                                batch_task_labels,
                                                                out_features)

                batch_losses.append(train_loss)
                batch_accs.append(train_accuracy)

                if config.replay:
                    buffer.add_data(batch_images, labels=batch_labels, task_labels=batch_task_labels)

                # Save last layer gradients w.r.t. loss of first task
                # first_task_val_images, first_task_val_labels = val_dataset[task_order[0]]
                # first_task_val_grads, _, _ = apply_model(
                #     state, 'Expected Value', first_task_val_images, first_task_val_labels)
                # last_layer_grads[task_id].append(
                #     (train_grads['Dense_1'], first_task_val_grads['Dense_1']))

            _, val_loss, val_accuracy = apply_model(
                state, 'Expected Value', val_images, val_labels, None, out_features)
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
                first_task_val_images, first_task_val_labels = val_dataset[task_order[0]]
                _, first_task_val_loss, first_task_val_accuracy = apply_model(
                    state, 'Expected Value', first_task_val_images, first_task_val_labels, None, out_features)

                first_task_losses.append(first_task_val_loss)
                first_task_accs.append(first_task_val_accuracy * 100)

                logging.info(
                    'val_loss on first task: %.4f, val_accuracy on first task: %.2f'
                    % (first_task_val_loss, first_task_val_accuracy * 100))

            # Increment epoch counter
            epoch += 1

    return state, first_task_losses, first_task_accs, last_layer_grads, all_train_accs, all_val_accs


def train_with_task_order(config, train_dataset, val_dataset, out_features, task_order):
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
            if config.model == 'MLP':
                model = MLP(out_features=out_features)
                first_image_shape = train_dataset[0][0][0].shape
                params = model.init(rng, jnp.ones([1, np.prod(first_image_shape)]))['params']
            elif config.model == 'MLPSimple':
                model = MLPSimple(out_features=out_features)
                first_image_shape = train_dataset[0][0][0].shape
                params = model.init(rng, jnp.ones([1, np.prod(first_image_shape)]))['params']
            elif config.model == 'CNN':
                model = CNN(out_features=out_features)
                first_image_shape = train_dataset[0][0][0].shape
                params = model.init(rng, jnp.ones(
                    [1, first_image_shape[0], first_image_shape[1], first_image_shape[2]]))['params']
            else:
                raise Exception('Model type does not exist.')
            state = create_train_state(model, params, init_rng, config)

            state, losses, accs, grads, train_accs, val_accs = train_and_evaluate(
                                                                config,
                                                                train_dataset,
                                                                val_dataset,
                                                                risk_functional,
                                                                state,
                                                                task_order,
                                                                out_features,
                                                                i)
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
    with open(str(task_order) + '.pickle', 'wb') as handle:
        pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = parser.parse_args()

    # Create datasets
    if args.dataset == 'MNIST':
        train_dataset, val_dataset = MNIST()
        out_features = 2
    elif args.dataset == 'CIFAR100':
        train_dataset, val_dataset = CIFAR100(train_aug=True)
        out_features = 20
    else:
        raise Exception('Dataset does not exist.')

    # Create task orderings
    np.random.seed(0)
    task_orders = [[i for i in range(len(train_dataset))]]
    for _ in range(args.task_permutations - 1):
        new_order = [i for i in range(len(train_dataset))]
        np.random.shuffle(new_order)
        task_orders.append(new_order)

    # Spawn training process
    pool = mp.Pool(processes=args.processes)
    func = partial(train_with_task_order, args, train_dataset, val_dataset, out_features)
    pool.map(func, task_orders)


if __name__ == '__main__':
    if platform.system() == "Darwin":
        mp.set_start_method('spawn')
    cpu_cores = [i for i in range(0, 4)] # Cores (numbered 0-11)
    os.system("taskset -pc {} {}".format(",".join(str(i) for i in cpu_cores), os.getpid()))

    main()