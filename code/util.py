import cv2
import numpy as np
import h5py
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from PIL import Image, ImageStat
import cv2

from keras import backend as K
from keras.callbacks import Callback
from keras.engine import InputSpec
from keras.optimizers import Adam, Optimizer
from keras.layers import Layer, BatchNormalization, Activation, add, Conv2D, Dropout, MaxPooling2D, AveragePooling2D, MaxPooling3D, AveragePooling3D, Conv3D
from keras.regularizers import l2
from tqdm import tqdm
from keras.utils.io_utils import HDF5Matrix

INPUT_SHAPE = (64, 64, 3)

def get_data(split=35000):
    filename = '../input/jpg_data.h5'
    if not os.path.exists(filename):
        num_images = 40479
        dataset = h5py.File(filename, 'w')
        dataset.create_dataset('images_jpg', (num_images, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]), dtype='f')
        dataset.create_dataset('classes', (num_images, 17), dtype='i')
        dataset.create_dataset('stats', (num_images, 278), dtype='f')

        df_train = pd.read_csv('../input/train.csv')
        flatten = lambda l: [item for sublist in l for item in sublist]
        labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

        label_map = {l: i for i, l in enumerate(labels)}
        i = 0
        stats = []
        for f, tags in tqdm(df_train.values, miniters=1000):
            img_path = '../input/train-jpg/{}.jpg'.format(f)
            img = cv2.imread(img_path)
            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[label_map[t]] = 1
            dataset['images_jpg'][i, :, :, :] = np.array(cv2.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1])), np.float16)/255.0
            dataset['classes'][i, :] = targets
            if i % 20 == 0:
                dataset.flush()
            i += 1
            stats.append(get_features(img_path))
        stats = np.array(stats, np.float16)
        stats[stats == np.Inf] = 0
        stats /= np.max(np.max(stats))
        dataset['stats'][:] = stats

        dataset.close()

    x_train = HDF5Matrix(filename, 'images_jpg', end=split)
    y_train = HDF5Matrix(filename, 'classes', end=split)
    stats_train = HDF5Matrix(filename, 'stats', end=split)

    x_valid = HDF5Matrix(filename, 'images_jpg', start=split)
    y_valid = HDF5Matrix(filename, 'classes', start=split)
    stats_valid = HDF5Matrix(filename, 'stats', start=split)

    return x_train, x_valid, y_train, y_valid, stats_train, stats_valid


def get_data_pickle():
    filename = '../input/jpg_data.pickle'
    if os.path.exists(filename):
        with open(filename, 'rb') as handle:
            x_train, x_valid, y_train, y_valid, stats_train, stats_valid = pickle.load(handle)
    else:
        x = []
        y = []
        stats = []

        df_train = pd.read_csv('../input/train.csv')
        flatten = lambda l: [item for sublist in l for item in sublist]
        labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))
        label_map = {l: i for i, l in enumerate(labels)}

        for f, tags in tqdm(df_train.values, miniters=1000):
            img_path = '../input/train-jpg/{}.jpg'.format(f)
            img = cv2.imread(img_path)
            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[label_map[t]] = 1
            x.append(cv2.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1])))
            y.append(targets)
            stats.append(get_features(img_path))

        y = np.array(y, np.uint8)
        x = np.array(x, np.float16) / 255.
        stats = np.array(stats, np.float16)
        stats[stats == np.Inf] = 0
        stats /= np.max(np.max(stats))

        print(x.shape)
        print(y.shape)
        print(stats.shape)

        split = 30000
        x_train, x_valid, y_train, y_valid = x[:split], x[split:], y[:split], y[split:]
        stats_train, stats_valid = stats[:split], stats[split:]
        with open(filename, 'wb') as f:
            pickle.dump((x_train, x_valid, y_train, y_valid, stats_train, stats_valid), f)
    return x_train, x_valid, y_train, y_valid, stats_train, stats_valid


def get_equal_data_pickle():
    total_labels = 200
    filename = '../input/jpg_data_equal .pickle'
    if os.path.exists(filename):
        with open(filename, 'rb') as handle:
            x_train, x_valid, y_train, y_valid, stats_train, stats_valid = pickle.load(handle)
    else:
        x_train = []
        x_valid = []
        y_train = []
        y_valid = []
        stats_train = []
        stats_valid = []

        df_train = pd.read_csv('../input/train.csv')
        flatten = lambda l: [item for sublist in l for item in sublist]
        labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))
        label_map = {l: i for i, l in enumerate(labels)}

        num_labels = np.zeros(17)

        for f, tags in tqdm(df_train.values, miniters=1000):
            img_path = '../input/train-jpg/{}.jpg'.format(f)
            img = cv2.imread(img_path)
            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[label_map[t]] = 1
            if np.any((num_labels + targets) < total_labels) or True:
                x_train.append(cv2.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1])))
                y_train.append(targets)
                stats_train.append(get_features(img_path))
                num_labels += targets
            else:
                x_valid.append(cv2.resize(img, (INPUT_SHAPE[0], INPUT_SHAPE[1])))
                y_valid.append(targets)
                stats_valid.append(get_features(img_path))

        print(num_labels)
        print(len(x_train))
        print(len(x_valid))

        y_train = np.array(y_train, np.uint8)
        x_train = np.array(x_train, np.float16) / 255.
        stats_train = np.array(stats_train, np.float16)
        stats_train[stats_train == np.Inf] = 0
        max_stats = np.max(np.max(stats_train))
        stats_train /= max_stats

        y_valid = np.array(y_valid, np.uint8)
        x_valid = np.array(x_valid, np.float16) / 255.
        stats_valid = np.array(stats_valid, np.float16)
        stats_valid[stats_valid == np.Inf_valid] = 0
        stats_valid /= max_stats

        with open(filename, 'wb') as f:
            pickle.dump((x_train, x_valid, y_train, y_valid, stats_train, stats_valid), f)
    return x_train, x_valid, y_train, y_valid, stats_train, stats_valid


# https://github.com/ypeleg/keras/blob/67eb00d42cd2f0a5ec0ac7aadc6ba6a276b714e6/keras/noisy_optimizers.py
def noisy_gradient(gradient, noise):
    return [g * K.random_normal(shape=K.shape(g), mean=1.0, std=noise) for g in gradient]


class NoisyAdam(Adam):
    def __init__(self, noise=0.05, **kwargs):
        self.noise = noise
        super(NoisyAdam, self).__init__(**kwargs)

    def get_gradients(self, loss, params):
        grads = super(NoisyAdam, self).get_gradients(loss, params)
        if self.noise == 0:
            return grads
        return noisy_gradient(grads, self.noise)


# https://github.com/ypeleg/keras/blob/67eb00d42cd2f0a5ec0ac7aadc6ba6a276b714e6/keras/layers/CReLu.py
class CReLU(Layer):
    '''    Based on: https://arxiv.org/pdf/1603.05201v2.pdf    '''
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] *= 2
        return tuple(shape)

    def call(self, x, mask=None):
        pos = K.relu(x)
        neg = K.relu(-x)
        return K.concatenate([pos, neg], axis=-1)


class EarlyStopping(Callback):
    """Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
    """

    def __init__(self, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, mode='auto', patience2=0):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.patience2 = patience2
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.wait2 = 0
        self.stopped_epoch = 0
        self.prev = None

        if mode not in ['auto', 'min', 'max']:
            print('EarlyStopping mode %s is unknown, '
                  'fallback to auto mode.' % (self.mode),
                   RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.prev = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            print('Early stopping requires %s available!' %
                  (self.monitor), RuntimeWarning)

        print(self.patience - self.wait, self.patience2 - self.wait2)
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1

        if current == self.prev or np.isnan(current):
            if self.wait2 >= self.patience2:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait2 += 1
        else:
            self.wait2 = 0
            self.prev = current

    def on_train_end(self, logs=None):
        print(logs.get(self.monitor), self.min_delta, self.best)
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))


class LearningRateChanger(Callback):
    def __init__(self, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, mode='auto'):
        super(LearningRateChanger, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            print('EarlyStopping mode %s is unknown, '
                  'fallback to auto mode.' % (self.mode),
                   RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            print('Early stopping requires %s available!' %
                  (self.monitor), RuntimeWarning)

        print(self.patience - self.wait, self.best, K.get_value(self.model.optimizer.lr))
        if current > 0.9 and epoch % 10 == 0:
            current_lr = K.get_value(self.model.optimizer.lr)
            new_lr = current_lr * 0.9
            print('new lr: ',  new_lr)
            K.set_value(self.model.optimizer.lr, new_lr)

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.wait = 0
                current_lr = K.get_value(self.model.optimizer.lr)
                new_lr = current_lr * 1.5
                K.set_value(self.model.optimizer.lr, new_lr)
            self.wait += 1


def conv_block(kernel_size, filters, weight_decay=0., stride=1, batch_momentum=0.99, batchnormalization=False, activation='relu', dim3=False):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        if K.image_dim_ordering() == 'tf':
            bn_axis = 3
            if dim3:
                bn_axis += 1
        else:
            bn_axis = 1

        if dim3:
            kernel_size0 = (1, 1, 1)
            kernel_sizes = (kernel_size, kernel_size, kernel_size)
            strides = (stride, stride, stride)
            conv = Conv3D
        else:
            kernel_size0 = (1, 1)
            kernel_sizes = (kernel_size, kernel_size)
            strides = (stride, stride)
            conv = Conv2D

        x = conv(nb_filter1, kernel_size0, strides=strides, kernel_regularizer=l2(weight_decay))(input_tensor)
        if batchnormalization:
            x = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(x)
        if activation == 1:
            x = CReLU()(x)
        else:
            x = Activation('relu')(x)

        x = conv(nb_filter2, kernel_sizes, padding='same', kernel_regularizer=l2(weight_decay))(x)
        if batchnormalization:
            x = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(x)
        if activation == 'CReLU':
            x = CReLU()(x)
        else:
            x = Activation('relu')(x)

        x = conv(nb_filter3, kernel_size0, kernel_regularizer=l2(weight_decay))(x)
        if batchnormalization:
            x = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(x)

        shortcut = conv(nb_filter3, kernel_size0, strides=strides, kernel_regularizer=l2(weight_decay))(input_tensor)
        if batchnormalization:
            shortcut = BatchNormalization(axis=bn_axis, momentum=batch_momentum)(shortcut)

        x = add([x, shortcut])
        if activation == 'CReLU':
            x = CReLU()(x)
        else:
            x = Activation(activation)(x)
        return x
    return f


def get_block(input, nc, batchnormalization, nb_filter1, nb_filter2, dropout1, dropout2, pooling, kernel_size1=3, kernel_size2=3):
    block = input
    num_dims = len(K.int_shape(input))

    if pooling == 'max' and num_dims == 4:
        block = MaxPooling2D(pool_size=(2, 2))(block)
    elif pooling == 'max' and num_dims == 5:
        block = MaxPooling3D(pool_size=(2, 2, 2))(block)
    elif pooling == 'avg' and num_dims == 4:
        block = AveragePooling2D(pool_size=(2, 2))(block)
    elif pooling == 'avg' and num_dims == 5:
        block = AveragePooling3D(pool_size=(2, 2, 2))(block)

    if num_dims == 4:
        block = Conv2D(nb_filter1, (kernel_size1, kernel_size1), kernel_initializer=nc['kernel_initializer'], padding=nc['padding'])(block)
    elif num_dims == 5:
        block = Conv3D(nb_filter1, (kernel_size1, kernel_size1, kernel_size1), kernel_initializer=nc['kernel_initializer'], padding=nc['padding'])(block)
    if batchnormalization:
        block = BatchNormalization()(block)
    if nc['activation'] == 'CReLU':
        block = CReLU()(block)
    else:
        block = Activation(nc['activation'])(block)
    if dropout1 > 0:
        block = Dropout(dropout1)(block)

    if nb_filter2 < 1:
        return block

    if num_dims == 4:
        block = Conv2D(nb_filter2, (kernel_size2, kernel_size2), kernel_initializer=nc['kernel_initializer'], padding=nc['padding'])(block)
    elif num_dims == 5:
        block = Conv3D(nb_filter2, (kernel_size2, kernel_size2, kernel_size2), kernel_initializer=nc['kernel_initializer'], padding=nc['padding'])(block)
    if batchnormalization:
        block = BatchNormalization()(block)
    if nc['activation'] == 'CReLU':
        block = CReLU()(block)
    else:
        block = Activation(nc['activation'])(block)
    if dropout2 > 0:
        block = Dropout(dropout2)(block)

    return block


def fbeta(y_true, y_pred, threshold_shift=0.):
    beta = 2
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())


def fbeta_loss(y_true, y_pred, threshold_shift=0.):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = (y_pred + threshold_shift)

    tp = K.sum((y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum((K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum((K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    res = (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
    return -1 * res

def binary_crossentropy_fbeta_loss(y_true, y_pred, threshold_shift=0.):
    return 1 + fbeta_loss(y_true, y_pred) + K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)


# https://www.kaggle.com/the1owl/planet-understanding-the-amazon-from-space/natural-growth-patterns-fractals-of-nature/run/1111331/notebook
def get_features(path):
    try:
        st = []
        #pillow
        im_stats_ = ImageStat.Stat(Image.open(path))
        #print(im_stats_.sum)
        #st += im_stats_.sum
        st += im_stats_.mean
        st += im_stats_.rms
        st += im_stats_.var
        st += im_stats_.stddev
        #cv2
        img = cv2.imread(path)
        bw = cv2.imread(path,0)
        st += list(cv2.calcHist([bw],[0],None,[256],[0,256]).flatten()) #histogram
        m, s = cv2.meanStdDev(img) #mean and standard deviation
        st += list(m)
        st += list(s)
        st += cv2.Laplacian(bw, cv2.CV_64F).var() #blurr
        st += (bw<10).sum()
        st += (bw>245).sum()
        #img = cv2.resize(img, (20, 20), cv2.INTER_LINEAR)
        return st
    except:
        print(path)


class Adagrad(Optimizer):
    """Adagrad optimizer.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    """

    def __init__(self, lr=0.01, epsilon=1e-8, decay=0., **kwargs):
        super(Adagrad, self).__init__(**kwargs)
        self.lr = K.variable(lr, name='lr')
        self.epsilon = epsilon
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.iterations = K.variable(0., name='iterations')

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        shapes = [K.get_variable_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))
            self.updates.append(K.update_add(self.iterations, 1))

        for p, g, a in zip(params, grads, accumulators):
            new_a = a + K.square(g)  # update accumulator
            self.updates.append(K.update(a, new_a))
            new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(Adagrad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    get_equal_data_pickle()