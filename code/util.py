import cv2
import numpy as np
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle

from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.layers import Layer, BatchNormalization, Activation, merge, Conv2D, Dropout, MaxPooling2D, AveragePooling2D, MaxPooling3D, AveragePooling3D, Conv3D
from keras.regularizers import l2
from tqdm import tqdm


def get_data():
    filename = '../input/jpg_data.pickle'
    if os.path.exists(filename):
        with open(filename, 'rb') as handle:
            x_train, x_valid, y_train, y_valid = pickle.load(handle)
    else:
        x_train = []
        x_test = []
        y_train = []

        df_train = pd.read_csv('../input/train.csv')
        flatten = lambda l: [item for sublist in l for item in sublist]
        labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

        label_map = {l: i for i, l in enumerate(labels)}

        for f, tags in tqdm(df_train.values[:20000], miniters=1000):
            img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))
            targets = np.zeros(17)
            for t in tags.split(' '):
                targets[label_map[t]] = 1
            x_train.append(cv2.resize(img, (32, 32)))
            y_train.append(targets)

        y_train = np.array(y_train, np.uint8)
        x_train = np.array(x_train, np.float16) / 255.

        print(x_train.shape)
        print(y_train.shape)

        split = 15000
        x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]
        with open(filename, 'wb') as f:
            pickle.dump((x_train, x_valid, y_train, y_valid), f)
    return x_train, x_valid, y_train, y_valid

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
    def get_output_shape_for(self, input_shape):
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


def conv_block(kernel_size, filters, stage, block, weight_decay=0., strides=(2, 2), batch_momentum=0.99, batchnormalization=False, activation=0):
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
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), subsample=strides,
                          name=conv_name_base + '2a', W_regularizer=l2(weight_decay))(input_tensor)
        if batchnormalization:
            x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', momentum=batch_momentum)(x)
        if activation == 1:
            x = CReLU()(x)
        else:
            x = Activation('relu')(x)

        x = Conv2D(nb_filter2, kernel_size, kernel_size, padding='same',
                          name=conv_name_base + '2b', W_regularizer=l2(weight_decay))(x)
        if batchnormalization:
            x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', momentum=batch_momentum)(x)
        if activation == 1:
            x = CReLU()(x)
        else:
            x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', W_regularizer=l2(weight_decay))(x)
        if batchnormalization:
            x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', momentum=batch_momentum)(x)

        shortcut = Conv2D(nb_filter3, (1, 1), subsample=strides,
                                 name=conv_name_base + '1', W_regularizer=l2(weight_decay))(input_tensor)
        if batchnormalization:
            shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', momentum=batch_momentum)(shortcut)

        x = merge([x, shortcut], mode='sum')
        if activation == 1:
            x = CReLU()(x)
        else:
            x = Activation('relu')(x)
        return x
    return f


def get_block(input, nc, batchnormalization, activation, nb_filter1, nb_filter2, dropout1, dropout2, pooling):
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
        block = Conv2D(nb_filter1, (3, 3), kernel_initializer=nc['kernel_initializer'], padding=nc['padding'])(block)
    elif num_dims == 5:
        block = Conv3D(nb_filter1, (3, 3, 3), kernel_initializer=nc['kernel_initializer'], padding=nc['padding'])(block)
    if batchnormalization > 0:
        block = BatchNormalization()(block)
    if activation == 1:
        block = CReLU()(block)
    else:
        block = Activation(nc['activation'])(block)
    if dropout1 > 0:
        block = Dropout(dropout1)(block)

    if num_dims == 4:
        block = Conv2D(nb_filter2, (3, 3), kernel_initializer=nc['kernel_initializer'], padding=nc['padding'])(block)
    elif num_dims == 5:
        block = Conv3D(nb_filter2, (3, 3, 3), kernel_initializer=nc['kernel_initializer'], padding=nc['padding'])(block)
    if batchnormalization > 0:
        block = BatchNormalization()(block)
    if activation == 1:
        block = CReLU()(block)
    else:
        block = Activation(nc['activation'])(block)
    if dropout2 > 0:
        block = Dropout(dropout2)(block)

    return block

