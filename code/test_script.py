# https://www.kaggle.com/anokas/planet-understanding-the-amazon-from-space/simple-keras-starter
import multiprocessing
import numpy as np
np.random.seed(123)  # for reproducibility
import tensorflow as tf
tf.set_random_seed(123)
import pickle
import time
from sklearn.metrics import fbeta_score

import keras.backend as K
from keras.applications import ResNet50, VGG19, VGG16
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, GlobalMaxPooling2D, concatenate, Activation, Concatenate, BatchNormalization
from keras.models import Input, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam, RMSprop, SGD, Nadam

import util
import config


def get_model(nc):
    batchnormalization = nc.get('batchnormalization', False)
    dropout1 = nc.get('dropout', 0)
    dropout2 = nc.get('dropout2', 0)
    pooling = nc.get('pooling', 'max')
    use_stats = nc.get('use_stats', False)

    inputs = Input(shape=util.INPUT_SHAPE, name='image_input')

    x0 = Conv2D(32, (7, 7), kernel_initializer='glorot_uniform', strides=(2, 2), padding='same', name='conv1')(inputs)
    x0 = util.CReLU()(x0)
    x0 = Dropout(dropout1)(x0)

    x = util.get_block(inputs, nc, batchnormalization, nb_filter1=32, nb_filter2=32, dropout1=dropout1, dropout2=dropout1, pooling=0, kernel_size1=7)
    if pooling == 'max':
        x = MaxPooling2D(pool_size=(2, 2))(x)
    elif pooling == 'avg':
        x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Concatenate(axis=-1)([x, x0])

    if nc.get('use_conv_layer'):
        x = util.conv_block(3, [8, 16, 8], weight_decay=0., stride=1, batch_momentum=0.99, batchnormalization=batchnormalization)(x)
        if pooling == 'max':
            x = MaxPooling2D(pool_size=(2, 2))(x)
        elif pooling == 'avg':
            x = AveragePooling2D(pool_size=(2, 2))(x)

    x = util.get_block(x, nc, batchnormalization, 32, 0, dropout1, dropout2, pooling)
    if nc.get('use_conv_layer'):
        x = util.conv_block(3, [8, 16, 8], weight_decay=0., stride=1, batch_momentum=0.99, batchnormalization=batchnormalization)(x)
        if pooling == 'max':
            x = MaxPooling2D(pool_size=(2, 2))(x)
        elif pooling == 'avg':
            x = AveragePooling2D(pool_size=(2, 2))(x)
    x = util.get_block(x, nc, batchnormalization, 32, 0, dropout1, dropout2, pooling)

    if pooling == 'max':
        x = MaxPooling2D(pool_size=(2, 2))(x)
    elif pooling == 'avg':
        x = AveragePooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dropout(dropout1)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout1)(x)
    x = Dense(128, activation='relu')(x)

    if use_stats:
        inputs2 = Input(shape=(278, ), name='image_input2')
        x2 = Dense(256, activation='relu')(inputs2)
        x2 = Dense(128, activation='sigmoid')(x2)
        x = concatenate([x, x2], axis=1)
        x = Dense(128, activation='relu')(x)
        x = Dropout(dropout1)(x)
        x = Dense(17, activation='sigmoid')(x)
        model = Model(inputs=[inputs, inputs2], outputs=x)
    else:
        x = Dropout(dropout1)(x)
        x = Dense(17, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=x)
    model.compile(loss=nc.get('loss'), optimizer=nc.get('optimizer'), metrics=nc.get('metrics'))
    print(model.summary())
    return model


def get_model_prenet(nc):
    inputs = Input(shape=util.INPUT_SHAPE, name='image_input')

    prenet = VGG16(include_top=False, input_tensor=inputs)
    print(prenet.summary())
    for layer in prenet.layers:
        if layer.name == 'block3_conv1':
            break
        layer.trainable = False
    x = prenet(inputs)

    x = Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(17, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    print(model.summary())
    model.compile(loss=nc.get('loss'), optimizer=nc.get('optimizer'), metrics=nc.get('metrics'))
    return model


def get_model_squeezenet(nc):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)
    @param nb_classes: total number of final categories
    Arguments:
    inputs -- shape of the input images (channel, cols, rows)
    https://www.kaggle.com/toshikuga/planet-understanding-the-amazon-from-space/deep-learning-on-planet-satellites-images-wip/notebook
    """
    dropout1 = nc.get('dropout', 0)

    inputs = Input(shape=util.INPUT_SHAPE, name='image_input')

    conv1 = Conv2D(
        96, (7, 7), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2, 2), padding='same', name='conv1')(inputs)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool1')(conv1)
    fire2_squeeze = Conv2D(
        16, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_squeeze')(maxpool1)
    fire2_expand1 = Conv2D(
        64, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand1')(fire2_squeeze)
    fire2_expand2 = Conv2D(
        64, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand2')(fire2_squeeze)
    merge2 = Concatenate(axis=-1)([fire2_expand1, fire2_expand2])
    merge2 = BatchNormalization()(merge2)
    merge2 = Dropout(dropout1)(merge2)

    fire4_squeeze = Conv2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_squeeze')(merge2)
    fire4_expand1 = Conv2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand1')(fire4_squeeze)
    fire4_expand2 = Conv2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand2')(fire4_squeeze)
    merge4 = Concatenate(axis=-1)([fire4_expand1, fire4_expand2])
    merge4 = BatchNormalization()(merge4)
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4')(merge4)
    maxpool4 = Dropout(dropout1)(maxpool4)

    fire5_squeeze = Conv2D(
        32, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_squeeze')(maxpool4)
    fire5_expand1 = Conv2D(
        128, (1, 1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand1')(fire5_squeeze)
    fire5_expand2 = Conv2D(
        128, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand2')(fire5_squeeze)
    merge5 = Concatenate(axis=-1)([fire5_expand1, fire5_expand2])
    merge5 = BatchNormalization()(merge5)

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge5)
    conv10 = Conv2D(
        17, (1, 1), kernel_initializer='glorot_uniform',
        padding='valid', name='conv10')(fire9_dropout)
    conv10 = BatchNormalization()(conv10)

    # The size should match the output of conv10
    avgpool10 = AveragePooling2D((7, 7), name='avgpool10')(conv10)
    flatten = Flatten(name='flatten')(avgpool10)

    if nc.get('use_stats', False):
        inputs2 = Input(shape=(278, ), name='image_input2')
        x2 = Dense(256, activation='relu')(inputs2)
        x2 = Dense(128, activation='sigmoid')(x2)
        x2 = Dense(17, activation='relu')(x2)
        x = concatenate([x2, flatten], axis=1)
        x = Dropout(0.3)(x)
        x = Dense(17, activation='sigmoid')(x)
        model = Model(inputs=[inputs, inputs2], outputs=x)
    else:
        softmax = Activation("sigmoid", name='sigmoid')(flatten)
        model = Model(inputs=inputs, outputs=softmax)

    model.compile(loss=nc.get('loss'), optimizer=nc.get('optimizer'), metrics=nc.get('metrics'))
    print(model.summary())
    return model


def train(name, model, nc):
    use_stats = nc.get('use_stats', False)

    x_train, x_valid, y_train, y_valid, stats_train, stats_valid = util.get_data_pickle()

    model_checkpoint = ModelCheckpoint(name + '_best.h5', monitor='val_fbeta', save_best_only=True, mode='max')
    early_s = util.EarlyStopping(monitor='val_fbeta', min_delta=0.01, patience=90, verbose=1, mode='max', patience2=5)
    learning_rate_up = util.LearningRateChanger(monitor='val_fbeta', min_delta=0.01, patience=70, verbose=1, mode='max')

    try:
        if use_stats:
            train_data = [x_train, stats_train]
            valid_data = [x_valid, stats_valid]
        else:
            train_data = x_train
            valid_data = x_valid

        if nc.get('class_weight', False):
            def create_class_weight(labels_dict, mu=0.15):
                import math
                total = np.sum(labels_dict)
                class_weight = dict()

                for k in range(len(labels_dict)):
                    score = math.log(mu * total / float(labels_dict[k]))
                    class_weight[k] = score if score > 1.0 else 1.0

                return class_weight

            # random labels_dict
            labels = [2330,  8076,   2695,    100,     98,   4477,  28203,    340,    339,  37840,    859,    332,  12338,   7262,    209,   7251,   3662]
            class_weight = create_class_weight(labels)
            print(class_weight)

        else:
            class_weight = None

        history = model.fit(train_data, y_train, batch_size=nc['batch_size'], epochs=nc['nb_epoch'],
                            verbose=2, shuffle=True , validation_data=(valid_data, y_valid), callbacks=[model_checkpoint, early_s, learning_rate_up], class_weight=class_weight)

        model.load_weights(name + '_best.h5')

        # batch_size = 128
        # gen = ImageDataGenerator(
        #     featurewise_center=True,
        #     featurewise_std_normalization=False,
        #     zca_whitening=False,
        #     zoom_range=[1.0, 1.2],
        #     fill_mode='nearest',
        #     horizontal_flip=True,
        #     vertical_flip=True,
        # ).flow(x_train, y_train, batch_size=batch_size)
        # history2 = model.fit_generator(
        #     gen,
        #     steps_per_epoch=x_train.shape[0] // batch_size,
        #     epochs=5,
        #     verbose=1,
        #     validation_data=(x_valid, y_valid)
        # )
    except Exception as exp:
        print(exp)
        K.clear_session()
        return [-1, -1]

    p_valid = model.predict(valid_data, batch_size=128)
    p_train = model.predict(train_data, batch_size=128)

    best_val = 0
    best_t = 0
    for t in np.arange(0, 1, 0.01):
        val = fbeta_score(y_train, np.array(p_train) > t, beta=2, average='samples')
        if val > best_val:
            best_val = val
            best_t = t

    res = fbeta_score(y_valid, np.array(p_valid) > best_t, beta=2, average='samples')

    return [res, best_t]

def run():
    net_config = config.net_config
    net_config['batchnormalization'] = True
    net_config['kernel_initializer'] = 'glorot_uniform'
    net_config['batch_size'] = 100
    net_config['dropout2'] = 0.3
    net_config['dropout'] = 0.5
    net_config['activation'] = 'relu'
    net_config['use_conv_layer'] = False
    net_config['pooling'] = 'max'
    net_config['use_stats'] = False
    net_config['class_weight'] = False

    net_config['loss'] = 'binary_crossentropy'

    model = get_model(net_config)
    res = train("test", model, net_config)
    print(res)

if __name__ == "__main__":
    run()
    
