# https://www.kaggle.com/anokas/planet-understanding-the-amazon-from-space/simple-keras-starter
import numpy as np  # linear algebra

import keras.backend as K
from keras.models import Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

import util
import config


def get_model(nc, input_shape):
    activation = nc.get('activation', 0)
    batchnormalization = nc.get('batchnormalization', 0)

    inputs = Input(shape=input_shape, name='image_input')

    x = util.get_block(inputs, nc, batchnormalization, activation, nb_filter1=32, nb_filter2=32, dropout1=0, dropout2=0, pooling=0)
    x = util.get_block(x, nc, batchnormalization, activation, 64, 64, nc['dropout'], nc.get('dropout2', 0), nc.get('pooling', 'max'))

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(17, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss=nc.get('loss'),  # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                  optimizer=nc.get('optimizer'),
                  metrics=nc.get('metrics'))
    return model


def train(name, model, nc):
    x_train, x_valid, y_train, y_valid = util.get_data()

    model_checkpoint = ModelCheckpoint(name + '_best.h5', monitor='val_loss', save_best_only=True, mode='min')
    early_s = util.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=300, verbose=1, mode='min', patience2=25)
    try:
        history = model.fit(x_train, y_train, batch_size=nc['batch_size'], epochs=nc['nb_epoch'],
                            verbose=2, shuffle=True, validation_data=(x_valid, y_valid), callbacks=[model_checkpoint, early_s])
    except Exception as exp:
        print(exp)
        K.clear_session()
        return -1

    print(history.history)
    from sklearn.metrics import fbeta_score
    p_valid = model.predict(x_valid, batch_size=128)
    print(y_valid)
    print(p_valid)
    print(fbeta_score(y_valid, np.array(p_valid) > 0.5, beta=2, average='macro'))

def run():
    net_config = config.net_config
    model = get_model(net_config, (32, 32, 3))
    train("test", model, net_config)

if __name__ == "__main__":
    run()
