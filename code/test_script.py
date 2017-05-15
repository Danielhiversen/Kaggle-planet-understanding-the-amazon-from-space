# https://www.kaggle.com/anokas/planet-understanding-the-amazon-from-space/simple-keras-starter
import numpy as np  # linear algebra

import keras.backend as K
from keras.models import Input, Model
from keras.layers import Dense, Dropout, Flatten, regularizers
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

import util
import config


def get_model(nc, input_shape):
    inputs = Input(shape=input_shape, name='image_input')

    x = inputs
    x = Conv2D(64, (3, 3), kernel_initializer=nc['kernel_initializer'], activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3), kernel_initializer=nc['kernel_initializer'], activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(16, (3, 3), kernel_initializer=nc['kernel_initializer'], activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.01)(x)
    x = Flatten()(x)
    x = Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)

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

    print "accuracy dist: ", sum((p_valid > 0.5) == y_valid)/float(len(y_valid)), sum((p_valid > 0.6) == y_valid)/float(len(y_valid)), sum((p_valid > 0.7) == y_valid)/float(len(y_valid)), sum((p_valid > 0.8) == y_valid)/float(len(y_valid))

    p_valid = p_valid > 0.5
    print(fbeta_score(y_valid, p_valid, beta=2, average='macro'))
    print "precision_recall_fscore: ",precision_recall_fscore_support(y_valid, p_valid, beta=2, average='macro', warn_for=[0,0])

    zero_accuracy = np.array(sum(y_valid == 0), dtype=float) / len(y_valid)
    print "constant func accuracy", np.array2string(np.max([1 - zero_accuracy, zero_accuracy], axis=0), max_line_width=200)
    print "class accuracy: ", np.array2string(np.array(sum(y_valid == p_valid), dtype=float)/len(y_valid), max_line_width=200)
    print "avg constant func accuracy", np.array2string(np.average(np.max([1 - zero_accuracy, zero_accuracy], axis=0)), max_line_width=200)
    print "avg class accuracy: ", np.array2string(np.average(np.array(sum(y_valid == p_valid), dtype=float)/len(y_valid)), max_line_width=200)

    print "nr correct: ", np.array2string(sum(y_valid == p_valid), max_line_width=200)
    print "nr wrong: ", np.array2string(sum(y_valid != p_valid), max_line_width=200)
    print
    print "true possitive:", np.array2string(sum(np.logical_and(y_valid == p_valid, y_valid == 1)), max_line_width=200)
    print "true negative: ", np.array2string(sum(np.logical_and(y_valid == p_valid, y_valid == 0)), max_line_width=200)
    print "false positie: ", np.array2string(sum(np.logical_and(y_valid != p_valid, y_valid == 0)), max_line_width=200)
    print "false negative:", np.array2string(sum(np.logical_and(y_valid != p_valid, y_valid == 1)), max_line_width=200)

def run():
    net_config = config.net_config
    model = get_model(net_config, (64, 64, 3))
    train("test", model, net_config)

if __name__ == "__main__":
    run()
