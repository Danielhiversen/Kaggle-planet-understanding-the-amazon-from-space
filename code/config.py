import util
from keras.optimizers import Adam, RMSprop, SGD, Nadam

net_config = {}
net_config['batch_size'] = 150
net_config['nb_epoch'] = 1000000
net_config['kernel_initializer'] = 'glorot_uniform'
net_config['padding'] = 'same'
net_config['batchnormalization'] = True
net_config['batch_size'] = 120
net_config['dropout2'] = 0.5
net_config['dropout'] = 0.5
net_config['activation'] = 'relu'
net_config['use_conv_layer'] = False
net_config['pooling'] = 'avg'
net_config['use_stats'] = True

net_config['loss'] = 'binary_crossentropy' # util.binary_crossentropy_fbeta_loss # util.fbeta_loss
#optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
# optimizer = RMSprop(lr=0.045, rho=0.9, epsilon=1.0)
#optimizer = "adam" #util.noisyAdam(lr=0.1)
net_config['optimizer'] = Adam(lr=0.005) # SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True) # Adam(lr=0.5) #  util.Adam_accumulate(accum_iters=12) #
net_config['metrics'] = [util.fbeta, 'binary_crossentropy'] #['mean_squared_error', 'mean_absolute_error']