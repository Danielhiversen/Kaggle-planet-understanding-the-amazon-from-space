import util
from keras.optimizers import Adam, RMSprop, SGD


net_config = {}
net_config['batch_size'] = 100
net_config['nb_epoch'] = 100
net_config['kernel_initializer'] = 'glorot_uniform'
net_config['padding'] = 'same'
net_config['activation'] = 'relu'
net_config['dropout'] = 0
net_config['dropout2'] = 0

net_config['loss'] = 'binary_crossentropy'
#optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
# optimizer = RMSprop(lr=0.045, rho=0.9, epsilon=1.0)
#optimizer = "adam" #util.noisyAdam(lr=0.1)
net_config['optimizer'] = Adam(lr=.001)
net_config['metrics'] = ['accuracy'] #['mean_squared_error', 'mean_absolute_error']

net_config['augment'] = False