# You may have to move / create a copy of this file out of the package because of ModuleNotFoundError.
# And maybe correct the imports too.

import numpy as np

from src.easyAI.network import Network
from src.easyAI.layers import FCLayer, ActivationLayer
from src.easyAI.activations import tanh, tanh_prime
from src.easyAI.lossfunction import mse, mse_prime

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 2, init_method="xavier"))
net.add(ActivationLayer(tanh, tanh_prime)) # choose here
net.add(FCLayer(2, 2, init_method="xavier"))
net.add(ActivationLayer(tanh, tanh_prime)) # choose here
net.add(FCLayer(2, 1))

# before training
out = net.predict(x_train)
print(out)

# train once
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)

# train 99 more times
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=99, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)
