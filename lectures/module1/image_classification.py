import numpy as np
from NearestNeighbor import NearestNeighbor
from load_full_CIFAR10 import load_CIFAR10


Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/')

# Flatten the images in both the training and the
# test data to single dimensional arrays of pixel intensity
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)


nn = NearestNeighbor()
nn.train(Xtr_rows, Ytr)
Y_predictions = nn.predict(Xte_rows)

# Accuracy is the amount average rate that correct predictions
# were made on on the test data
print "accuracy: %f" % ( np.mean(Y_predictions == Yte))
