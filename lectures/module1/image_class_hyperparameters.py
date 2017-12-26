import numpy as np
from NearestNeighbor import NearestNeighbor
from load_full_CIFAR10 import load_CIFAR10


Xtr, Ytr, Xte, Yte = load_CIFAR10('data/cifar10/')
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)


# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]


# Iterate over hyperparameter selections for k-values
print "beginning hyperparameter search... \n"
validation_accuraccies = []
for k in [5, 10, 20, 50, 100]:

    nn = NearestNeighbor()
    nn.train(Xtr_rows, Ytr)
    Y_predictions = nn.predict(Xte_rows)

    acc = np.mean(Y_predictions == Yval)
    print 'accuracy: %f' % (acc, )

    validation_accuraccies.append((k, acc))
