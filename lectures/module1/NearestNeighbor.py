import numpy as np
import ipdb as db

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.Ytr = y

    def predict(self, X, k=1):
        num_test = X.shape[0]
        Y_pred = []

        for i in xrange(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis = 1)

            # Find the array indices of the 'k' lowest values in the "distances"
            # list.
            index = 0
            indexed_distances = []
            for value in distances:
                indexed_distances.append((index, value))
                index = index + 1

            kNearestNeighbors = np.sort(distances)[:k]
            indexList = [item for item, val in enumerate(indexed_distances) if val[1] in kNearestNeighbors]
            Y_pred.append([self.Ytr[indexList]])

        return Y_pred
