import numpy as np
import ipdb as db

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        # - y is simply a 1 dimensional array of values denoting the
        #   class that hte object is in
        # - The nearest neighbor classifier simply remembers all the training data
        #   so no real action needs to be taken here
        self.Xtr = X
        self.Ytr = y

    def predict(self, X, k=1):
        num_test = X.shape[0]
        Y_pred = []

        print "number of tests: %i", num_test

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

            print len(indexList)


        return Y_pred
