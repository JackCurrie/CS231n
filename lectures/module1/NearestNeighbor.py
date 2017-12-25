import numpy as np

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

    def predict(self, X):
        num_test = X.shape[0]
        Y_pred = np.zeros(num_test, dtype=self.Ytr.dtype)

        print("number of tests: %i", num_test)
        for i in xrange(num_test):

            if 1 % 100 == 0:
                print("test %i" % i)

            # Find the nearest training image in the set to the current image
            # Use the L1 distance (sum of absolute value difference)
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis = 1)
            min_index = np.argmin(distances)
            Y_pred[i] = self.Ytr[min_index] # predict the label of the nearest example


        return Y_pred
