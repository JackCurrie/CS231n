# So this isn't actually being used/validated, Just following the
# implementations from https://cs231n.github.io/linear-classify/
# in order to make the readings a bit more clear
#

def L_i(x, y, W):
    """
    unvectorized version. Compute the multiclass svm loss for a single example (x,y)
    - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
    with an appended bias dimension in the 3073-rd position (i.e. bias trick)
    - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
    - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
    """
    delta = 1.0
    scores = W.dot(x) # becomes vector of classes, between 1 and 10 for CIFAR
    correct_class_scores = scores[y]
    D = W.shape[0]

    loss_i = 0.0

    # xrange is like the "range" function, but it returns the actual values
    # being iterated over
    for j in xrange(D):
        # Only add incorrect class predictions to the loss functions
        if j == y:
            continue
        loss_i += max(0, scores[j] - correct_class_scores + delta)

    return loss_i


def L_i_vectorized(x, y, W):
    """
    A faster half-vectorized implementation. half-vectorized
    refers to the fact that for a single example the implementation contains
    no for loops, but there is still one loop over the examples (outside this function)
    """
    delta = 1.0 # Delta is a hyperparameter which will be discussed later
    scores = W.dot(x)

    # compute the margins for all classes in one operation
    margins = np.maximum(0, scores - scores[y] + delta)

    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i


def L(X, y, W):
    """
    fully-vectorized implementation :
    - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
    - y is array of integers specifying correct class (e.g. 50,000-D array)
    - W are weights (e.g. 10 x 3073)
    """
    # Karpathy's notes:
    # evaluate loss over all examples in X without using any for loops
    # left as exercise to reader in the assignment

    # Set the hyperparameters
    #   - I think that we need both a delta and a lambda here
    #   - idk appropriate values for these, but I think that IRL there would
    #     be a known range that I would iterate over both of them with
    delta = 1.0
    lamba_val = 1.0

    # We are going to want to get a "loss vector", containing the loss for each of
    # the examples in "X" when compared to the output "y" with weights W
    #   - Get the data loss
    #   - And get the regularization loss
    N = X.shape[0]
    data_loss = np.max(0, W.dot(X) - y + delta) / N

    # The regularization loss is the lambda hyperparameter multiplied by each
    # value of the weight vector squared. This penalizes higher weight values.
    regularization_loss = lambda_val * (W ** 2)

    return data_loss + regularization_loss
