
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Here is just some practice pulling out all of the corner submatrices
# of the "a matrix"
topLeftSubMat = a[:2, :2]
bottomLeftSubMat = a[1:, :2]
topRightSubMat = a[:2, 2:]
bottomRightSubMat = a[1:, 2:]

# print(topLeftSubMat)
# print(bottomLeftSubMat)
# print(topRightSubMat)
# print(bottomRightSubMat)



# Here let's get some rows and coluns of the matrix
firstCol = a[:, 0]
secondCol = a[:, 1]
firstRow = a[0]
secondRow = a[1, :]

# print(firstCol)
# print(secondCol)
# print(firstRow)
# print(secondRow)



# Now let's talk boolean array indexing. This is a pretty remarkable
# feature where you can turn a matrix into a matrix of booleans based
# of certain conditions that each member may or may not meet.
a = np.array([[1,2], [3, 4], [5, 6]])

bool_indexed_a = (a > 2)

# Print the array and the array of booleans
# print(a)
# print(bool_indexed_a)

# Print the actual values in the array which meet the booleans
# indexing condition
# print(a[bool_indexed_a])
# print(a[a > 2])




# Types in numpy. A;l numpy arrays are of homogeneous datatypes
# Types in the array can be specified byt the second parameter of
# the array method
a = np.array([1, 2], np.float64) # Creates a numpy array of type float64
# print(a.dtype) # dtype of float64
a = np.array([1, 2])
# print(a.dtype) # int64 chosen by default for int
a = np.array([1.0, 2.0])
# print(a.dtype) # float64 chosen by default for decimals




# ELEMENT-WISE OPERATIONS
#   - These are available as operator overloads and as functions
#     in numpy. Hell yeah!
#
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# print(x - y)
# print(np.subtract(x, y))
# print(x / y)
# print(np.divide(x, y))
# print(x * y)
# print(np.multiply(x, y))










#
