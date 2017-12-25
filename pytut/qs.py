
# Yo why does not every single language have list comprehensions?
#
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + right

print(quicksort([4, 4, 2,  5, 1, 8, 9, 9, 0, -12]))
