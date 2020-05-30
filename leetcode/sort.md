# quick sort
```python
def qsort(array, l, r):
    def partition(array, l, r):
        pivot = array[r]
        pivot_index = l - 1
        for i in range(l,r):
            if array[i] < pivot:
                pivot_index += 1
                array[i], array[pivot_index] = array[pivot_index], array[i]
        pivot_index += 1
        array[pivot_index], array[r] = array[r], array[pivot_index]
        return pivot_index

    if l < r:
        pivot_index = partition(array, l, r)
        qsort(array, pivot_index+1, r)
        qsort(array, l, pivot_index-1)
```

# merge sort
```python
def mergesort(array):
    def merge(l, r):
        result = []
        l_i, r_i = 0, 0
        while (l_i < len(l) and r_i < len(r)):
            if l[l_i] < r[r_i]:
                result.append(l[l_i])
                l_i += 1
            else:
                result.append(r[r_i])
                r_i += 1
        if l_i < len(l):
            result.extend(l[l_i:])
        else:
            result.extend(r[r_i:])
        return result

    if len(array) <= 1:
        return array
    m = len(array) // 2
    l = mergesort(array[:m])
    r = mergesort(array[m:])
    return merge(l, r)
```

# bubble sort:
```python
def bubblesort(array):
    index = len(array) - 1
    while (index > 0):
        for i in range(index):
            if array[i] > array[i+1]:
                array[i], array[i+1] = array[i+1], array[i]
        index -= 1
```

# select sort
```python
def selectsort(array):
    result = []
    while (len(array) > 0):
        min_value = array[0]
        min_index = 0
        for i in range(len(array)):
            if array[i] < min_value:
                min_value = array[i]
                min_index = i
        result.append(min_value)
        # array.pop(min_index)
        array = np.delete(array, min_index)
    return result
```

# insert sort
```python
def insertsort(array):
    def low_bound(arr, l, r, target):
        while (l < r):
            m = l + (r - l) // 2
            if arr[m] < target:
                l = m + 1
            else:
                r = m
        return l

    result = []
    result.append(array[0])
    for i in range(1, len(array)):
        insert_index = low_bound(result, 0, len(result), array[i])
        result.insert(insert_index, array[i])
    return result
```
