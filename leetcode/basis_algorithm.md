# Basis algorithm PYTHON

### sorted dictionary
- sort by key (construct new dict)
reverse=True, sort from big to small
```PYTHON
for key in sorted(mydict.keys(), reverse=True):
    new_dict[key] = mydict[key]
```
- sort by value (construct new dict)
```PYTHON
for key, value in sorted(mydict.items(), key=lambda ele: ele[1]):
    new_dict[key] = value
```
- return dict first element
```PYTHON
list(my_dict.keys())[0]
list(my_dict.value())[0]
```

### sorted 多重排序
```PYTHON
sorted(mylist, key=lambda ele:(ele[1],ele[0]))
