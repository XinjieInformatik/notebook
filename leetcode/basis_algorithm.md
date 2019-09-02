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
```

### list remove
#### python list remove(ele), pop(ele_idx), del array[idx]，只移除一个元素（第一个满足条件的元素

```PYTHON
a = ['a', 'a', 'b', 'c']
a.remove('a')
Out[15]: ['a', 'b', 'c']

a = ['a', 'a', 'b', 'c']
a.pop(1)
Out[17]: 'a'
Out[18]: ['a', 'b', 'c']

a = ['a', 'a', 'b', 'c']
del a[1]
Out[21]: ['a', 'b', 'c']
```

### str remove
#### strip 与 replace ele 均会移除str中所有该元素，可考虑通过索引切片移除
```PYTHON
# strip 只能移除首尾满足条件的元素
s = 'aabbcd'
s.strip('a')
Out[23]: 'bbcd'
s.strip('b')
Out[34]: 'aabbcd'

s.replace('a','')
Out[26]: 'bbcd'
s
Out[27]: 'aabbcd'

s[:4] + s[5:]
Out[30]: 'aabbd'
```
