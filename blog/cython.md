# Cython starter
Cython is Python with C data types. Almost any piece of Python code is also valid Cython code.

[basis official tutorial](https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html)
[Using C++ in Cython](https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html#wrapping-cplusplus)
[cython with numpy](https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html#memoryviews)


setup file, run with `python setup.py build_ext --inplace`

```python
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize('find_primes.pyx', annotate=True)
)
```
