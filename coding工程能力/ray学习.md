# RAY优秀实践学习

用@ray.remote装饰单进程要处理的内容

```python
import ray
import time
from typing import List, Any
from utils import time_cost

ray.init()


@ray.remote
def process_unit(*args, **kwargs) -> None:
    time.sleep(1)

    return


@time_cost
def process_list(items: List[Any]) -> None:
    futures = [process_unit.remote(item) for item in items]
    result = ray.get(futures)

    return result


@ray.remote
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n 
    
    def process(self, n: int) -> None:
        for _ in range(n):
            time.sleep(1)


if __name__ == "__main__":
    # use @ray.remote in process unit
    # ray parallelizing: func level
    fake_input = [i for i in range(1000)]
    process_list(fake_input)

    # ray parallelizing: instace level
    counters = [Counter.remote() for i in range(4)]
    [c.increment.remote() for c in counters]
    futures = [c.read.remote() for c in counters]
    print(ray.get(futures))
```


```python
import ray
import time
from typing import List, Any
from utils import time_cost


@ray.remote
def process_unit(*args, **kwargs) -> None:
    time.sleep(1)

    return


@time_cost
def process_list(items: List[Any]) -> None:
    futures = [process_unit.remote(item) for item in items]
    result = ray.get(futures)

    return result


@ray.remote
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n

    def process(self) -> None:
        time.sleep(1)
        return 1


@time_cost
def process(num: int) -> None:
    counter = Counter.remote()
    futures = []
    for idx in range(num):
        futures.append(counter.process.remote())

    result = ray.get(futures)

    # ray parallelizing: instace level
    counters = [Counter.remote() for i in range(4)]
    [c.increment.remote() for c in counters]
    futures = [c.read.remote() for c in counters]
    print(ray.get(futures))

    return result

class CounterNorm(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def read(self):
        return self.n

    def __call__(self, idx: int) -> None:
        self.increment()
        print(self.n, idx, id(self))
        time.sleep(1)

        return idx


@time_cost
def process_norm(num: int) -> None:
    @ray.remote
    def wrapper(func, *args, **kwargs):
        return func(*args, **kwargs)

    counter = CounterNorm()
    futures = []
    for idx in range(num):
        event_task = wrapper.remote(counter, idx)
        futures.append(event_task)
    result = ray.get(futures)

    return result


if __name__ == "__main__":
    ray.init()

    # use @ray.remote in process unit
    # ray parallelizing: func level. 
    # NOTE: future=func.remote(*args, **kws); result=ray.get(future)
    fake_input = [i for i in range(1000)]
    process_list(fake_input)

    # ray parallelizing: class level. class生成多个对象并行，类/对象属性各自独立
    # NOTE: obj=class.remote(), obj.func.remote(*args, **kws), ray.get()
    process(20)

    # ray parallelizing: class level. 用@ray.remote修饰class的某个函数
    # 这时候类的多个对象之间的属性是独立的
    result = process_norm(20)
    print(result)

```