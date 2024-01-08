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