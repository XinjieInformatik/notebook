# python 多线程/多进程实践

## 原理

### 全局解释器锁（GIL）
为了有效地管理内存、进行垃圾回收以及在库中调用机器码，Python 拥有一个名为全
局解释器锁（GIL）的工具。它是无法被关闭的，这意味着线程在其他语言中所擅长的并
行处理在Python 中是无用的。GIL 的主要作用是阻止任何两个线程在同一时间运行（即便
它们有任务需要完成）。在这里，“有任务”意味着使用CPU，因此不同的线程访问磁盘或
网络是完全可以的。

### 多进程
多进程模块通过调动新的操作系统进程来实现。在Windows 机器上，这一操作的代价
相对来说比较昂贵；在Linux 上，进程在内核中的实现方式和线程一样，因此其开支受限
于每个进程中运行的Python解释器。
频繁IO任务, 避免多进程来回切换, 可考虑固定几个进程.

### 线程池(Pool)
Pool(进程池)可以提供指定数量的进程供用户调用，当有新的请求提交到pool中时，如果池还没有满，那么就会创建一个新的进程用来执行该请求；但如果池中的进程数已经达到规定最大值，那么该请求就会等待，直到池中有进程结束，才会创建新的进程来它。
https://docs.python.org/zh-cn/3/library/multiprocessing.html

多进程常见方法的异同.

| method             | multi-args | blocking | ordered-results |
| ------------------ | ---------- | -------- | --------------- |
| Pool.map           | no         | yes      | yes             |
| Pool.map_async     | no         | no       | yes             |
| Pool.apply         | yes        | yes      | no              |
| Pool.apply_async   | yes        | no       | no              |
| Pool.starmap       | yes        | yes      | yes             |
| Pool.starmap_async | yes        | no       | no              |

- Pool.imap and Pool.imap_async: lazier version of map and map_async.

多进程backend用`spawn`比`fork`更安全

- `fork`: 除了必要的启动资源外，其他变量, 包, 数据等都继承自父进程，并且是copy-on-write的，也就是共享了父进程的一些内存页，因此启动较快，但是由于大部分都用的父进程数据，所以是不安全的进程.
- `spawn`: 从头构建一个子进程，父进程的数据等拷贝到子进程空间内，拥有自己的Python解释器，所以需要重新加载一遍父进程的包，因此启动较慢，由于数据都是自己的，安全性较高.

## Examples
这里提供3中(`ProcessPoolIter`, `PoolIter`, `Pool`)快速构建多进程程序的方式, 其中`Pool`是原生的, 基于迭代器使用比较方便, 但是注意处理的还是整个list. 另外两种基于hatbc, 迭代器处理, 稳定性应该也比较高了.

```python
import time
import multiprocessing
from hatbc.distributed.multi_worker_iter import MultiWorkerIter
from typing import List, Dict
from concurrent.futures import (
    ProcessPoolExecutor as _ProcessPoolExecutor,
    ThreadPoolExecutor as _ThreadPoolExecutor,
)
from hatbc.distributed.client._process_pool_executor import ProcessPoolExecutor
from tqdm import tqdm


class _WorkerInitParams(object):
    def __init__(self, misc: dict, scale: int, thresh: int):
        self.misc = misc
        self.scale = scale
        self.thresh = thresh


def _worker_initializer(misc: dict, scale: int, thresh: int):
    global _worker_params
    _worker_params = _WorkerInitParams(
        misc, scale, thresh
    )


def count(frame: int, thresh: int):
    for i in range(0, 10000000):
        i = i + 1
    return i * frame * thresh


def _mp_worker_fn(frame: int):
    frame *= _worker_params.scale
    res = count(frame, _worker_params.thresh)

    return res


def _mp_worker_fn2(frame: int, scale:int, thresh:int):
    frame *= scale
    res = count(frame, thresh)

    return res


def _iter_fn(frames: List[int], scale:int, thresh:int, misc: dict):
    for frame in frames:
        yield frame, scale, thresh


def istarmap(self, func, iterable, chunksize=1):
    self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = pool.Pool._get_tasks(func, iterable, chunksize)
    result = pool.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(
                result._job, pool.starmapstar, task_batches
            ),
            result._set_length,
        )
    )

    return (item for chunk in result for item in chunk)


if __name__ == "__main__":
    max_workers = 10
    backend = "spawn"
    frames = [i for i in range(10)]
    new_frames = []
    method = "PoolIter" # ProcessPoolIter, PoolIter, Pool
    if max_workers is None or max_workers > 0:
        if method == "Pool":
            start_time = time.time()

            multiprocessing.pool.Pool.istarmap = istarmap
            mp_ctx = multiprocessing.get_context(backend)
            worker_pool = mp_ctx.Pool(processes=max_workers)

            new_frames = list(
                tqdm(
                    worker_pool.istarmap(
                        func=_mp_worker_fn2,
                        iterable=_iter_fn(frames, 1, 3, {"misc": 1})
                    ), 
                    total=len(frames), 
                    desc="tbd",
                )
            )

            worker_pool.close()
            # 调用join之前，先调用close函数，否则会出错。
            # 执行完close后不会有新的进程加入到pool, join函数等待所有子进程结束
            worker_pool.join()
            worker_pool.terminate()

            print(new_frames)
            print(time.time() - start_time)

        elif method == "PoolIter":
            start_time = time.time()
            mp_ctx = multiprocessing.get_context(backend)

            worker_pool = mp_ctx.Pool(
                processes=max_workers,
                initializer=_worker_initializer,
                initargs=({"misc": 1}, 1, 3),
            )

            worker_iter = MultiWorkerIter(
                input_iter=frames,
                worker_fn=_mp_worker_fn,
                client=worker_pool,
                keep_order=True,
            )

            for data in tqdm(worker_iter, desc="desc"):
                new_frames.append(data)

            worker_pool.close()
            worker_pool.join()
            worker_pool.terminate()

            print(new_frames)
            print(time.time() - start_time)

        elif method == "ProcessPoolIter":
            start_time = time.time()
            mp_ctx = multiprocessing.get_context(backend)

            worker_pool = ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=mp_ctx,
                initializer=_worker_initializer,
                initargs=({"misc": 1}, 1, 3),
            )

            worker_iter = MultiWorkerIter(
                input_iter=frames,
                worker_fn=_mp_worker_fn,
                client=worker_pool,
                keep_order=True,
            )

            for data in tqdm(worker_iter, desc="desc"):
                new_frames.append(data)

            worker_pool.shutdown()

            print(new_frames)
            print(time.time() - start_time)

        else:
            raise NotImplementedError()

    else:
        start_time = time.time()
        _worker_initializer({"misc": 1}, 1, 3)
        for data in frames:
            new_frames.append(_mp_worker_fn(data))

        print(new_frames)
        print(time.time() - start_time)
```
