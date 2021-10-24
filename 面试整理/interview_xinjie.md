# 面试问答整理
### python 知识点
#### 深拷贝,浅拷贝
直接赋值：为对象取别名,两个对象的id相同. a=1, b=a
浅拷贝(copy)：创建一个新对象，这个对象有着原始对象属性值的一份精确拷贝。如果属性是基本类型，拷贝的就是基本类型的值，如果属性是引用类型，拷贝的就是内存地址 ，所以如果其中一个对象改变了这个地址，就会影响到另一个对象。
深拷贝(deepcopy)：拷贝父对象,并递归的拷贝原对象所包含的子对象.深拷贝出来的对象与原对象没有任何关联. 2维数组

```python
def deepcopy(entry):
  if isinstance(entry, dict):
    return {k: deepcopy(v) for k, v in entry.items()}
  elif isinstance(entry, list):
    return [deepcopy(item) for item in entry]
  elif isinstance(entry, tuple):
    return (deepcopy(item) for item in entry)
  else:
    return entry
```

#### GIL
GIL 是python的全局解释器锁，同一进程中假如有多个线程运行，一个线程在运行python程序的时候会霸占python解释器（加了一把锁即GIL），使该进程内的其他线程无法运行，等该线程运行完后其他线程才能运行。如果线程运行过程中遇到耗时操作，则解释器锁解开，使其他线程运行。所以在多线程中，线程的运行仍是有先后顺序的，并不是同时进行。

多进程中因为每个进程都能被系统分配资源，相当于每个进程有了一个python解释器，所以多进程可以实现多个进程的同时运行，缺点是进程系统资源开销大。


#### *args, **kwargs
当我们不知道向函数传递多少参数时，比如我们向传递一个列表或元组，我们就使用*args。

```python
def func(*args):
    for i in args:
        print(i)
func(3,2,1,4,7)
```

在我们不知道该传递多少关键字参数时，使用**kwargs来收集关键字参数。
```python
def func(**kwargs):
    for i in kwargs:
        print(i,kwargs[i])
func(a=1,b=2,c=7)
```

#### 装饰器
装饰器本质上是一个Python函数，它可以让其它函数在不作任何变动的情况下增加额外功能，装饰器的返回值也是一个函数对象. 比如：插入日志、性能测试、事务处理、缓存、权限校验等。有了装饰器我们就可以抽离出大量的与函数功能无关的雷同代码进行重用。装饰器其实就是一个闭包.
```python
def log(text):
    def decorator(func):
        def wrapper(*args, **kw):
            print("{} {}():" % (text, func.__name__))
            return func(*args, **kw)
        return wrapper
    return decorator

@log('execute')
def now():
    print('2015-3-25')
>>> now()
>>> execute now():
>>> 2015-3-25

import time
def time_decorator(func):
    def wrapper(*args, **kw):
        start = time.time()
        result = func(*args, **kw)
        print(time.time() - start)
        return result
    return wrapper
```
面向切面编程. 切面:切入到 指定类or指定方法 的代码片段. 切入点:切入到 哪些类or方法

#### 生成器
生成器是一个返回迭代器的函数，不需要像迭代器的类一样写__iter__()和__next__()方法，只需要一个yiled关键字，每次遇到yield时函数会暂停并保存当前所有的运行信息，返回yield的值,并在下一次执行next()方法时从当前位置继续运行.

#### 迭代器
迭代器是一个可以记住遍历的位置的对象。迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退。迭代器有两个基本的方法：iter() 和 next()。

#### python 内存管理
参考: https://juejin.im/post/5ca2471df265da307b2d45a3

##### 内存池机制
**内存池机制**，用于对内存的申请和释放管理。内存池的概念就是预先在内存中申请一定数量的，大小相等的内存块留作备用，当有新的内存需求时，就先从内存池中分配内存给这个需求，不够了之后再申请新的内存. 这样做最显著的优势就是能够减少内存碎片，提升效率。
##### 垃圾回收机制
**垃圾回收机制**，一是找到内存中无用的垃圾对象资源，二是清除找到的这些垃圾对象，释放内存给其他对象使用。Python 采用了`引用计数`为主,`标记清除`和`分代回收`为辅助策略。
###### 1 引用计数
`引用计数`: 每一个对象,都会有一个计数字段.当一个对象有新的引用时，它的ob_refcnt就会增加，当引用它的对象被删除，它的ob_refcnt就会减少. 一旦对象的引用计数为0，该对象被回收，对象占用的内存空间将被释放.
```sh
优点:
1. 简单
2. 实时性：一旦没有引用，内存就直接释放了。不用像其他机制等到特定时机。

缺点:
1. 需要额外的空间维护引用计数。
2. 不能解决对象的循环引用。(主要缺点)
```
```
>>>a = { } #对象A的引用计数为 1
>>>b = { } #对象B的引用计数为 1
>>>a['b'] = b  #B的引用计数增1
>>>b['a'] = a  #A的引用计数增1
>>>del a #A的引用减 1，最后A对象的引用为 1
>>>del b #B的引用减 1, 最后B对象的引用为 1
```

###### 2 标记清除
标记清除主要是解决循环引用问题。
标记清除算法是一种基于追踪回收（tracing GC）技术实现的垃圾回收算法。分为两个阶段：第一阶段是标记阶段，GC会把所有的活动对象打上标记，第二阶段是把那些没有标记的对象非活动对象进行回收。
那么GC又是如何判断哪些是活动对象哪些是非活动对象的呢？对象之间通过引用（指针）连在一起，构成一个有向图，对象构成这个有向图的节点，而引用关系构成这个有向图的边。从根对象（root object）出发，沿着有向边遍历对象，可达的（reachable）对象标记为活动对象，不可达的对象就是要被清除的非活动对象。根对象就是全局变量、调用栈、寄存器。
![20200521_230432_58](assets/20200521_230432_58.png)
在上图中，我们把小黑圈视为全局变量，也就是把它作为root object，从小黑圈出发，对象1可直达，那么它将被标记，对象2、3可间接到达也会被标记，而4和5不可达，那么1、2、3就是活动对象，4和5是非活动对象会被GC回收。
标记清除算法作为 Python 的辅助垃圾收集技术主要处理的是容器对象(container)，比如list、dict、tuple等，因为对于字符串、数值对象是不可能造成循环引用问题。Python使用一个双向链表将这些容器对象组织起来。
Python 这种简单粗暴的标记清除算法也有明显的缺点：清除非活动的对象前它必须顺序扫描整个堆内存，哪怕只剩下小部分活动对象也要扫描所有对象。

###### 3 分代回收
分代回收是一种以空间换时间的操作方式。
Python将内存根据对象的存活时间划分为不同的集合，每个集合称为一个代，Python将内存分为了3“代”，分别为年轻代（第0代）、中年代（第1代）、老年代（第2代），他们对应的是3个链表，它们的垃圾收集频率与对象的存活时间的增大而减小。新创建的对象都会分配在年轻代，年轻代链表的总数达到上限时，Python垃圾收集机制就会被触发，把那些可以被回收的对象回收掉，而那些不会回收的对象就会被移到中年代去，依此类推，老年代中的对象是存活时间最久的对象，甚至是存活于整个系统的生命周期内。同时，分代回收是建立在标记清除技术基础之上。分代回收同样作为Python的辅助垃圾收集技术处理容器对象。

#### list 底层
Python中的list是一个动态数组，它储存在一个连续的内存块中，随机存取的时间复杂度是O(1)，但插入和删除时会造成内存块的移动，时间复杂度是O(n)。同时，当数组中内存不够时，会重新申请一块内存空间并进行内存拷贝。
PyListObject五个属性: ob_refcnt, *obtype, ob_size, **ob_item, allocated.*
python的列表总是会被频繁的添加或者删除元素，因此频繁的申请释放内存显然是不明智的,所以python的列表在创建时总是会申请一大块内存，申请的内存大小就记录在 allocated 上, 已经使用的就记录在 ob_size.
当通过 PyObject_GC_New 创建列表之后，其实里面的元素都是null.
list 赋值 步骤
1 参数类型检查
2 索引 有效性检查 不可超出索引
3 设置元素
list insert 步骤
1 参数检查
2 从新调整列表容量 通过 list_resize 方法确定 是否需要申请内存
3 确定插入点
4 插入元素 (列表插入时 都会将后面的位置的元素重新移动)
list append 步骤
1 参数检查
2 容量检查
3 调用 list_resize 方法检查是否需要申请内存
4 添加元素

参考:https://blog.csdn.net/lucky404/article/details/79596319
https://juejin.im/post/595f0de75188250d781cfd12
http://wklken.me/posts/2014/08/10/python-source-list.html

#### dict 底层实现
哈希表，根据键Key, 直接对储存位置进行访问的数据结构。存储位置可直接通过哈希函数计算。
常用的哈希函数:
1. 直接定置法. 取关键字的某个线性函数值为散列地址. hash(k)=ak+b，a,b为常数.
2. 除留余数法. 取关键字除p的余数为哈希地址. hash(k) = k mod p, p<=m. m 为哈希表长度.
3. 数字分析法. 取关键字的若干数位组成哈希地址.

随着数据的增加,当通过哈希函数计算的存储地址已经有值了,会发生哈希冲突. python 通过开放定址法解决哈希冲突, JAVA hashMap通过拉链法, 此外还有再构建哈希函数等方法.
- 开放定址法: 产生哈希冲突时, python 通过一个二次探测函数(增量序列:线性,平方,伪随机), 计算下一个候选位置, 当下一个位置可用, 将数据插入该位置, 如果不可用则再次调用探测函数, 获得下一个候选位置.
开放定址法存在的问题: 通过多次使用二次探测函数f(增量序列)，每一个位置对上一个位置都有依赖, 这形成了一个 ‘冲突探测链’, 当需要删除探测链上中间的某个数据时, 会导致探测链断裂, 无法访问到后序位置. 所以采用开放定地法，删除链路上的某个元素时，不能真正的删除元素，只能‘伪删除’.
python字典的三种状态 Unused, Active, Dummy. 当字典中的 key 和 value 被删除后字典不能从Active 直接进入 Unused 状态 否则会出现冲突链路中断,实际上python进行删除字典元素时，会将key的状态改为Dummy ,这就是 python的 ‘伪删除’.
- 拉链法: 将通过哈希函数映射到同一个存储位置的所有元素保存在一个链表中. JAVA 1.8之后, 当链表长度超过阈值时, 将链表转为红黑树.

载荷因子 = 填入表中的元素个数 / 散列表的长度
对于开放定址法，载荷因子很重要，应严格限制在0.7-0.8以下。超过0.8，查表时的缓存不命中（cache missing）按照指数曲线上升。超过载荷因子阈值, 需要resize扩容哈希表。(扩容后hashcode需要重新计算)

python字典源码(https://github.com/python/cpython/blob/master/Objects/dictobject.c, https://github.com/python/cpython/blob/master/Include/dictobject.h)
参考: https://blog.csdn.net/lucky404/article/details/79606089
https://zh.wikipedia.org/wiki/%E5%93%88%E5%B8%8C%E8%A1%A8
https://coolcao.com/2019/07/17/hashmap/

#### + 与 join 的区别
字符串是不可变对象，当用操作符+连接字符串的时候，每执行一次+都会申请一块新的内存，然后复制上一个+操作的结果和本次操作的右操作符到这块内存空间，因此用+连接字符串的时候会涉及好几次内存申请和复制。而join在连接字符串的时候，会先计算需要多大的内存存放结果，然后一次性申请所需内存并将字符串复制过去，这是为什么join的性能优于+的原因。所以在连接字符串数组的时候，我们应考虑优先使用join。
#### is 与 == 区别
- is 判断id是否相等
- == 判断值是否相等
> 1 is 1.0　False
> 1 == 1.0　True

#### type, isinstance 区别
isinstance才能用于含继承关系的判断
```
class A:
    pass
class B(A):
    pass
isinstance(A(), A)    # returns True
type(A()) == A        # returns True
isinstance(B(), A)    # returns True
type(B()) == A        # returns False
```

TODO: python 知识点 http://vissssa.gitee.io/blog/posts/e11f968c/
#### 单例模式
好处: 对于频繁使用的对象，可以省略创建对象所花费的时间，这对于那些重量级对象而言，是非常可观的一笔系统开销；由于 new 操作的次数减少，因而对系统内存的使用频率也会降低，这将减轻 GC 压力，缩短 GC 停顿时间。

单例模式保证了在程序运行中该类只实例化一次，并且提供了一个全局访问点。
`__new__()`是一个静态方法,会返回一个创建的实例,在`__init__()`之前被调用，用于生成实例对象。利用这个方法和类的属性的特点可以实现设计模式的单例模式。
线程安全的单例模式.
多线程环境下，由于单例模式总是会去判断 实例是否被创建，但是多个线程有可能会拿到相同的结果，这样就无法实现单例模式了，因此遇到多线程的环境时，需要加锁。加了锁之后，每个线程判断 if cls.instance is None 这里就变成了线程安全。因此可以实现多线程环境下，始终只有一个实例.
```python
import threading
def synchronized(func):
    func.__lock__ = threading.Lock()
    def lock_func(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)
    return lock_func

class Singleton(object):
    instance = None
    @synchronized
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

if __name__ == "__main__":
    a = Singleton(3)
    print("a单例! id为 %s" % id(a))
    b = Singleton(4)
    print("b单例! id为 %s" % id(b))
```

1 使用`__new__`方法
```python
class Singleton(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            orig = super(Singleton, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
        return cls._instance

class MyClass(Singleton):
    a = 1
```
2 装饰器版本
```python
def singleton(cls):
    instances = {}
    def getinstance(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]
    return getinstance

@singleton
class MyClass:
  ...
```
3 import方法
```python
# mysingleton.py
class My_Singleton(object):
    def foo(self):
        pass
my_singleton = My_Singleton()

# to use
from mysingleton import my_singleton
my_singleton.foo()
```

#### 静态方法(staticmethod),类方法(classmethod),实例方法,普通方法
```python
def foo(x):
    print "executing foo(%s)"%(x)

class A(object):
    def foo(self,x):
        print "executing foo(%s,%s)"%(self,x)

    @classmethod
    def class_foo(cls,x):
        print "executing class_foo(%s,%s)"%(cls,x)

    @staticmethod
    def static_foo(x):
        print "executing static_foo(%s)"%x
```
self和cls是对类或者实例的绑定,对于一般的函数来说我们可以这么调用foo(x),这个函数就是最常用的,它的工作跟任何东西(类,实例)无关.对于实例方法,我们知道在类里每次定义方法的时候都需要绑定这个实例,就是foo(self, x),为什么要这么做呢?因为实例方法的调用离不开实例,我们需要把实例自己传给函数,调用的时候是这样的a.foo(x)(其实是foo(a, x)).类方法一样,只不过它传递的是类而不是实例,A.class_foo(x)

#### 类变量,实例变量
类变量：是可在类的所有实例之间共享的值（也就是说，它们不是单独分配给每个实例的）
实例变量：实例化之后，每个实例单独拥有的变量。
```python
class Test(object):
    num_of_instance = 0
    def __init__(self, name):
        self.name = name
        Test.num_of_instance += 1

if __name__ == '__main__':
    print Test.num_of_instance   # 0
    t1 = Test('jack')
    print Test.num_of_instance   # 1
```

#### Python中单下划线和双下划线
- xx：公有变量
- _xx：私有化属性或方法，但还是可以在外部被直接调用
- __xx：私有化属性或方法，无法在外部直接访问
- __xx__：前后双下划线，系统定义名字（这就是在python中强大的魔法方法）
- xx_：后置单下划线，用于避免与Python关键词的冲突

```python
>>> class MyClass():
...     def __init__(self):
...         self.__superprivate = "Hello"
...         self._semiprivate = ", world!"
...
>>> mc = MyClass()
>>> print mc.__superprivate
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: myClass instance has no attribute '__superprivate'
>>> print mc._semiprivate
, world!
>>> print mc.__dict__
{'_MyClass__superprivate': 'Hello', '_semiprivate': ', world!'}
```

#### Python中的作用域
本地作用域（Local）→当前作用域被嵌入的本地作用域（Enclosing locals）→全局/模块作用域（Global）→内置作用域（Built-in）

#### 面向对象
继承可以把父类的所有功能都直接拿过来，这样就不必重零做起，子类只需要新增自己特有的方法，也可以把父类不适合的方法覆盖重写。
对于静态语言（例如Java）来说，如果需要传入Animal类型，则传入的对象必须是Animal类型或者它的子类，否则，将无法调用run()方法。
对于Python这样的动态语言来说，则不一定需要传入Animal类型。我们只需要保证传入的对象有一个run()方法就可以了：

#### python 解释器原理
Python是一种解释型语言，它的源代码不需要编译，可以直接从源代码运行程序。Python解释器将源代码转换为字节码，然后把编译好的字节码转发到Python虚拟机（Python Virtual Machine，PVM）中执行。
当我们执行Python代码的时候，在Python解释器用四个过程“拆解”我们的代码：
1. 词法分析，如果你键入关键字或者当输入关键字有误时，都会被词法分析所触发，不正确的代码将不会被执行。
2. 语法分析，例如当"for i in test:"中，test后面的冒号如果被写为其他符号，代码依旧不会被执行。
3. 生成.pyc字节码文件。
4. 将编译好的字节码转发Python虚拟机中进行执行：由PVM来执行这些编译好的字节码。

字节码bytecode的好处就是加载快，而且可以跨平台，同样一份bytecode，只要有操作系统平台上有相应的Python解释器，就可以执行，而不需要源代码。不同版本的Python编译的字节码是不兼容的.一般来说一个Python语句会对应若干字节码指令，Python的字节码是一种类似汇编指令的中间语言，但是一个字节码指令并不是对应一个机器指令（二进制指令），而是对应一段C代码. 一个Python的程序会有若干代码块组成，例如一个Python文件会是一个代码块，一个类，一个函数都是一个代码块，一个代码块会对应一个运行的上下文环境以及一系列的字节码指令。

参考 https://www.ituring.com.cn/article/507878

#### MVC

- 模型（Model） - 程序员编写程序应有的功能（实现算法等等）、数据库专家进行数据管理和数据库设计(可以实现具体的功能)。
- 视图（View） - 界面设计人员进行图形界面设计。
- 控制器（Controller）- 负责转发请求，对请求进行处理。
 
#### 色彩空间
HSV(色相, 饱和度, 明度), HSL(色相, 饱和度, 亮度), LAB(亮度, 绿到红, 蓝到黄)

#### 霍夫变换
霍夫变换(Hough Transform)可以理解为图像处理中的一种特征提取技术，通过投票算法检测具有特定形状的物体。霍夫变换运用两个坐标空间之间的变换将在一个空间中具有相同形状的曲线或直线映射到另一个坐标空间中的一个点形成峰值，从而把检测任意形状的问题转化为统计峰值问题。
参考: https://github.com/GYee/CV_interviews_Q-A/blob/master/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/15_Hough%E5%8F%98%E6%8D%A2%E6%A3%80%E6%B5%8B%E7%9B%B4%E7%BA%BF%E4%B8%8E%E5%9C%86%E7%9A%84%E5%8E%9F%E7%90%86.md

#### HOG
待整理
https://github.com/GYee/CV_interviews_Q-A/blob/master/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/02_HOG%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86.md

#### LBP
待整理
https://github.com/GYee/CV_interviews_Q-A/blob/master/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/01_LBP%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86.md
