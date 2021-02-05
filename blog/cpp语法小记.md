# cpp 语法小记

### #ifndef, #define 用法
例如对于头文件 stdio.h, 加上 #ifndef, #define, 是防止多重定义。
```cpp
#ifndef _STDIO_H_
#define _STDIO_H_
......
#endif
```

### 两种include方式
```
#include <iostream> 搜索系统函数库
#include "max.h" 先搜索当前目录，再搜索系统函数库
```
