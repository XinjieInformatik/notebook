# python动态调用cpp

<!-- TOC tocDepth:2..3 chapterDepth:2..6 -->

- [背景](#背景)
- [环境配置](#环境配置)
- [简单示例](#简单示例)
    - [用pybind11进行封装](#用pybind11进行封装)
    - [编译cpp动态库.so](#编译cpp动态库so)
    - [python调用](#python调用)
- [实际示例](#实际示例)
    - [入参 py::object](#入参-pyobject)
    - [出参 py::typing:Dict](#出参-pytypingdict)
    - [example](#example)
    - [CMakeLists.txt](#cmakeliststxt)
    - [setup.py](#setuppy)

<!-- /TOC -->

## 1. 背景
python中调用c++库，可以：
1. 通过subprocess.check_call调用编译好的二进制执行文件，输入输出以文件形式中转；
2. 发布c++库的pip包，同时基于例如pybind11的能力 ，提供相关接口。

方式2优势：1. 维护方便，接口定义清晰； 2. 数据流直接通过对象传递，避免以文件形式中转的IO落盘与黑盒； 3. 数据定义清晰，对于python/c++各自的改动都不大。
该文档对python动态调用c++进行了梳理，能够涵盖绝大多数的相关开发需求，提供很好的借鉴与快速上手指南

## 2. 环境配置
pybind11: https://pybind11.readthedocs.io/en/stable/basics.html
先编译 make install，然后pip也装一下pybind11的python package.

## 3. 简单示例
`add_module`是package名，`add`是函数名
### 3.1. 用pybind11进行封装
```cpp
// add.cpp
#include <pybind11/pybind11.h>

namespace py=pybind11;

int add(int num1, int num2 = 0) {
    int ans = num1 + num2;
    return ans;
}

PYBIND11_MODULE(add_module, m) {
    m.doc() = "pybind11 add example";
    m.def("add", &add, py::arg("num1"), py::arg("num2")=0);
}
```
### 3.2. 编译cpp动态库.so
会生成 e.g. add_module.cpython-38-x86_64-linux-gnu.so
```bash
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) add.cpp -o add_module$(python3-config --extension-suffix)
```
### 3.3. python调用
```python
import sys
sys.path.append("path to add_module.so")
import add_module
result = add_module.add(num1=2, num2=3)
```

## 4. 实际示例
举一个类传递的例子
### 4.1. 入参 py::object
封装的入参需要指定类型，例如：py::typing:str, py::typing:Dict, py::typing:Tuple, py::object等。对于接收到的数据结构，需要.cast<type>()转换成对应具体的数据类型。
### 4.2. 出参 py::typing:Dict
一般组成py::typing:Dict, 然后结合python Dataclass的能力恢复成python对象。
### 4.3. example
python调用某个cpp的函数，写一个接口类`InterfaceClass`, 其中借用pybind11的能力写一个`ConvertfromPyClass1ToClass1`函数，将python对象转为对应的cpp对象，正常处理cpp对象，然后再通过`ConvertfromClass2ToPyClass2Dict`转回dict，python调用接口类接受后通过dataclass恢复成python对象。

```cpp
#include <pybind11/pybind11.h>
#include "MyClass.h"
namespace py = pybind11;

class InterfaceClass {
public:
    InterfaceClass():opt_ptr_(nullptr){
        opt_ptr_=std::make_shared<Lib1::OptClass>();
    }
    // 接收pyClass1的数据结构，以及float参数p，返回对应pyClass2数据结构的一个字典
    py::dict Process(py::object PyClass1_obj1, float p) {
        obj1 = Class1();
        obj2 = Class2();
        // 根据PyClass1的attr("成员变量名")接收到，按照一定规则赋值给c++的Class1对象 by pybind11
        obj1 = ConvertfromPyClass1ToClass1(py::object PyClass1_obj1);
        // 调用库的优化函数
        obj2 = opt_ptr_->OptClassTrajOptimizer(obj1, p);
        // 根据PyClass2与 Class2的数据结构，按照一定规则赋值给对应PyClass2数据结构的py::dict
        py::dict out_dict = ConvertfromClass2ToPyClass2Dict(obj2);
        return out_dict;
    }
private：
    std::shared_ptr<Lib1::OptClass> opt_ptr_;
};

PYBIND11_MODULE(my_module, m) {
    py::class_<InterfaceClass>(m, "InterfaceClass")
        .def(py::init<>())
        .def("process", &InterfaceClass::Process);
}
```
python调用
```python
import my_module
interface_obj = my_module.InterfaceClass()
out_dict2 = interface_obj.process(PyClass1_obj1, p)
PyClass1_obj2 =PyClass2.from_dict(out_dict2)
```

可以通过`pybind11.attr`获取传入对象的属性。
```cpp
py::typing:Dict ConvertfromPyClass1ToClass1(const py::object &input_object) {
// py_tracklet: {timestamp, object_measures}
for (auto item : input_object.attr("tracklet_info").cast<py::dict>()) {
    py::typing::Dict<py::str, py::object> object_measures =
        item.second.cast<py::typing::Dict<py::str, py::object>>();
    for (auto item2 : object_measures) {
        py::object py_sensor_measure = item2.second.cast<py::object>();
        gt_sensor_measure.time_stamp_ = std::stoll(
            py_sensor_measure.attr("timestamp").cast<std::string>());
        gt_sensor_measure.ego_pose_ = convert_numpy_to_eigen_matrix(
            py_sensor_measure.attr("ego_pose").cast<py::array_t<double>>());
        ...
    }
    
    return gt_sensor_measure;
}
```
numpy与Eigen的格式互换
```cpp
Eigen::VectorXd convert_numpy_to_eigen_vector(py::array_t<double> input) {
  py::buffer_info buf_info = input.request();
  if (buf_info.ndim != 1)
    throw std::runtime_error(
        "Incompatible buffer dimension in convert_numpy_to_eigen_vector()!");

  Eigen::VectorXd vec(buf_info.shape[0]);
  std::memcpy(vec.data(), buf_info.ptr, buf_info.size * sizeof(double));

  return vec;
}

Eigen::MatrixXd convert_numpy_to_eigen_matrix(py::array_t<double> input) {
  py::buffer_info buf_info = input.request();
  if (buf_info.ndim != 2)
    throw std::runtime_error(
        "Incompatible buffer dimension in convert_numpy_to_eigen_matrix()!");

  Eigen::MatrixXd mat(buf_info.shape[0], buf_info.shape[1]);
  std::memcpy(mat.data(), buf_info.ptr, buf_info.size * sizeof(double));

  return mat.transpose();
}
```

### 4.4. CMakeLists.txt
```cpp
cmake_minimum_required(VERSION 3.12)
project(my_module)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(pybind11)
pybind11_add_module(my_module MyClass_python.cpp MyClass.h)
```

### 4.5. setup.py
python setup.py bdist_wheel
```python
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

__version__ = '0.0.1'

# This is the C++ extension module definition
ext_modules = [
    Extension(
        'my_module',
        ['MyClass_python.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
]

setup(
    name='my_module',
    version=__version__,
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/my_module',
    description='A test project using pybind11',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
```