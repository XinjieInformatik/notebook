# pytorch 学习笔记

## 基本语法

### 初始化tensor

```python
#------------- 初始化 --------------
torch.zeros(3, 2)
torch.rand(3, 2)
torch.ones(3, 2, dtype=torch.int)
torch.tensor([3, 1, 2], dtype=torch.float)
x = torch.randn_like(x0, dtype=torch.float) # override dtype, result has same size
```

### 基本操作
```python
#------------- 操作 --------------
x[:2, ...] # 支持所有和 numpy 一样的切片操作
x.size() # = x.shape  return tuple torch.Size([2,3])
x.view(-1) # = x.reshape(-1)
y.add_(x) # _表示会inplace y
# All the Tensors on the CPU except a CharTensor support converting to NumPy and back.
# 注意 numpy_b, tensor_a 共享内存，更改a or b, b or a 也会改变
numpy_b = tensor_a.numpy() # tensor to numpy array
tensor_b = torch.from_numpy(a) # numpy to tensor(CPU)

device = torch.device('cuda')
x = x.to(device) # or x.to('cuda') x.to('cpu')

# requires_grad 使tensor可以backprop
x = torch.ones(2, 2, requires_grad=True)
x.requires_grad_(True)
x.grad # 计算回传到x的梯度
with torch.no_grad():
    # ... 范围内不会backprop

```

### 自定义pytorch的net
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # pytorch 自定义net例子
    # layer 在__init__中写
    # 注意 pooling, reshape(view), activation 等操作不在__init__中
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # forward 里写从输入到out经过的layer与操作
    # pooling, reshape(view), activation 等操作，在这里写，把layer组织起来
    # 可在下方自定义操作如num_flat_features，在forward里用
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
# out 为 forward return的 x
out = net(input)
print(net) # 输出的是__init__中的结构
```

### loss Function

```python
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()
loss = criterion(output, target)
```

### optimizer

```python
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01)

# in the training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```

### pytorch dataset

定义一个自己的dataset, 至少保证如下3个方法 __init__, __getitem__, __len__
然后传给pytorch的DataLoader即可
读取label等一次性准备工作在 __init__ 中写
在遍历dataloader时候对每个image与GT的处理，batch的构造在 __getitem__ 里写

```python
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

# RandomDataset 继承 class Dataset 因此有 transform
mydataset=RandomDataset(input_size, data_size, transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))
rand_loader = DataLoader(dataset=mydataset, batch_size=batch_size, shuffle=True)
```

transformer. 在pytorch中把transforms(同时对label与图片变化)写成callable classes,
然后compose transforms，放入mydataset,放入DataLoader

```python
class Rescale(object):
    # __init__ 决定 Compose 时候的参数输入
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    # sample 照着写
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]
        return {'image': img, 'landmarks': landmarks}

class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
```

```python
composed = transforms.Compose([Rescale(256),
                               ToTensor()])
```

## 进阶
