---
layout: post
title:  "Pytorch Quantization 적용"
date:   2022-07-11 18:50:11 +0900
categories: [pytorch, hardware-optimization]
comments: true
---

딥러닝 모델이 실제 device에 deploy 하는데 2가지 문제점이 있다.  
1. 느린 inferfence time
2. 큰 model parameter size

Pytorch는 float32에서 int8로 데이터 크기를 줄여 연산을 하는 Quantization을 제공한다.
직접 짠 모델에서 quantization을 어떻게 적용하는지 알아보자.  
전체코드는 이 [링크](https://github.com/wonbeomjang/blog-code/blob/main/resnet-quantization.py)에 있다.

## Work Flow
1. float32에서 학습시킨 model 혹은 pretrain model을 가져온다.
2. model을 eveluation으로 변경 후 layer fusion을 적용한다.  (Conv + BN + RELU)
3. forward의 input엔 torch.quantization.QuantStub(), output엔 torch.quantization.DeQuantStub()을 적용한다. 
4. quantization configuration을 적용한다.
5. layer를 fuse한다.
6. QAT (quantized aware training) 을 진행한다.
7. 모델을 cpu에 올려놓고 eval mode로 바꾼 후 float32모델을 int8모델로 변환시킨다.

## Code
### 1. Declear Model
모델은 resnet 사용하기로 한다. 그리고 편의를 위해 학습은 미리 시켰다고 가정한다.  
먼저 resnet의 BottleNeck을 선언하고 resnet18을 구현한다.
```python
import torch
from torch import Tensor
import torch.nn as nn
from torch.quantization import fuse_modules


def conv2d(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block


class BottleNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(BottleNeck, self).__init__()
        self.layer1 = conv2d(in_channels, out_channels, kernel_size)
        self.layer2 = conv2d(out_channels, out_channels, kernel_size)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        y = self.layer1(x)
        y = self.layer2(y)
        x = self.conv1x1(x)
        return x + y


class ResNet18(nn.Module):
    def __init__(self, num_classes, bottle_neck: nn.Module = BottleNeck):
        super(ResNet18, self).__init__()

        self.conv1 = conv2d(1, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2)

        self.layer1 = nn.Sequential(bottle_neck(64, 64), bottle_neck(64, 64), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(bottle_neck(64, 128), bottle_neck(128, 128), nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(bottle_neck(128, 256), bottle_neck(256, 256), nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(bottle_neck(256, 512), bottle_neck(512, 512), nn.MaxPool2d(2, 2))

        self.avgpool = self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
```

### 2,3. Deploy Layer Fusion
이후 각 모듈에 layer fusion을 적용한다. 
layer를 건드리지 않고 상속을 쓰면 결과적으로 parameter가 같기 때문에 QuantizableResNet18은 ResNet18의 파라미터를 쓸 수 있다.  
QuantizableBottleNeck에서는 두 tensor를 더하는 연산이 있으므로 기존의 방식이 아닌 FloatFunctional을 이용하야한다.
```python
from torch.nn.quantized import FloatFunctional

class QuantizableBottleNeck(BottleNeck):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(QuantizableBottleNeck, self).__init__(in_channels, out_channels, kernel_size)
        self.float_functional = FloatFunctional()

    def fuse_model(self) -> None:
        fuse_modules(self.layer1, ["0", "1", "2"], inplace=True)
        fuse_modules(self.layer2, ["0", "1", "2"], inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.layer1(x)
        y = self.layer2(y)
        x = self.conv1x1(x)
        return self.float_functional.add(x, y)


class QuantizableResNet18(ResNet18):
    def __init__(self, num_classes: int):
        super(QuantizableResNet18, self).__init__(num_classes, QuantizableBottleNeck)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self) -> None:
        fuse_modules(self.conv1, ["0", "1", "2"], inplace=True)
        for m in self.modules():
            if type(m) is QuantizableBottleNeck:
                m.fuse_model()
```

### 4,5. Apply Quantize Configuration and Fuse Layer

```python
# torch.load resnet18 parameter....

net.eval()
net.qconfig = torch.quantization.get_default_qconfig("fbgemm")
net.fuse_model()
```

### 6,7. QAT, Convert int8 model
QAT를 진행한 다음에 float32모델을 int8모델로 변환시킨다.
```python
net.train()
net = torch.quantization.prepare_qat(net)
train(net, train_loader)

net = net.cpu().eval()
net = torch.quantization.convert(net)
```

## Result
간단하게 MNIST dataset으로 학습시켰다.
epoch 5, image size 224, optimzer Adam을 사용하였다.

<table align="center">
    <tr align="center">
        <td></td>
        <td>Accuracy (%)</td>
        <td>Inference Time (ms)</td>
        <td>Model Parameter (MB)</td>
    </tr>
    <tr align="center">
        <td>ResNet</td>
        <td>98.79</td>
        <td>66</td>
        <td>46.31</td>
    </tr>
    <tr align="center">
        <td>Quantizable ResNet</td>
        <td>96.42</td>
        <td>33</td>
        <td>11.69</td>
    </tr>
</table>
