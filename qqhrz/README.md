# qqhrz
对动手学深度学习pytorch版的代码实现和一些图神经网络的代码实现
## 1. 简介
这个包是对动手学深度学习pytorch版的代码实现和一些图神经网络的代码实现，由本人书写，存在着许多不足，希望大家见谅。写这个包既是对自己的练习，也是希望可以给同组师弟、师妹们一些帮助。
## 2. 包的结构
qqhrz包中包含两个模块：ztorch和zgnn
### 2.1 ztorch
深度学习pytorch版的代码实现。可以通过”from qqhrz import ztorch as qz“进行调用。
### 2.2 zgnn
对一些图神经网络的代码实现。可以通过”from qqhrz import zgnn as zg“进行调用。
## 3. 依赖
使用本包时，需要先安装pytorch==1.12.0和torchvision==0.13.0，cpu版和gpu版均可，建议安装gpu版本。
## 4. 版本更新
0.0.7：加入Transformer模型和BERT模型以及BERT模式输入数据的处理。<br>
0.0.8：解决了运行速度慢的问题，原因是其中有些张量是在cpu上的，导致在cpu上计算，速度慢。<br>
0.1.0：加入zgnn模块，包含对一些图神经网络的实现。
## 5. 希翼
希望自己能够不断完善，不断改进代码，提升自己的能力，给大家一些帮助。