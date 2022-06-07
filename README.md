# Paddle-UNETR

## 一、简介
参考论文：《UNETR: Transformers for 3D Medical Image Segmentation》[论文链接](https://arxiv.org/abs/2103.10504v3)

近年来，具有收缩路径和扩展路径（例如，编码器和解码器）的全卷积神经网络（FCNN）在各种医学图像分割应用中表现出突出的地位。在这些体系结构中，编码器通过学习全局上下文信息成为一个不可或缺的角色，而此过程中获取的全局上下文表示形式将被解码器进一步用于语义输出预测。尽管取得了成功，但作为FCNN的主要构建模块的卷积层的局限性，限制了在此类网络中学习远程空间相关性的能力。受自然语言处理（NLP）转换器在远程序列学习中的最新成功的启发，将体积（3D）医学图像分割的任务重新设计为序列到序列的预测问题。特别是，我们介绍了一种称为UNEt Transformers （UNETR）的新架构，该架构利用纯Transformers 作为编码器来学习输入量的序列表示并有效地捕获全局多尺度信息。转换器编码器通过不同分辨率的跳跃连接直接连接到解码器，以计算最终的语义分段输出。

[参考项目地址链接](https://github.com/Project-MONAI/research-contributions/tree/master/UNETR/BTCV)

[AI studio 项目地址](https://aistudio.baidu.com/aistudio/projectdetail/3441354)

## 二、复现精度
官方只提供了validation set的测试代码，对于在线测试代码和文件提交格式没有公布说明，所以这里先在validation set下测试精度：

官方repo:76.8%，复现repo:76.3%。

## 三、数据集
在脾脏分割任务上进行实验：

* Target: 13 abdominal organs including 1. Spleen 2. Right Kidney 3. Left Kideny 4. Gallbladder 5. Esophagus 6. Liver 7. Stomach 8. Aorta 9. IVC 10. Portal and Splenic Veins 11. Pancreas 12. Right adrenal gland 13. Left adrenal gland.
* Task: Segmentation
* Modality: CT
* Size: 30 3D volumes (24 Training + 6 Testing)
* Size: BTCV MICCAI Challenge

## 四、环境依赖
paddlepaddle-gpu==2.2.2  cuda 10.2

nibabel==3.1.1

tqdm==4.59.0

## 五、快速开始

### step1: 加载数据集

[BTCV腹部CT数据集下载](https://aistudio.baidu.com/aistudio/datasetdetail/107078)

加载数据集文件放在本repo的dataset/下 

**Install dependencies**
```bash
pip install -r requestments.txt
```

### step2: 训练

```bash
python3 main.py 
```

训练的模型数据和日志会放在本repo的runs/下

### step3: 验证评估

测试时会加载本repo的runs/下保存的模型

验证模型
```bash
python test.py
```

可以[下载训练好的模型数据](https://aistudio.baidu.com/aistudio/datasetdetail/107078)，放到本repo下，然后直接执行验证指令。

训练中包含周期性valset的评估结果放在runs/下

## 六、代码结构与参数说明

### 6.1 代码结构

```
├─dataset  #数据集文件                     
├─einops  #操作张量包                
├─monai  #医学影像深度学习包                      
├─networks #网络结构                        
├─optimizers  #定义优化器                         
├─runs  #模型日志保存
├─utils  #工具
│  main.py  #主文件，用于启动训练                       
│  opts.py  #参数配置                                     
│  requirements.txt  #环境
│  test.py  #测试  
│  trainer.py  #训练                       
```

### 6.2 参数说明

可以在opt.py中设置训练与评估相关参数

## 七、模型信息

关于模型的其他信息，可以参考下表：

| 信息 | 说明 |
| --- | --- |
| 发布者 | Lieber |
| 时间 | 2022.01 |
| 框架版本 | Paddle 2.2.2 |
| 应用场景 | 图像分割 |
| 支持硬件 | GPU、CPU |
| 下载链接 | [最优模型](https://aistudio.baidu.com/aistudio/datasetdetail/107078)|
| 在线运行 | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/3441354)|

