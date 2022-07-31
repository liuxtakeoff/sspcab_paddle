# CutPaste-paddle

## 目录

- [1. 简介]()
- [2. 数据集和复现精度]()
- [3. 准备数据与环境]()
    - [3.1 准备环境]()
    - [3.2 准备数据]()
    - [3.3 准备模型]()
- [4. 开始使用]()
    - [4.1 模型训练]()
    - [4.2 模型评估]()
    - [4.3 模型预测]()
- [5. 模型推理部署]()
- [6. 自动化测试脚本]()
- [7. LICENSE]()
- [8. 参考链接与文献]()

## 1. 简介
cutpaste是一种简单有效的自监督学习方法，其目标是构建一个高性能的两阶段缺陷检测模型，在没有异常数据的情况下检测图像的未知异常模式。首先通过cutpaste数据增强方法学习自监督深度表示，然后在学习的表示上构建生成的单类分类器，从而实现自监督的异常检测。

**论文:** [CutPaste: Self-Supervised Learning for Anomaly Detection and Localization](https://arxiv.org/pdf/2104.04015v1.pdf)

**参考repo:** [pytorch-cutpaste](https://github.com/Runinho/pytorch-cutpaste)

在此非常感谢`Runinho`等人贡献的[pytorch-cutpaste](https://github.com/Runinho/pytorch-cutpaste) ，提高了本repo复现论文的效率。

**aistudio体验教程:** [CutPaste_paddle](https://aistudio.baidu.com/aistudio/projectdetail/4378924?contributionType=1&shared=1)


## 2. 数据集和复现精度

- 数据集大小：共包含15个物品类别，解压后总大小在4.92G左右
- 数据集下载链接：[mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
- 训练权重下载链接：[logs](https://pan.baidu.com/s/1t7XA3y8a8w6t8w1pdkuIiA) 提取码：j74i
# 复现精度（Comparison to Li et al.）
| defect_type   |    CutPaste (3-way) | Runinho. CutPaste (3-way) | Li et al. CutPaste (3-way) |
|:--------------|--------------------:|-------------------:|-----------------------------:|
| bottle        |                97.7 |               99.6 |                         98.3 |
| cable         |                75.9 |               77.2 |                         80.6 |
| capsule       |                86.2 |               92.4 |                         96.2 |
| carpet        |                97.0 |               60.1 |                         93.1 |
| grid          |                99.7 |              100.0 |                         99.9 |
| hazelnut      |                92.1 |               86.8 |                         97.3 |
| leather       |               100.0 |              100.0 |                        100.0 |
| metal_nut     |                96.7 |               87.8 |                         99.3 |
| pill          |                86.2 |               91.7 |                         92.4 |
| screw         |                78.8 |               86.8 |                         86.3 |
| tile          |                91.8 |               97.2 |                         93.4 |
| toothbrush    |                94.7 |               94.7 |                         98.3 |
| transistor    |                94.1 |               93.0 |                         95.5 |
| wood          |                96.1 |               99.4 |                         98.6 |
| zipper        |                95.9 |               98.8 |                         99.4 |
| average       |                92.2 |               91.0 |                         95.2 |


## 3. 准备数据与环境


### 3.1 准备环境

首先介绍下支持的硬件和框架版本等环境的要求：

- 硬件：GPU显存建议在6G以上
- 框架：
  - PaddlePaddle >= 2.2.0
- 环境配置：直接使用`pip install -r requirements.txt`安装依赖即可。

### 3.2 准备数据

- 全量数据训练：
  - 下载好 [metec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad/) 数据集
  - 将其解压到 **Data** 文件夹下
- 少量数据训练：
  - 无需下载数据集，使用lite_data里的数据即可


### 3.3 准备模型

- 默认使用resnet18预训练模型进行训练，如想关闭,需要传入参数：`python train.py --no_pretrained`

## 4. 开始使用


### 4.1 模型训练

- 全量数据训练：
  - 下载好 [metec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad/) 数据集后，将其解压到 **./Data** 文件夹下
  - 运行指令`python tools/train.py --epochs 10000 --batch_size 32 --cuda True`
- 少量数据训练：
  - 运行指令`python tools/train.py --data_dir lite_data --type lite --epochs 5 --batch_size 4 --cuda False --no_pretrained`
- 部分训练日志如下所示：
```
> python tools/train.py --data_dir lite_data --type lite --epochs 5 --batch_size 4 --cuda False --no_pretrained
Namespace(batch_size=4, cuda='False', data_dir='lite_data', epochs=5, freeze_resnet=20, head_layer=1, lr=0.03, model_dir='logs', optim='sgd', pretrained=False, save_interval=500, test_epochs=-1, type='l
ite', variant='3way', workers=0)
using device: cpu
training bottle
loading images
loaded 209 images
epoch:1/5 loss:1.2578 avg_reader_cost:0.05 avg_batch_cost:3.01 avg_ips:0.75
epoch:2/5 loss:1.6850 avg_reader_cost:0.02 avg_batch_cost:2.81 avg_ips:0.70
epoch:3/5 loss:1.5016 avg_reader_cost:0.02 avg_batch_cost:2.75 avg_ips:0.69
...
``` 


### 4.2 模型评估

- 全量数据模型评估：`python eval.py --cuda True`
- 少量数据模型评估：`python tools/eval.py --data_dir lite_data --type lite --cuda False`
```
> python tools/eval.py --data_dir lite_data --type lite --cuda False
Namespace(cuda='False', data_dir='lite_data', density='sklearn', head_layer=1, model_dir='logs', save_plots=True, type='lite')
evaluating bottle
loading model logs/bottle/final.pdparams
loading images
loaded 8 images
using density estimation GaussianDensitySklearn
bottle AUC: 0.875
average auroc:0.8750
``` 

### 4.3 模型预测（需要预先完成4.1训练及4.2验证）

- 基于原始代码的模型预测：`python tools/predict.py --data_type bottle --img-path images/demo0.png --dist_th 0.5`
- 基于推理引擎的模型预测：
```
python deploy/export_model.py
python deploy/infer.py --data_type bottle --img-path images/demo0.png --dist_th 0.5
```
部分结果如下：
```
> python deploy/export_model.py
inference model has been saved into deploy

> python deploy/infer.py --data_type bottle --img-path images/demo0.png --dist_th 0.5
image_name: images/demo0.png, class_id: 0, prob: 0.07689752858017344
``` 


## 5. 模型推理部署

模型推理部署详见4.3节-基于推理引擎的模型预测。


## 6. 自动化测试脚本

[tipc创建及基本使用方法。](https://github.com/PaddlePaddle/models/blob/release/2.2/tutorials/tipc/train_infer_python/test_train_infer_python.md)


## 7. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 8. 参考链接与文献
**参考论文:** [CutPaste: Self-Supervised Learning for Anomaly Detection and Localization](https://arxiv.org/pdf/2104.04015v1.pdf)

**参考repo:** [pytorch-cutpaste](https://github.com/Runinho/pytorch-cutpaste)
