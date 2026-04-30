# 基于三层MLP的EuroSAT遥感图像分类实验

## 项目简介
本项目使用 Python 从零搭建**三层全连接神经网络（MLP）**，
对 EuroSAT 遥感数据集进行 10 类地物图像分类实验。
实现数据预处理、模型训练、断点续训、结果可视化、错例分析等完整流程。

## 项目文件结构
- data_loader.py     数据集加载与预处理
- mlp_model.py       三层MLP网络模型定义
- train_main.py      主训练脚本、训练日志、绘图
- README.md          项目说明文件

## 运行环境
- Python 3.10
- 依赖库：numpy、matplotlib、scikit-learn、pillow

安装依赖：
pip install numpy matplotlib scikit-learn pillow

## 运行方式
1. 直接运行训练主程序
python train_main.py

2. 支持断点续训
程序会自动检测 best_model.pkl，
有模型就接着训练，没有就从头开始训练。

## 实验结果
- 模型：三层全连接 MLP
- 测试集分类准确率：约 57%
- 输出内容：训练损失曲线、验证准确率曲线、混淆矩阵、权重可视化、错例样本分析


## 实验亮点
1. 加入异常处理，解决模型文件损坏报错问题
2. 全程中文训练日志，易观察训练过程
3. 自动保存模型，支持断电继续训练
4. 完整可视化与错例分析，满足课程实验要求
