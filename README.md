# 计算机视觉lab1
## 使用说明
1. 运行 train2seach.py，得到最佳超参数
2. 超参数代入train.py(已写入)，运行train.py，得到train和test的loss，以及accuracy。存入visualized_pic文件夹。
3. 运行 visualized_params.py，得到输入输出层网络参数
## 文件结构
- dataset 数据及相关
    - 四个.gz文件为mnist数据集原文件
    - mnist.py用于处理数据集数据
- visualized_pic 用于存放可视化结果
    - Accuracy为测试的accuracy曲线
    - Loss为训练和测试的loss曲线
    - W1、W2为输入输出层的网络参数
- train2serach.py 找最优超参数
- train.py 训练
- two_layer_net.py 两层神经网络分类器模型
- util.py 辅助文件
- visualzie_params.py 可视化网络参数


**google drive link：**
https://drive.google.com/drive/folders/1kACV2krk4ewushLdHWHR2NaOhGUA94Qq?usp=share_link