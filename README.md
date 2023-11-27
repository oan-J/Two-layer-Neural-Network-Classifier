# Two-layer Neural Network Classifier
<!--计算机视觉lab1-->

## Instructions
1. Run _train2seach.py_ to obtain the **optimal hyperparameters**.
2. Input the **optimal hyperparameters** into _train.py_(already done) and run _train.py_ to obtain **train loss**, **test loss** and **accuracy**. Save them into _visualized_pic_ folder.
3. Run visualized_params.py to obtain the input and output layer network parameters.

## File Structure
<!--
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
-->
- **dataset Folder:**
    - The four .gz files contain the original MNIST dataset files.
    - mnist.py is used to process the dataset.

- **visualized_pic Folder:**
    - Accuracy stores the accuracy curve from testing.
    - Loss stores the training and testing loss curves.
    - W1 and W2 represent the network parameters for the input and output layers.

- **train2search.py:**
    - Searching for optimal hyperparameters.

- **train.py:**
    - Training the model.

- **two_layer_net.py:**
    - Implementation of a two-layer neural network classifier model.

- **util.py**

- **visualize_params.py:**
    - Visualizing network parameters.

**google drive link：**
https://drive.google.com/drive/folders/1kACV2krk4ewushLdHWHR2NaOhGUA94Qq?usp=share_link
