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


## Results
- **Train & Test loss**
  ![Image text](https://github.com/oan-J/Two-layer-Neural-Network-Classifier/blob/main/pic/loss.png)
There is a negligible difference between Train Loss and Test Loss, which is hardly noticeable in the picture.\
Here is part of the data.


| Epoch  | Train Loss | Test Loss  |
|--------|------------|------------|
| epoch0 | 6.98266293 | 6.97194756 |
| epoch1 | 2.93364027 | 2.9218476  |
| epoch2 | 1.69614409 | 1.68708984 |
| epoch3 | 1.11730214 | 1.10961794 |
| epoch4 | 0.80365476 | 0.79690414 |
| epoch5 | 0.66260756 | 0.65496229 |
| epoch6 | 0.60566073 | 0.59731014 |
| epoch7 | 0.58063861 | 0.57374801 |
| epoch8 | 0.5320621  | 0.52374169 |
| epoch9 | 0.5350061  | 0.52709259 |
| epoch10| 0.53868387 | 0.53087806 |
| epoch11| 0.55264111 | 0.54647499 |
| epoch12| 0.51887335 | 0.5140945  |
| epoch13| 0.5131956  | 0.5055338  |
| epoch14| 0.54379708 | 0.53731163 |
| epoch15| 0.54798614 | 0.54056323 |



  
- **Accuracy: test_accuracy finally goes to 0.9504**
  ![Image text](https://github.com/oan-J/Two-layer-Neural-Network-Classifier/blob/main/pic/acc.png)
Probably overfitting……\
The best hyperparameters found were:\
Learning rate:  0.3\
L2 lambda:  0.001\
Hidden size:  245\

- **Network parameters of each layer**
  ![Image text](https://github.com/oan-J/Two-layer-Neural-Network-Classifier/blob/main/pic/para.png)




**google drive link：**
https://drive.google.com/drive/folders/1XOJPYTXeKhrMKaIz7dax4JWUAhPpeUS5
