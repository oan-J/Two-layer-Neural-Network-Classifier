import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from util import SGD, shuffle

# 数据集预处理
(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True)
x_train, y_train = x_train[:2000], y_train[:2000]  # 取太多运行太慢，不取多效果不好，等着吧o.O
x_train, y_train = shuffle(x_train, y_train)

# 划分训练集和验证集
rate = 0.2
val_num = int(x_train.shape[0] * rate)
x_train, x_val = x_train[val_num:], x_train[:val_num]
y_train, y_val = y_train[val_num:], y_train[:val_num]


# 找最优超参
def train2search(lr, l2_lambda, hidden_size):
    network = TwoLayerNet(input_size=784, hidden_size_list=[hidden_size, hidden_size],
                          output_size=10, l2_lambda=l2_lambda)
    optimizer = SGD(lr=lr)
    train_size = x_train.shape[0]
    batch_size = 50  # cbc:试几十到几百之间

    train_loss_list = []
    test_loss_list = []
    iterations_per_epoch = max(train_size // batch_size, 1)
    best_val_loss = float('inf')
    patience = 0
    max_patience = 5

    # 早停
    i = 0
    while True:
        batch = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch]
        y_batch = y_train[batch]

        grads = network.gradient(x_batch, y_batch)
        optimizer.update(network.params, grads)

        if i % iterations_per_epoch == 0:
            train_loss = network.loss(x_train, y_train)
            test_loss = network.loss(x_val, y_val)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            if test_loss < best_val_loss:
                best_val_loss = test_loss
                patience = 0
            else:
                patience += 1
                print("patience:", patience)

            if patience == max_patience:
                print('Early stopping: validation loss did not improve for {} epochs'.format(patience))
                break

        i = i + 1

    return train_loss_list, test_loss_list


# 定义待搜索的超参数取值列表
lr_list = [0.1,0.2,0.3]   # [0.1,0.2,0.3]
# l2_lambda_list = [0.003,0.001,0.002]    # 0.002 [0.003,0.001,0.002]
l2_lambda_list = [0.002]    # 0.002 [0.003,0.001,0.002]
# hidden_size_list = [255,240,250]    # 250 [255,240,250]
hidden_size_list = [250]

best_lr = None
best_l2_lambda = None
best_hidden_size = None
best_performance = float('inf')  # 初始设置为正无穷

results_val = {}
results_train = {}

# 网格搜索
for lr in lr_list:
    for l2_lambda in l2_lambda_list:
        for hidden_size in hidden_size_list:

            train_loss_list, val_loss_list = train2search(lr, l2_lambda, hidden_size)
            key = f'lr:{lr}, l2_lambda:{l2_lambda}, hidden_size:{hidden_size}'
            results_val[key] = val_loss_list
            results_train[key] = train_loss_list

            if val_loss_list[-1] < best_performance:
                best_performance = val_loss_list[-1]
                best_lr = lr
                best_l2_lambda = l2_lambda
                best_hidden_size = hidden_size

print("\n************************************************\n")
print("Best hyperparameters:")
print("Learning rate: ", best_lr)
print("L2 lambda: ", best_l2_lambda)
print("Hidden size: ", best_hidden_size)
print("Best performance: ", best_performance)
print("\n************************************************\n")

for key, value in results_val.items():
    print(key + ': ' + str(value))
