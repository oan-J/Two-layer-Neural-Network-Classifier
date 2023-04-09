import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
from util import SGD

# train
(x_train, y_train), (x_test, y_test) = load_mnist(normalize=True)

network = TwoLayerNet(input_size=784, hidden_size_list=[245, 245],
                      output_size=10, l2_lambda=0.01)
optimizer = SGD(lr=0.3)
max_epochs = 50
train_size = x_train.shape[0]
batch_size = 500

train_loss_list = []
test_loss_list = []
test_acc_list = []

iterations_per_epoch = max(train_size // batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    y_batch = y_train[batch_mask]

    grads = network.gradient(x_batch, y_batch)
    optimizer.update(network.params, grads)

    if i % iterations_per_epoch == 0:
        train_loss = network.loss(x_train, y_train)
        test_loss = network.loss(x_test, y_test)
        test_acc = network.accuracy(x_test, y_test)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        print("epoch" + str(epoch_cnt) + ", train loss:" + str(train_loss) + ",test loss:" + str(test_loss))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

network.save_params('params.pkl')
print('！two_layers_train done ！')


# visualize loss
x = np.arange(max_epochs)
plt.figure(figsize=(20, 10), dpi=70)  # 设置图像大小
plt.plot(x, train_loss_list, color="red", linewidth=2.0, linestyle="-", label="train loss")
plt.plot(x, test_loss_list, color="blue", linewidth=2.0, linestyle="--", label="test loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, max(train_loss_list[0], test_loss_list[0]) + 0.5)
plt.legend(["train loss", "test loss"], ncol=2)
plt.savefig('visualized_pic/Loss.png', dpi=100)
plt.savefig('visualized_pic/Loss.svg')
plt.show()


# visualize accuracy
x = np.arange(max_epochs)
plt.plot(x, test_acc_list)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1.0)
plt.savefig('visualized_pic/Accuracy.png', dpi=100)
plt.savefig('visualized_pic/Accuracy.svg')
plt.show()

print("test_acc_list finally goes to：",test_acc_list[-1])