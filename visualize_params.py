import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

network = TwoLayerNet(input_size=784, hidden_size_list=[64, 64],
                      output_size=10, l2_lambda=0.05)

network.load_params('params.pkl')

# visualize W1
plt.imshow(network.params['W1'], cmap='Reds', interpolation='nearest')
plt.ylabel("Input Layer")
plt.xlabel("Hidden Layer")
plt.colorbar(label='Parameter Value')
plt.savefig('visualized_pic/W1.png', dpi=100)
plt.savefig('visualized_pic/W1.svg')
plt.show()

# visualize W2
plt.imshow(network.params['W2'], cmap='Blues', interpolation='nearest')
plt.xlabel("Output Layer")
plt.ylabel("Hidden Layer")
plt.colorbar(label='Parameter Value')
plt.savefig('visualized_pic/W2.png', dpi=100)
plt.savefig('visualized_pic/W2.svg')
plt.show()
