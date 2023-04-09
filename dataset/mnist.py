import os.path
import gzip
import pickle
import os
import numpy as np

# cbc：调用层级可修改

data_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = os.path.join(dataset_dir, "mnist.pkl")

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def load_label(file_name):
    file_path = os.path.join(dataset_dir, file_name)

    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    print("! Load labels done !")
    return labels


def load_img(file_name):
    file_path = os.path.join(dataset_dir, file_name)

    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)

    print("! Load imgs done !")
    return data


def convert_numpy():
    dataset = {}
    dataset['train_img'] = load_img(data_file['train_img'])
    dataset['train_label'] = load_label(data_file['train_label'])
    dataset['test_img'] = load_img(data_file['test_img'])
    dataset['test_label'] = load_label(data_file['test_label'])

    return dataset


def init_mnist():
    dataset = convert_numpy()

    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("! Create pickle file done !")


def one_hot(X):
    T = np.eye(10)[X]
    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        dataset['train_img'] = dataset['train_img'].astype(np.float32) / 255.0
        dataset['test_img'] = dataset['test_img'].astype(np.float32) / 255.0

    if one_hot_label:
        dataset['train_label'] = one_hot(dataset['train_label'])
        dataset['test_label'] = one_hot(dataset['test_label'])

    if not flatten:
        dataset['train_img'] = dataset['train_img'].reshape(-1, 1, 28, 28)
        dataset['test_img'] = dataset['test_img'].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist()
