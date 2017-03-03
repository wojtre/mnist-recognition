import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np

import knn.KNNClassifier as knn
import utils.mnist as mnist


def test():
    train_data, train_labels = mnist.load_mnist(mode='train', path='data/')
    # test_data, test_labels = mnist.load_mnist(mode='test', path='data/')
    errors = np.array(knn.tune_hyperparams(train_data[:1000], train_labels[:1000]))
    X = errors[:, 0]
    Y = errors[:, 1]
    Z = errors[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def class_error(classifier, test_data, test_labels):
    error = 0
    for i in range(len(test_labels)):
        if classifier.classify(test_data[i]) != test_labels[i]:
            error += 1
    return 100 * error / len(test_labels)


test()
