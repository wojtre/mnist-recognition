import utils.mnist as mnist
import knn.KNNClassifier as knn


def test():
    train_data, train_labels = mnist.load_mnist(mode='train', path='data/')
    test_data, test_labels = mnist.load_mnist(mode='test', path='data/')
    pca_comp = 45
    knn_6 = knn.KNNClassifier(train_data, train_labels, 6, pca_comp)
    print(class_error(knn_6, test_data[:500], test_labels[:500]))


def class_error(classifier, test_data, test_labels):
    error = 0
    for i in range(len(test_labels)):
        if classifier.classify(test_data[i]) != test_labels[i]:
            error += 1
    return 100*error / len(test_labels)


test()
