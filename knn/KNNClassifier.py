import numpy as np
from scipy.stats import mode
from sklearn.decomposition import PCA


class KNNClassifier:
    def __init__(self, data, labels, k, pca_comp=0):
        self.labels = labels
        self.k = k
        self.pca_comp = pca_comp
        if pca_comp > 0:
            self.data = self._pca(data, pca_comp)
        else:
            self.data = data

    def classify(self, x):
        if self.pca_comp > 0:
            x = self.pca.transform(x.reshape(1, -1))
        dists = (self.data - x)
        errors = np.einsum('ij, ij->i', dists, dists)
        indx_nearest = np.argsort(errors)[:self.k]
        nearest = self.labels[indx_nearest]
        return mode(nearest)[0][0]

    def _pca(self, data, pca_comp):
        self.pca = PCA(pca_comp)
        self.pca.fit(data)
        return self.pca.transform(data)


def crossvalidation_error(data, labels, folds, k, pca_comp):
    from testClassifiers import class_error
    errors = []
    fold_size = round(data.shape[0] / folds)
    for i in range(folds):
        training_data = np.append(data[0:i * fold_size], data[(i + 1) * fold_size:], axis=0)
        training_labels = np.append(labels[0:i * fold_size], labels[(i + 1) * fold_size:], axis=0)
        test_data = data[i * fold_size:(i + 1) * fold_size]
        test_labels = labels[i * fold_size:(i + 1) * fold_size]
        knn = KNNClassifier(training_data, training_labels, k, pca_comp)
        errors.append(class_error(knn, test_data, test_labels))
    return np.mean(errors)


def tune_hyperparams(data, labels):
    folds = 10
    errors = []
    for k in range(1, 10):
        for pca_comp in range(20, 100, 10):
            errors.append([k, pca_comp, crossvalidation_error(data, labels, folds, k, pca_comp)])
    return errors
