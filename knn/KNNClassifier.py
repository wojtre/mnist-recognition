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
