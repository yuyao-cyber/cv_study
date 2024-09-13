import numpy as np

class CPCA(object):
    def __init__(self,X,K):
        self.X = X
        self.K = K
        self.centrX = []
        self.C = []
        self.U = []
        self.Z = []
        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        centrlX = []
        mean = np.array([np.mean(centrl) for centrl in self.X.T])
        centrlX = self.X - mean
        return centrlX

    def _cov(self):
        ns = np.shape(self.centrX)[0]
        C = np.dot(self.X.T, self.X)/(ns-1)
        return C

    def _U(self):
        a,b = np.linalg.eig(self.C)
        ind = np.argsort(-1*a)
        UT = [b[:,ind[i]] for i in range(self.K)]
        print("UT is ", UT)
        U = np.transpose(UT)
        return U
    
    def _Z(self):
        Z = np.dot(self.X, self.U)
        return Z
    
if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X,K)