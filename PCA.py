'''
PCA with the Iris dataset – manual example 使用Iris数据来示例PCA主成分分析，使用numpy手工实现和s cikit-learn中的PCA方式实现
'''

import numpy as np
def pca(X):

    # calculate the mean vector
    mean_vector = X.mean(axis=0)

    # calculate the covariance matrix。协方差矩阵是对称矩阵，行数和列数为特征的个数。
    cov_mat = np.cov((X-mean_vector).T)

    # 计算协方差矩阵的特征值
    # calculate the eigenvectors and eigenvalues of our covariance matrix of the iris dataset
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

    #用保留的特征向量来变换原数据，生成新的数据矩阵
    # store the top two eigenvectors in a variable。假如这里选定了前两个特征向量。
    top_5_eigenvectors = eig_vec_cov[:,:5].T
    # show the transpose so that each row is a principal component, we have two rows == two

    # we will multiply the matrices of our data and our eigen vectors together
    Z = np.dot(X, top_5_eigenvectors.T)
    return np.dot(top_5_eigenvectors,Z)
