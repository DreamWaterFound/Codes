#!/usr/bin/python3
#!-*-coding:UTF8-*-

## Python实现PCA

# 包环境准备
import numpy as np
import matplotlib.pyplot as plt


def pca(X,k):#k is the components you want
    """主成成分分析"""
    #mean of each feature
    n_samples, n_features = X.shape
    mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
    #normalization
    norm_X=X-mean
    #scatter matrix -- 这个就是常常提到的那个"散度矩阵"
    scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
    #Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # select the top k eig_vec
    # 注意这句话有点不是好懂,注意ele是eig_pairs中的一个成就能够对应上了
    feature=np.array([ele[1] for ele in eig_pairs[:k]])
    #get new data
    data=np.dot(norm_X,np.transpose(feature))
    return data
 

#  原始数据
X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
 
plt.scatter(X[:,1],X[:,0],color='red')
plt.xticks(())
plt.yticks(())

plt.show()

# print("X=")
# print(X[:,1])
# print(X[:,0])
print(pca(X,1))