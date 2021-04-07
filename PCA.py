import numpy as np
import matplotlib.pyplot as plt
import copy

#Principal Component Analysis

#Takes a (d,N) matrix as input 
#Returns (d,d) matrix whose rows are the principal components
#Returns (d,d) covariance matrix of input X
def PCA(X):
    d = X.shape[0]
    N = X.shape[1]
    mean = np.sum(X, axis =1, keepdims = True) / N
    X_norm = X - mean

    #Covariance matrix of X
    cov_mat_x = np.matmul(X_norm,X_norm.T) / N

    #Performing Eigen decomposition of covariance matrix
    var_along_pc, pc = np.linalg.eig(cov_mat_x)
    
    #To verify the correctness of the algorithm, we can show that that the covariance matrix is diagonalised after applying pca
    
    #Veryfying that the covariance matrix of principal components is diagonal
    Y = np.matmul(pc.T,X_norm)
    cov_mat_y = np.matmul(Y,Y.T) / N
    print("Covariance matrix of Y:")
    print(cov_mat_y)
    
    return var_along_pc, pc

var_along_pc , pc = PCA(Image.reshape(-1,3).T)
print("Principal Components:")
print(pc)
print("Eigen values of X:")
print(var_along_pc)

#As we can see the diagonal elements in the covariance matrix of Y = the eigen values of covariance matrix of X
#This also verifies the correctness of our algorithm
