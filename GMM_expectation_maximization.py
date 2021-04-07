import numpy as np
import matplotlib.pyplot as plt

#Generating the GMM data.

#Function Outputs X of shape (N,dim)
def generate(N,dim,K,mean,covs,pi_k):
    nums = np.round(pi_k * N)
    assert sum(nums) == N,"Length of X not equal to N (Rounding Error)"

    X = np.random.multivariate_normal(mean[0],covs[0],int(nums[0]))
    for i in range(1,K):
        to_concat = np.random.multivariate_normal(mean[i],covs[i],int(nums[i]))
        X = np.concatenate([X,to_concat], axis = 0)
    
    return X

#Utility function to calculate the multi variate normal distribution
def normpdf(x, d, mean, covariance):
    det = np.linalg.det(covariance)
    norm_const = 1.0 / (np.power(2*np.pi,d/2) * np.power(det,0.5))
    x_mu = x - mean
    inv = np.linalg.inv(covariance)        
    result = np.exp(-0.5 * np.dot(x_mu,np.dot(inv,x_mu.T)))
    return norm_const * result

#Utility function for K means algorithm (Used for initialising parameters)
def k_means(X,K):
    
    n = X.shape[0]
    d = X.shape[1]
    
    #Creating an inital matrix 'u' of K centroids by random selection
    random_indices = np.random.choice(n, size = K, replace=False)
    u = X[random_indices, :]
    
    centroid_change = 1 #Arbitraty value not equal to 0
    #The iterations continue until there is no difference between centroid positions for successive iterations 
    while (centroid_change != 0):
        
        #Another matrix of size (K,d) is created to store the updated u
        u_new = np.zeros((K,d))
        cluster_count = np.zeros(K).astype(int)
        cluster_labels = np.zeros(n).astype(int)
        
        #Assigning clusters to each of n vectors
        for i in range(n):
            cluster = np.argmin(np.sum(np.square(u - X[[i],:]), axis = 1, keepdims = True))
            u_new[cluster,:] += X[i,:]
            cluster_count[cluster] += 1
            cluster_labels[i] = cluster
            
        #u_new gives the updated cluster centroid matrix
        u_new = np.floor(u_new / cluster_count.reshape(K,1))
        
        centroid_change = np.sum(u_new - u)
        #Initialising values for next iteration
        u = u_new
        
    #converged values for means
    means = []
    for i in range(K):
        means.append(u[[i],:])
    
    #converged values for pi_k
    pi_k = cluster_count / n
    
    #Calculating the covariance matrix
    cluster_data = dict()
    for i in range(n):
        cluster = int(cluster_labels[i])
        if cluster in cluster_data:
            cluster_data[cluster] = np.concatenate([cluster_data[cluster],X[[i],:] - u[[cluster],:]],axis=0)
        else:
            cluster_data[cluster] = X[[i],:] - u[[cluster],:]
    
    cov_mats = [0 for i in range(K)]
    for i in range(K):
        cov_mats[i] = np.dot(cluster_data[i].T,cluster_data[i]) / cluster_count[i]

    return means , cov_mats, pi_k

#The main function for EM algorithm
def EM(X,K):
    
    N = X.shape[0]
    D = X.shape[1]
    
    #Initialising Parameters using k means clustering
    mean , covs , pi_k = k_means(X,K)
    
    itr = 1
    change = 1.0
    while np.abs(change) > 0.001:
        
        #E-step
        gamma_matrix = np.zeros((N,K)) #responsibility matrix
        denoms_gamma = np.zeros(N)
        
        for i in range(N):
            cumulat = 0
            for j in range(K):
                cumulat += pi_k[j] * normpdf(X[i,:],D,mean[j],covs[j])
            denoms_gamma[i] = cumulat
            
        for i in range(N):
            for j in range(K):
                gamma_matrix[i,j] = pi_k[j] * normpdf(X[i,:],D,mean[j],covs[j]) / denoms_gamma[i]
        
        #M-step
        n_k = np.sum(gamma_matrix, axis = 0)

        #Calculating the updated mean
        new_mean = [0 for i in range(K)]
        for i in range(K):
            cumulat = np.zeros((1,D))
            for j in range(N):
                cumulat += gamma_matrix[j,i]*X[[j],:]
            new_mean[i] = cumulat / n_k[i]
        
        #Calculating the updated covariance matrices
        new_covs = [0 for i in range(K)]
        for i in range(K):
            cumulat = np.zeros((D,D))
            for j in range(N):
                cumulat += gamma_matrix[j,i]*(np.dot((X[[j],:] - new_mean[i]).T,X[[j],:] - new_mean[i]))
            new_covs[i] = cumulat/n_k[i]
            
        #Calculating the updated pi_k's
        new_pi_k = n_k / N
        
        new_mean = np.array(new_mean)
        new_covs = np.array(new_covs)
        change = np.sum(mean-new_mean) + np.sum(pi_k-new_pi_k) + np.sum(covs-new_covs)
        #Printing the values after iteration
        print("Iteration:",itr)
        print("Updated means:")
        print(new_mean)
        print("Updated covariances:")
        print(new_covs)
        print("Updated pi_k:")
        print(new_pi_k)
        print("Change:",change)

        
        #Initialising the next iteration
        itr += 1
        mean = new_mean
        pi_k = new_pi_k
        covs = new_covs
        
    return new_mean,new_covs,new_pi_k

#Example :
dim = 2 #Indicates the dimensions of the data
K = 3 #Indicates the number of gaussian components in the GMM
N = 5000 #The number of observations

mean = np.array([[0,0],[-5,-5],[5,0]])
covs = np.array([[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]])
pi_k = np.array([0.2,0.5,0.3])

X = generate(N,dim,K,mean,covs,pi_k)

print("Ground Truth parameters:")
print("K:", K)
print("Mean:")
print(mean)
print("Covariance matrices:")
print(covs)
print("pi_k's:")
print(pi_k)
print(X.shape)


m,c,p = EM(X,K)
#As we can see, the estimated parameters converge to the ground truth parameters.
