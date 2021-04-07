import numpy as np
import matplotlib.pyplot as plt
import copy

#Image Source : http://sipi.usc.edu/database/preview/misc/4.2.07.png - '4.2.07.tiff'
Image = plt.imread('4.2.07.tiff')

#K-Means Clustering

#Function takes inputs : x of size = (n,d) and value of K
#Prints the error at the end of each iteration
#Returns the K centroids as a (K,d) size matrix.
#Also returns matrix of original size = (n,d) where each column is replaced by it's corresponding cluster centroid.

def k_means(X,K):
    
    n = X.shape[0]
    d = X.shape[1]
    
    #Creating an inital matrix 'u' of K centroids by random selection
    random_indices = np.random.choice(n, size = K, replace=False)
    u = X[random_indices, :]
    
    itr = 1 #Variable denoting the number of iterations
    centroid_change = 1 #Arbitraty value not equal to 0
    
    #The iterations continue until there is no difference between centroid positions for successive iterations 
    while (centroid_change != 0):
        
        #Another matrix of size (K,d) is created to store the updated u
        u_new = np.zeros((K,d))
        cluster_count = np.zeros(K).astype(int)
        cost = 0
        cluster_labels = np.zeros(n).astype(int)
        
        #Assigning clusters to each of n vectors
        for i in range(n):
            cluster = np.argmin(np.sum(np.square(u - X[[i],:]), axis = 1, keepdims = True))
            
            u_new[cluster,:] += X[i,:]
            cluster_count[cluster] += 1

            cost += np.sum(np.square(X[i,:] - u[cluster,:] / 255))
            cluster_labels[i] = cluster
        
        #u_new gives the updated cluster centroid matrix
        u_new = np.floor(u_new / cluster_count.reshape(K,1))
        
        #Printing the cost in this iteration
        print("Iteration",itr)
        print("cost:",cost) #Note: The pixels are scaled down by 255 while calculating cost to prevent overflow
        centroid_change = np.sum(u_new - u)
        print("change in centroids:", centroid_change)
        
        #Initialising values for next iteration
        itr += 1
        u = u_new
    #Printing the final centroids
    print("Optimised Centroids :")
    print(u)

    #Returning the matrix of original size = (d,n) where each column is replaced by it's corresponding cluster centroid.
    result = np.zeros((n,d))
    for i in range(n):
        result[i,:] = u[cluster_labels[i],:]    
    
    return result.astype(int)

#Example : K = 2
ans = k_means(Image.reshape(-1,3),2)
plt.imshow(ans.reshape(Image.shape))
plt.title("K=2")
