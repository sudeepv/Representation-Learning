import numpy as np
import matplotlib.pyplot as plt
import copy

#Maximum Likelihood Estimation

#Function to calculate the Maximum likelood estimates
#Input - X - the input matrix and label - denoting the type of random variable

def MLE(X,label):
    #Binomial Random Variable
    if label == "Binomial":
        #The parameters are : u
        N = X.size
        u_ML = np.sum(X) / N #Equal to the prob that observation = 1 from the give data
        return u_ML
    
    #Poisson Random Variable
    if label == "Poisson":
        N = X.size
        #The parameters are : lambd
        lambd_ML = np.sum(X) / N #Equal to the sample mean
        return lambd_ML
    
    #Exponential Random Variable
    if label == 'Exponential':
        N = X.size
        #The parameters are : beta
        beta_ML = np.sum(X) / N#Reciprocal of the sample mean
        return beta_ML
    
    #Gaussian Random Variable (Univariate)
    if label == 'Gaussian':
        N = X.size
        #The parameters are : u (mean) and sigma (standard deviation)
        u_ML = np.sum(X) / N
        sigma_ML = np.sum(np.square(X - u_ML)) / N
        return u_ML, sigma_ML
    
    #Laplacian Random Variable
    if label == 'Laplacian':
        N = X.size
        #The parameters are loc (mean) and scale(standard deviation)
        loc_ML = np.median(X)
        scale_ML = np.sum(X - u_ML) / N
        return loc_ML, scale_ML

#(d)Gaussian

#Plotting the MLE of mean and variance in the case of univariate gaussian
#u = 0, sigma = 1

N = np.arange(10,10000,100)
mean_ground = 0
sigma_ground = 1
m = []
s = []
real_mean = []
real_sigma = []

for n in N:
    X = np.random.normal(mean_ground, sigma_ground, n)
    val_m , val_s = MLE(X,"Gaussian")
    m.append(val_m)
    s.append(val_s)
    real_mean.append(mean_ground)
    real_sigma.append(sigma_ground)
    
plt.figure()

plt.subplot(2,2,1)
plt.plot(N,m)
plt.plot(N,real_mean)
plt.xlabel("N")
plt.ylabel("mean")

plt.subplot(2,2,3)
plt.plot(N,s)
plt.plot(N,real_sigma)
plt.xlabel("N")
plt.ylabel("sigma")

plt.show()

#Finding the expected values of mean and var

mean_total = 0
var_total = 0
n_iter = 1000
n = 5
for i in range(n_iter):
    temp_mean, temp_var = MLE(np.random.normal(mean_ground, sigma_ground, n),"Gaussian")
    
    mean_total += temp_mean
    var_total += temp_var
    
expected_mean = mean_total / n_iter
expected_sigma = var_total / (n_iter)

print("Actual u :", mean_ground)
print("E(u_ML)", expected_mean) 
print("Actual sigma :", sigma_ground)
print("E(sigma_ML)", expected_sigma)

#Though the expected value of mean_ML is equal to the actual value of actual mean
#The expected value of variance_ML is considerably smaller than the actual variance
#i.e the variance estimate is biased
#We can prove that E(var_ml) = N-1/N * var i.e var is systematically under estimating the variance
