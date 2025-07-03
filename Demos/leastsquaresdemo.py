import numpy as np
import matplotlib.pyplot as plt

#numbetr of data points
m=100
noise_mean = 0.0
noise_dev = 0.5
data_slope = 5.
data_intercept = 1.0
x = np.linspace(0,1,m)

#generate some noisy data
noise = np.random.normal(noise_mean,noise_dev,m)
y = data_slope*x+data_intercept+np.random.normal(noise_mean,noise_dev,m)

#plot to see what this looks like
plt.scatter(x,y)


#generate least squares fit y = a1 + a2x
#form normal equations
A = np.zeros((2,2))
A[0,0] = m
A[1,0] = np.sum(x)
A[0,1] = np.sum(x)
A[1,1] = np.sum(x*x)


#rhs
b = np.zeros(2)
b[0] = np.sum(y)
b[1] = np.sum(x*y)

print("Old A",A)
print("Old B",b)

#instead use Z^TZ a = Z^Ty
#recall function is y = a1+a2x
Z = np.ones((m,2))
Z[:,1] = x

A = np.matmul(Z.T,Z)
b = np.matmul(Z.T,y)

print("New A",A)
print("New B",b)
#solve
a = np.linalg.solve(A,b)
print("y = ",str(a[0])," + ",str(a[1])," x\n")

#plot line on top of scatter set
plt.plot(x,a[0]+a[1]*x,'-r')
plt.show()
