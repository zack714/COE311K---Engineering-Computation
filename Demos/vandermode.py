import numpy as np
import matplotlib.pyplot as plt

#given data, create interpolating polynomial
#with Vandermode matrix

x1 = 0.0
x2 = 1.0
n = 5
x = np.linspace(x1,x2,n)
#come up with some data
y = np.sin(x*np.pi)

#fit data wth order n-1 polynomial
Z = np.zeros((n,n))

#fill in with vandermode matrix
for j in range(n-1,-1,-1):
    Z[:,n-1-j] = x**(j)
#print(Z)

#solve for our coefficients
a = np.linalg.solve(Z,y)
print("Coefficients: ")
print(a)

x_plot = np.linspace(x1,x2,100)
y_plot = a[0]*x_plot**4 + a[1]*x_plot**3 + a[2]*x_plot**2 + a[3]*x_plot + a[4]



plt.scatter(x,y,label = "data")
plt.plot(x_plot,y_plot, label = "interpolant")

#plot the "true" data
y_true = np.zeros(100)
y_true[x_plot>0.5] = 1.0
plt.legend()
plt.show()