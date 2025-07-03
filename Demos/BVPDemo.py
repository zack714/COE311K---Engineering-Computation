import numpy as np
import matplotlib.pyplot as plt

x0 = 0.0
L = 10.0
n = 4
x = np.linspace(x0,L,n)
fx = x**2-1
#our solution
T = np.zeros(n)
T0 = 0.0
TL = 10.0
dx= [1]-x[0]

#form our linear set of equations
A = np.zeros((n,n))
b = np.zeros((n))
#fill out non-boundary rows
for i in range(1,n-1):
    A[i,i-1] = 1/(dx**2)
    A[i,i] = -2.0/(dx**2)
    A[i,i+1] = 1.0/(dx**2)
    '''
#just simply replace 1st and last rows
'''
#destroys symmetry but easier to code
A[0,0] = 1.0/(dx**2)
A[-1,-1] = 1.0/(dx**2)

#instead, preserve symmetry by subtracting over our BC
#modify columns in matrix
b[1] -= A[1,0]*T0
A[1,0] = 0.0
b[-2] -= A[-2,-1]*TL
A[-2,-1] = 0.0
print(A)

exit()

#set RHS vector
b[1:n-1] = fx[1:n-1]
b[0] = T0/(dx**2)
b[-1] = TL/(dx**2)

#solve and plot results
T = np.linalg.solve(A,b)

T_analytic = x**4/12.0-x**2/2.0-(77+1.0/3.0)*x

plt.plot(x,T,'--',label="Approximate solution with centered F-D")
plt.plot(x,T_analytic,label="Exact")
plt.xlabel('x(m)')
plt.ylabel('T (Celsius)')
plt.title("Temperature of beam for made up heat equation")
plt.grid()
plt.legend()
plt.show()
print(T)