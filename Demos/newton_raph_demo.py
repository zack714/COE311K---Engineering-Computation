import numpy as np
import matplotlib.pyplot as plt

#Newton raphson function
def my_newton_raphson(f,dfdx,xo,tol=1e-5,maxit=15):
    #assumes f and dfdx are functions of a single variable

    #save the old guess for error eval
    x_old = xo
    error = 99999.
    it = 0
    x_history = [x_old]
    #while loop until we converge
    while(error>tol and it<maxit):

        #update 
        x = x_old - (f(x_old)/dfdx(x_old))
        #evaluate error
        err = np.absolute((x-xold)/x)
        #update xold
        xold=x
        #update iteration number 
        it+=1
        print("Iteration ",str(it)," x = ",str(x)," err = ",str(err))

    return x, f(x), err,it,x_history

#define my nonlinear function here
f = lambda x:x**3-1
dfdx = lambda x: 3*(x**2)
x0 = 3
x,fx_root,err,it_no,x_history = my_newton_raphson(f,dfdx,x0)

my_newton_raphson(f,dfdx,x0)
nplot = 100
x_plot = np.linspace(-1,3,nplot)
yplot = f(x_plot)

plt.plot(x_plot,y_plot,label = "f(x)")
'''
#scatter plot the points from the newton raphson
plt.scatter(x_history,np.zeros(len(x_history)),label = "newton iterations")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Newton iterations for f(x) = x**3 - 1 with x_0 = 3")
plt.grid()'
'''

#simple gradient descent
def my_simple_gradient_descent(f,dfdx,xo,tol=1e-5,maxit=15,gamma = 0.5):
    x_old = xo
    err = 9999999.
    it = 0
    while(err>tol and it<maxit):
        x = x_old-gamma*dfdx(x_old)
        #evaluate error
        err = np.absolute((x-xold)/x)
        #update xold
        xold=x
        #update iteration number 
        it+=1
        #print("Iteration ",str(it)," x = ",str(x)," err = ",str(err))

f = lambda x: x**3-1
dfdx = lambda x:2*(x)
x0=3
x_opt,fx_opt,err,it,x_history= my_simple_gradient_descent(f,dfdx,xo,tol=1e-5,maxit=15,gamma = 0.5)
nplot = 100
x_plot = np.linspace(-1,3,nplot)
yplot = f(x_plot)

plt.plot(x_plot,y_plot,label = "f(x)")

#scatter plot the points from the newton raphson
y_history = f(x_history)
plt.scatter(x_history,np.zeros(len(x_history)),label = "newton iterations")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Gradient descent ")
plt.grid()