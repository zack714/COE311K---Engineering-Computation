#Name: Christian Aruba
#EID: cma3939
#Date: 1/30/2025

import numpy as np
import math
import matplotlib.pyplot as plt
import os

print(os.getcwd())

"""
c_d = drag coeffcient
m = mass
t = time
g = acceleration due to gravity
"""
def exact_velocity(c_d,m,t,g):
    #v = velocity at a certain instant in time
    v = np.sqrt((g*m)/c_d)*np.tanh(np.sqrt((g*c_d)/m)*t)

    return v
#create a linearly spaced vector that ranges from t = 0 to 12 seconds, with time increments of 1/2
time = np.arange(0,13,0.5)

res = exact_velocity(0.25,68.1,time,9.81)
#print(res)

#now we will use a Forward Euler Approximation to estimate the velocity
def forward_Euler_velocity(c_d,m,t,g):
    #the total amount of snapshots in time
    nt = t.size
    #to get v the same size as the time vector
    v = np.zeros(np.shape(t))

    #initial velocity will always be 0
    v[0] = 0

    #a counter to move through th time vector and input values into the vector list
    i = 1

    while i<nt:
        dt = t[i]-t[i-1]
        """
        print("v[i]: "+str(v[i])+", v[i-1]: "+str(v[i-1]))
        """
        #formula to approximate velocity
        v[i] = v[i-1]+dt*(g-(c_d/m)*(v[i-1]**2))
        i+=1

    return v
"""
plt.plot(res,time,"b--",label="Time Step Sizes",linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Exact Velocity (m/s)")
plt.grid()
plt.title("Time vs. Exact Velocity")
plt.show()
plt.savefig("Time vs. Exact Velocity Graph")
"""

#euler method printing
res2 = forward_Euler_velocity(0.25,68.1,time,9.81)
print(res2)


plt.figure(figsize = (8,6))
"""
#plotting the true solution first
plt.plot(time,res,"r--",label="True Solution",linewidth=2)

#then plotting the numerical solution
plt.plot(time,res2,"b--",label="Numerical Method",linewidth=2)
plt.xlabel("Time (s)",size=20)
plt.ylabel("Velocity (m/s)",size=20)
plt.title("Exact Solution vs. Approximation")
plt.grid()
plt.legend()
plt.show()
plt.savefig("True Solution vs. Approximation")
"""
#Problem 5


#create vector holding time step sizes
time_steps = [0.0625, 0.125, 0.250, 0.5, 1, 2]


#start for loop, looping over time step vector


# empty vector that will hold rmse value for each time step
rmse_vals = np.zeros(len(time_steps))
k = 0
for ts in time_steps:

    #vector that holds snapshots in time; it's the "t" in the HW Prompt
    #these snapshots and the amount of them there are will change/decrease as the time step gets larger
    temp_t = np.linspace(0,12,int((12/ts)))

    #length of the vector (or list, in this case)
    nt = len(temp_t)

    #print(str(type(nt)))
    #print("Length of the vector: "+str(nt))

    #these temps will hold the results of the true values and approximations based on the current time step
    temp_trueVal = exact_velocity(0.25,68.1,temp_t,9.81)
    print("Temp t: "+str(temp_t))
    temp_approxVal = forward_Euler_velocity(0.25,68.1,temp_t,9.81)

    print("Temp trueVal: "+str(temp_trueVal))
    print("Temp approxVal: "+str(temp_approxVal))

    #holds the sum
    temp_sum = 0

    #using zip method to get through two lists at the same time
    for trueVal,approxVal in zip(temp_trueVal, temp_approxVal):

            #print("Type of values: "+str(type(approxVal)))
            diff_squared = (trueVal-approxVal)**2

            #print("Type of diff_squared: " + str(type(diff_squared)))

            temp_sum+=diff_squared

    #print("Type of temp_sum: "+str(type(temp_sum)))
    rmse_vals[k] = math.sqrt(temp_sum/nt)

    k = k+1
    print("NEW TIME STEP!")

print("RMSE Values: "+str(rmse_vals))

plt.plot(time_steps,rmse_vals,"g--",label="Time Step Sizes",linewidth=2)
plt.xlabel("Time Steps")
plt.ylabel("RMSE Values")
plt.title("Time vs. Root Mean Squared Error Values")
plt.grid()
plt.show()
plt.savefig()









#call euler approx function and store its result in a temp vector

#ditto for true solution function

#compute the RMSE and store it in a vector


#Bonus problems
def mat_mat_mul(A,B):
    '''
Your code goes here, compute C=AB
'''
#return C
def approximate_sin(x,es=0.0001,maxit=50):
    '''
# computes the Maclaurin series of exponential function
# Inputs
# float: x, value at which series evaluated
# float: es, stopping criterion (default = 0.0001)
# int: maxit, maximum iterations (default = 50)
# Outputs
# float: fx, estimated value of exp(x)
# float: ea, approximate relative error (%)
# int: iter_num, number of iterations
    #return sol,ea,iter_num
'''
