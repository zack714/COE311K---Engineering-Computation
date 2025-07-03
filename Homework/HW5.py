import numpy as np
import matplotlib.pyplot as plt
import math as m
error='error'

def my_finite_diff(fx,xi,fd_type):
    '''
    Takes in fx - input data we want to differentiate
    xi - the x-coordinates of fx
    fd_type - a string that is either 'Backward', 'Forward', or 'Centered'
    Be sure to check the size of fx and xi are the same
    if the fd_type is not any of the 3 strings return error
    return dfxi, a vector that is the size of xi-2 that estimates the derivative at
    interior points
    '''
    dfxi = np.zeros(len(xi)-1)
    if len(fx) != len(xi):
        print("ERROR: Vectors xi and fx are not the same length.")
        return -1

    #if user wants a backwards finite difference...
    if fd_type=="Backward":
        for i in range(1,len(xi)-1):
            dfxi[i-1] = (fx[i]-fx[i-1])/(xi[i]-xi[i-1])
    #else if user wants a forward finite difference...
    elif fd_type=="Forward":
        for i in range(1,len(xi)-1):
            dfxi[i-1] = (fx[i+1]-fx[i])/(xi[i+1]-xi[i])
    #else if user wants a centered finite difference
    elif fd_type=="Centered":
        for i in range(1,len(xi)-1):
            dfxi[i-1] = (fx[i+1]-fx[i-1])/(xi[i+1]-xi[i-1])
    #else if the user did not use any of these inputs...
    else:
        print("ERROR: User must imput the one of the 3 strings: 'Backward', 'Forward', or 'Centered' for their choice of finite" \
        "difference approximation. Please try again.")
        return -1
    return dfxi
'''
# define grid
x = np.arange(0, 2*np.pi,0.1)
# compute function
y = np.cos(x)
forward_diff = my_finite_diff(y,x,"Forward")
# compute corresponding grid
x_diff = x[:-1:]
# compute exact solution
exact_solution = -np.sin(x_diff)
# Plot solution
plt.figure(figsize = (12, 8))
plt.plot(x_diff, forward_diff, '--', \
         label = 'Finite difference approximation')
plt.plot(x_diff, exact_solution, \
         label = 'Exact solution')
plt.legend()
plt.show()
'''
def fourth_order_diff(f,xi,h):
    '''
    Takes in f - a function of one argument
    xi - vector of the x-coordinates we wish to estimate the derivative at
    h - distance defining the finite difference approximations
    return dfxi, a vector that is the size of xi-2 that estimates the derivative at
    all points xi
    '''

    #check that h is positive
    if(h<=0):
        print("ERROR: h must be positive (>=0).")
        return -1

    xi = np.atleast_1d(xi)
    dfxi = np.zeros(len(xi))

    for i in range(0,len(dfxi)):
        dfxi[i] = (-f(xi[i]+2.0*h)+8.0*f(xi[i]+h)-8.0*f(xi[i]-h)+f(xi[i]-2.0*h))/12.0*h

    return dfxi

#convergence analysis
f = lambda x: pow(m.e,-x)
dfdx = -1.0*pow(m.e,-0.6)

#heights
h1 = 1.0
h2 = 0.5
h3 = 1/3
h4 = 0.25
h5 = 0.2
h6 = 1/6
h7 = 1/7
h8 = 1/8
h9 = 1/9
hlocs = [h1,h2,h3,h4,h5,h6,h7,h8,h9]
h = np.array(hlocs)

hlabels = ["1",".5",".3333",".25",".20",".16",".143",".125",".111"]

#errors
x = 0.6
errors = np.zeros(len(h))
for i in range (0,len(errors)):
    errors[i] = abs(fourth_order_diff(f,x,h[i]) - dfdx)


#elocs = [e1,e2,e3,e4,e5,e6,e7,e8,e9]
#elabels = [str(e1),str(e2),str(e3),str(e4),str(e5),str(e6),str(e7),str(e8),str(e9)]

#plot step size vs. error

for i in range(0,len(errors)):
    plt.scatter(h[i],errors[i],marker = 'x',s=100,color = 'black')

plt.xticks(hlocs,hlabels)
#plt.yticks(elocs,elabels)
plt.xlabel("Heights (h)")
plt.ylabel("Error")

plt.show()

#rates of convergences
rate = np.zeros(8)
for i in range(1,len(rate)+1):
    rate[i-1] = m.log(errors[i]-errors[i-1])/m.log(h[i]/h[i-1])
print("Rate: "+str(rate))




def my_composite_trap(x,fx):
    '''
    x - the x-coordinates of fx
    Takes in fx - input data we want to integrate
    Be sure to check the size of fx and x are the same
    return I, a scalar that is the estimated integral via trap rule
    '''
    if(len(x)!=len(fx)):
        print("ERROR: x and fx must be the same size.")
        return -1
    
    I=0.0

    for i in range (0,len(x)-1):
        I+= ((x[i+1]-x[i])/2)*(fx[i]+fx[i+1])

    return I
xi = np.array([0,1,2])
fxi = np.array([5,8,11])

i1 = my_composite_trap(xi,fxi)
print(i1)


def solve_freefall_RK4(x0,v0,nt,dt,g,cd,m):
    '''
    x0 - initial position (scalar)
    v0 - initial velocity (scalar)
    nt - number of time steps to take (integer)
    dt - time step size (scalar)
    g - gravitational constant (scalar)
    cd - drag coefficient (scalar)
    m - mass (scalar)
    Use RK4 to estimate both x,v at time 0 up to nt*dt
    return 2 vectors x,v (first is position second is velocity) each vector should
    be length nt+1
    '''
     # Initialize arrays to store position and velocity at each time step
    x = np.zeros(nt + 1)
    v = np.zeros(nt + 1)
    
    # Set initial conditions
    x[0] = x0
    v[0] = v0
    
    # RK4 integration
    for i in range(nt):
        # First step (k1)
        dx1 = v[i]
        dv1 = g - (cd/m) * v[i]**2
        k1_x = dt * dx1
        k1_v = dt * dv1
        
        # Second step (k2)
        dx2 = v[i] + k1_v/2
        dv2 = g - (cd/m) * (v[i] + k1_v/2)**2
        k2_x = dt * dx2
        k2_v = dt * dv2
        
        # Third step (k3)
        dx3 = v[i] + k2_v/2
        dv3 = g - (cd/m) * (v[i] + k2_v/2)**2
        k3_x = dt * dx3
        k3_v = dt * dv3
        
        # Fourth step (k4)
        dx4 = v[i] + k3_v
        dv4 = g - (cd/m) * (v[i] + k3_v)**2
        k4_x = dt * dx4
        k4_v = dt * dv4
        
        # Update position and velocity using weighted average
        x[i+1] = x[i] + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        v[i+1] = v[i] + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    
    return x, v


def solve_BVP_FD(T0,T1,k,dx):
    '''
    T0 - temperature on left boundary (scalar)
    T1 - temperature on right boundary (scalar)
    k - conductivity (scalar)
    dx - distance between each point ()
    Use Centered finite difference to estimate T
    return one vector T ( the vector should be length int(1/dx) + 1 )
    '''

    # Create the grid
    x = np.arange(0, 1 + dx, dx)
    n = len(x)

    if n < 3:
        raise ValueError("dx too large — must allow for at least one interior point")

    # Initialize temperature array
    T = np.zeros(n)
    T[0] = T0
    T[-1] = T1

    # Interior points only
    N = n - 2
    A = np.zeros((N, N))
    b = np.zeros(N)

    for i in range(N):
        # Skip x=0
        xi = x[i + 1]  
        b[i] = - (dx ** 2) * xi / k
        A[i, i] = -2
        if i > 0:
            A[i, i - 1] = 1
        if i < N - 1:
            A[i, i + 1] = 1

    # Apply boundary conditions
    b[0] -= T0
    b[-1] -= T1

    # Solve the system
    T[1:-1] = np.linalg.solve(A, b)
    return T

# Test parameters
# Temperature at left boundary
T0 = 0       
# Temperature at right boundary
T1 = 1       
# Conductivity coefficient
k = 0.1       

dx=0.1
    
# Compute numerical solution
T_numerical = solve_BVP_FD(T0, T1, k, dx=0.1)
print(T_numerical)

# Compute analytical solution for comparison
x = np.arange(0, 1 + dx, dx)
T_analytical = (-x**3)/(6*k) + (1 + 1/(6*k))*x
maxErr = np.max(np.abs(T_numerical-T_analytical))
print(maxErr)


