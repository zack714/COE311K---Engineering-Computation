#NAME: Christian Aruba
#DATE: 3/15/25
import numpy as np
import matplotlib.pyplot as plt

error="error"
"""
#Problem 1
x = np.array([0.1,0.2,0.4,0.6,0.9,1.3,1.5,1.7,1.8])
y = np.array([0.75,1.25,1.45,1.25,0.85,0.55,0.35,0.28,0.18])

#sum of x and the sum of all x values squared
xsum = np.sum(x)
xsum_squared = np.sum(np.square(x))

#sum of the natural logs of all y values and the sum of the natural logs of all y values times their corresponding x values
ylogsum = np.sum(np.log(y))
ylogtimesx_sum = np.sum(np.log(y)*x)

#design matrix
Z = np.array([
    [9, xsum],
    [xsum, xsum_squared]
])

#dependent variable
Y = np.array([
    [ylogsum],
    [ylogtimesx_sum]
])

init_parameters = np.dot(np.linalg.inv(Z),Y)
final_parameters = np.array([np.exp(init_parameters[0]),init_parameters[1]])
print("Z: "+str(Z)+"\n Y: "+str(Y))
print("\nParameters: "+str(final_parameters))

plt.scatter(x,y)
plt.plot(x,final_parameters[0]*x*np.exp(final_parameters[1]*x),'-r')
plt.show()
"""

def least_squares_poly(x,y,k):
    #create a design matrix of shape (k+1,k+1)
    Z = np.empty((k+1,k+1))
    Y = np.empty((k+1,1))
    n = len(x)

    #calculating the design matrix
    #go through n (# of x values) rows 
    
    for r in range(0,k+1):
        #from the 0th column to the k+1th column
        for c in range(0,k+1):
            #in each column j of a row, take all x values, raise them to the (r+c)th power, and add them
            Z[r,c] = np.sum(x**(r+c))

    #calculate Y
    for r in range(0,k+1):
        Y[r,0] = np.sum(y*x**r)

    #take the inverse of Z
    Z_inv = np.linalg.inv(Z)

    #a is the dot product of Y and Z (which is now Z^T*Z)'s inverse
    a = np.dot(Z_inv,Y)
    return a



'''

# Generate 10 random x values between -5 and 5
np.random.seed(42)  # For reproducibility
x = np.linspace(-5, 5, 10)

# Compute y values based on the given quadratic equation with some noise
y = 5 + x + 2*x**2 + np.random.normal(loc=0, scale=2, size=len(x))  # Adding noise with std deviation 2

a = least_squares_poly(x,y,2)
print("A: "+str(a))

'''


def least_squares_fourier(x,y,k,omega_o):
    '''
Takes in x - input data
y - data you want to fit curve to, should be same length as x
k - the order of the fourier series you want to fit data to
omega_o - the fundamental frequency of the Fourier series that was shown in
the definition of the Fourier series
returns "a" , the vector of of length 2k + 1 coefficients that least squares
finds

'''
    

    #declare vector that holds coefficients
    a = np.zeros(2*k+1)

    #m is num of data points
    m = len(x)

    print("Number of data points: "+str(m))

    #build Z.T*Z
    zTransZ = np.diag([m] + [m/2.0] * (2*k))

    #build the b vector
    b = np.zeros(2*k+1)

    #the first element in b will always be the outputs summed up
    b[0] = np.sum(y)

    #every odd element of b will be a sum of the ith y times the cosine of omega_o*the ith t times the kth frequency
    #outer loop: sums over each row in b
    #to keep track of the frequencies
    
    #every odd element of b will be a sum of the ith y times the cosine of omega_o*the ith t times the kth frequency
    #outer loop: sums over each row in b
    #to keep track of the frequencies
    # Cosine and Sine terms
    for i in range(1, k+1):  
        b[2*i-1] = np.sum(y * np.cos(omega_o * x * i))  # Cosine terms
        b[2*i] = np.sum(y * np.sin(omega_o * x * i))    # Sine terms

        print(zTransZ)
        #now take the inverse of zTransZ and dot it with b
        a = np.dot(np.linalg.inv(zTransZ),b)

        print("b: "+str(b))
        print(zTransZ)
        #now take the inverse of zTransZ and dot it with b

        print("Sum of y (b[0]):", b[0])
        print("Mean of y:", np.mean(y))
    a = np.linalg.solve(zTransZ,b)
    return a

t = np.linspace(0,2*np.pi,100)
y = -7+2*np.cos(t)-3*np.sin(t)+1*np.cos(2*t)+0.5*np.sin(2*t)
#print("y: "+str(y))
res = least_squares_fourier(t,y,2,1)
print("a equals: " + str(res))


def my_dft(x):
    '''
    Takes in x - input data
    returns F, the vector containing the fourier coefficients
    '''
    # Number of data points
    n = len(x)  

    # Initialize output array
    F = np.zeros(n, dtype=complex)  
        
    # Fundamental frequency
    omega_0 = 2 * np.pi / n  
        
    # Iterate over output frequencies
    for k in range(n):  

        # Sum over input signal
        for j in range(n):  
            F[k] += x[j] * np.exp(-1j * k * omega_0 * j)
        

    return F

#problem 5
fs = 128
n = 64

#time step
ts = 1.0/fs
#t sub n
t_n = (n-1)*ts

#time period
t = np.linspace(0, t_n, n)

#spacing between frequencies
df = fs/n

#minimum and maximum frequencies
fmin = 0
fmax = fs/2

print("The time step is: "+ str(ts)+ ", the time period is "+str(t_n)+", the spacing between frequencies is\
 "+str(df)+", the minimum and maximum frequencies are "+str(fmin)+", and "+str(fmax)+".")

ft = 1.5 +1.8*np.cos(2*np.pi*12*t) + 0.8*np.sin(2*np.pi*20*t) - 1.25*np.cos(2*np.pi*28*t)
n_ft = len(ft)

dft_res = my_dft(ft)
#print(dft_res)

plt.plot(t,ft)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()

amplitude = np.abs(dft_res)/len(dft_res)

# Take only the first half (positive frequencies)
frequencies = np.linspace(0, fs/2, n//2)

# Compute amplitude spectrum (first half of DFT)
# Multiply by 2 to account for symmetry
amplitudes = 2 * np.abs(dft_res[:n//2]) / n  

plt.figure(figsize=(8, 4))
plt.plot(frequencies, amplitudes, 'o-', markersize=4, label="Amplitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Fourier Transform - Amplitude Spectrum")
plt.grid(True)
plt.legend()
plt.show()

def my_poly_interp(x,fx,xi,ptype):
    '''
    
Takes in x - input data
fx - data you want to interpolate
xi - the coordinates you want to interpolate at
ptype - a string defining which interpolation to be performed that is either
"Lagrange" or "Newton"
returns fxi, the interpolated values at each point xi
'''

    n = len(x)
    # Initialize result array with zeros
    fxi = np.zeros_like(xi)
    
    if ptype == "Lagrange":
        # Compute Lagrange basis polynomials and interpolate
        for i in range(n):
            # Initialize Lagrange basis polynomial
            L = np.ones_like(xi)
            for j in range(n):
                if i != j:
                    # Compute product term
                    L *= (xi - x[j]) / (x[i] - x[j])
                    # Sum weighted basis polynomials
            fxi += fx[i] * L
    
    elif ptype == "Newton":
        # Compute divided differences table
        # Initialize divided difference table
        divided_diff = np.zeros((n, n))
        # First column is function values
        divided_diff[:, 0] = fx
        for j in range(1, n):
            for i in range(n - j):
                divided_diff[i, j] = (divided_diff[i+1, j-1] - divided_diff[i, j-1]) / (x[i+j] - x[i])
        # Compute Newton interpolation polynomial
        for i in range(n):
            # Start with leading coefficient
            term = divided_diff[0, i]
            for j in range(i):
                 # Multiply by (xi - x_j) terms
                term *= (xi - x[j])
            # Sum terms to get final interpolated values
            fxi += term
    
    else:
        raise ValueError("Invalid interpolation type. Choose 'Lagrange' or 'Newton'")
    
    return fxi
