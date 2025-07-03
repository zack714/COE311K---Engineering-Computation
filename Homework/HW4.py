import numpy as np
error="error"
#NAME: Christian Aruba
#DATE: 4/3/2025
def my_linear_interpolation(x,fx,xi):
    '''
    Takes in x - input data
    fx - data you want to fit curve to, should be same length as x
    xi - the points you wish to interpolate at
    return fxi, a vector same size as xi with interpolated values using linear
    interpolation
    return error if any points in xi lie outside the range of x
    '''

    if(len(fx)!=len(x)):
        print("fx and x must share the same length.")
        return
    
    fxi = np.arange(len(xi),dtype="float")
    for i in range(len(xi)):
        xinterp = xi[i]

        #check for out of bounds values
        if xinterp < x[0]:
            print("interpolation point is out of bounds.")
            return
        elif xinterp > x[-1]:
            print("Interpolation point is out of bounds.")
            return
        else:
            for j in range(len(x)-1):
                if x[j]<=xinterp<=x[j+1]:
                    break
         #apply the interpolation formula to find fxi and store it at the ith index in fxi vector
        x0, x1 = x[j], x[j+1]
        y0, y1 = fx[j], fx[j+1]
        print(xinterp,x0,x1,y0,y1)
        fxi[i] = y0 + float((y1-y0)*(xinterp-x0))/float(x1-x0)

        
    
    return fxi

'''   
x = np.array([0,1,2])
fx = np.array([1,3,2])
xi = np.array([0.5,1.5])
fxi = my_linear_interpolation(x,fx,xi)
print("fxi: "+str(fxi))
''' 

def my_cubic_spline_interpolation(x,fx,xi):
    '''
    Takes in x - input data
    fx - data you want to fit curve to, should be same length as x
    xi - the points you wish to interpolate at
    return fxi, a vector same size as xi with interpolated values using cubic
    splines
    return error if any points in xi lie outside the bounds of x

    '''

    #nested for loop to check if any points in xi are outside x
    for interp in xi:
        if interp < x[0] or interp >x[-1]:
            print("Error. Interpolation point "+str(interp)+" is out of bounds. ")
            return -1
    
    
    n = len(x)

    #number of splines 
    m = n-1

    #length of the interval of a spline
    h = np.zeros(m)
    for i in range(m):
        h[i] = x[i+1]-x[i]

    coeffs = np.zeros(3*(n-1))
    
    #Unknowns
    fxi = np.zeros_like(xi)
    A = np.zeros((3*m,3*m))
    rhs = np.zeros(3*m)
    
    #row index
    r=0

    ###assembling the spline equations###

    #for every spline equation
    for i in range(m):
        #use slicing to work on all 3 cells at the same time
        A[r,3*i:3*i+3] = [h[i],h[i]**2,h[i]**3]
        rhs[r] = fx[i+1] - fx[i]  # f(x_{i+1}) - f(x_i)
        r+=1

    #enforce rule 3: the first derivative rule
    for i in range(m-1):
        A[r,3*i:3*i+3] = [1,2*h[i],3*h[i]**2]
        A[r,3*(i+1)] = -1
        rhs[r] = 0
        r+=1

    #enforce rule 4: the second derivative rule
    for i in range(m-1):
        A[r,3*i+1:3*i+3] = [2,6*h[i]]
        A[r,3*(i+1)+1] = -2
        rhs[r] = 0
        r+=1
    
    #Row 3*n-5 will enforce rule 5: natural boundary equation (1): C1's constant will always be 0 on this row
    A[r,1] = 2
    rhs[r] = 0
    r+=1

    #Row 3*n-4 will enfore rule 6: natural boundary equation (2): 2*Cn-2+6*Dn-2*(xn-1-xn-2) = 0
    A[r, 3*(m-1)+1:3*(m-1)+3] = [2,6*h[m-1]]       # 2*c_{n-2}
    r+=1

    coeffs = np.linalg.solve(A,rhs)

    #now find fxi's for each xi
    for k in range(len(xi)):
        interp = xi[k]

        i = 0
        
        #increment i until we find the endpoints of k'th xi
        while i<len(x)-2 and not (x[i]<=interp<=x[i+1]):
            i+=1

        #if the interpolant happens to be the last point...
        if interp == x[-1]:
            #shift back to the point before that. There's no spline segment that starts at the last point.
            i = len(x)-2
        
        #finding the k'th constants for the k'th xis
        dx = interp-x[i]
        b = coeffs[3*i]
        c = coeffs[3*i+1]
        d = coeffs[3*i+2]
        a = fx[i]

        fxi[k] = a+b*dx+c*dx**2+d*dx**3
    
    return fxi

#x = np.array([0,1,2])
#fx = np.array([1,3,2])
#xi = np.array([0.5,1.5])
#fxi = my_cubic_spline_interpolation(x,fx,xi)
#print(fxi)p
    

def my_bisection_method(f,a,b,tol,maxit):
    '''
    Takes in f - a possibly nonlinear function of a single variable
    a - the lower bound of the search interval
    b - the upper bound of the search interval
    tol - allowed relative tolerance
    maxit - allowed number of iteraions
    return root,fx,ea,nit
    root - the final estimate for location of the root
    fx - the estimated values for f at the root (will be near 0 if we are close
    to the root)
    ea - the final relative error
    nit - the number of iterations taken
    return error if the sign of f(a) is the same as the sign of f(b) since this
    means it is possible
    that a root doesnt lie in the interval
    '''

    # Check if the function has opposite signs at a and b
    if f(a) * f(b) > 0:
        print("Error: Function must have opposite signs at interval endpoints")
        return "error", None, None, None
    
    #iteration counter
    nit = 0

    #relative error
    ea = 100.0

    #initial guess
    root_old = a

    while nit<maxit and ea>tol:
        #compute midpoint
        root = (a+b)/2

        #Evalutate function at midpoint
        fx = f(root)

        #check if we found exact root
        if fx==0:
            ea = 0
            break
    
        #update interval
        if f(a)*fx <0:
            #true root is to the left of our old guess; make 
            #b our right bound
            b = root
        else:
            #root is in the right half
            a = root

    #Calculate relative error if possible
    if root !=0:
        ea = abs((root-root_old)/root)
    
    root_old = root

    nit+=1
    return root,fx,ea,nit

def modified_secant_method(f,x0,tol,maxit,delta):
    '''
    Takes in f - a possibly nonlinear function of a single variable
    x0 - the initial guess for the root
    tol - allowed relative tolerance
    maxit - allowed number of iteraions
    delta - the size of the finite difference approximation
    return root,fx,ea,nit
    root - the final estimate for location of the root
    fx - the estimated values for f at the root (will be near 0 if we are close
    to the root)
    ea - the final relative error
    nit - the number of iterations taken
    no error checking is necessary in this case'
    '''

    #x0 = x0.astype(float)
    #tol = tol.astype(float)
    #delta = delta.astype(float)

    #make initial guess the old guess
    x_old = x0
    #initialize nit as 0
    nit = 0
    #initialize error as a large number
    ea = 100.0
    #while it<maxit and error>tol...
    while nit<maxit and ea>tol:
        #initialize new guess using formula 
        x_new = x_old - (delta*f(x_old))/(f(x_old+delta)-f(x_old))
        #calculate relative error
        ea = abs((x_new-x_old)/x_new)
        #increment nit
        nit+=1
        #make x_new=x_old for the next loop
        x_old=x_new
    #calculate f(x_new)
    fx = f(x_old)
    #return root, fx, ea, nit
    root = x_old
    return root, fx, ea, nit

f = lambda x: (x**2)+6*x+5
res = modified_secant_method(f,-6.0,0.01,50,0.1)
print(res)