import numpy as np
error="error"
def naive_LU(A):
    #check if A is square
    n,m = A.shape
    if n!=m:
        print("A must be square for a LU decomposition.")
        return -1

    #initalize L and U
    L = np.eye(n)
    #so we don't change the user's matrix
    U = A.copy()
#go through each row and make sure you don't have a zero pibot
    for i in range(n):
        for j in range(i+1,n):
            if U[i,i] == 0:
                print("Zero pivot detected.")
                return -1
            #element in L is the scaling factor row opps are preformed with
            L[j,i] = U[i,j]/U[i,i]
            U[j,i:] -= L[j,i]*U[i,i:]
    return L,U

    pass
def solve_LU(L,U,b):
    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)

    #Forward substitution (Ly = b)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i],y[:i])

    #Backward substitution (Ux=y)
    for i in range(n-1,-1,-1):
        if U[i,i] == 0:
            print("Zero pivot encountered.")
            return -1
        x[i] = (y[i] - np.dot(U[i,i+1:],x[i+1:]))/U[i,i]
    return x


    pass
def inv_using_naive_LU(A):
    n = A.shape[0] #get the first value in the tuple returned by shape (rows)
    A_inv = np.zeros((n,n))

    #use naive LU decomposition function
    L, U = naive_LU(A)

    #solve for each column of the identity matrix
    for i in range(n):
        #initialize identity matrix
        e = np.zeros(n)
        e[i] = 1
        #go through all rows on the ith column of A_inv and fill it in
        A_inv[:,i] = solve_LU(L,U,e)

    return A_inv

    pass
def Richardson_it(A,b,omega,tol,max_it):
    n = len(b)
    x = np.zeros(n)
    #take the  L2 norm of Ax-b
    error = np.linalg.norm(np.dot(A,x) - b,2)


    for n_it in range(max_it):
        #compute new solution
        x_new = n+omega*(b-np.dot(A,x))
        #compute error
        error = np.linalg.norm(np.dot(A,x_new)-b,2)

        if error<tol:
            break

        x = x_new

    return x, n_it, error

    pass
def largest_eig(A,tol,max_it):
    n = A.shape[0]
    x = np.ones(n)
    #to make sure the vector has a magnitude of 1
    x /= np.linalg.norm(x)
    eig_old = 0
    error = 1

    for n_it in range(max_it):
        x_new = np.dot(A,x)
        #estimate eigenvalue
        eig_new = np.linalg.norm(x_new)
        #normalize x_new
        x_new = x_new/eig_new

        #compute relative error
        error = abs((eig_new-eig_old)/eig_new)

        #see if function has converged
        if error < tol:
            break

        x=x_new
        eig_old = eig_new

    return eig_new, x, n_it, error
    pass


##########################################################################
##########################################################################
#
# BONUS
#
##########################################################################
##########################################################################
def my_Cholesky(A):
    """
    In case inputs are such that the algorithm should not (and will not) work,
    use command 'return error'
    to return an error message that indicates the autograder that you have
    handled the error cases well
    In other cases return U
    """
pass
def my_Gauss_Siedel(A,b,tol,max_it):
    """
    In case inputs are such that the algorithm should not (and will not) work,
    use command 'return error'
    to return an error message that indicates the autograder that you have
    handled the error cases well
    In other cases return x (the solution)
    """
pass