import numpy as np

def lu_with_pivoting(A):
    '''
    Takes a square numpy matrix
    and computes LU decomposition with
    partial pivoting so that
    PA=LU
    returns P,L,U
    '''

    m,n = A.shape
    if(m!=n):
        return
    P = np.zeros((n,n))
    U = np.zeros((n,n))
    L = np.eye(n)
    U[:,:] = A[:,:]
    #stores what will fill out the permutation matrix
    p = np.arange(n)

    #go over each row except the last
    for i in range(n-1):
        #find max absolute value in column of our pivot

        max_index = i+np.argmax(U[i:,i]) #starting at the ith row
        #use this to swap pivot to get the max value on the diagonal
        U[[i,max_index],:] = U[[max_index,i],:]
        #do similar to L
        L[[i,max_index],:i] = L[[max_index,i],:i]
        #keep track of what we swapped in p
        p[[i, max_index]] = p[[max_index,i]]
        #do row operations
        for j in range(i+i,n):
            L[j,i] = U[j,i]/U[i,i]
            U[j,i:] = U[j,i:] - L[j,i]*U[i,i:]
    P[:,:] = P[p,:]
    return P,L,U

#test
if __name__ == "__main__":
    A = np.array([[1,2,1],
        [2,-1,2],
          [3,-1,2]]

    )
    P, L, U = lu_with_pivoting(A)


