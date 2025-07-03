import numpy as np

def naive_Gauss(A,b):
    '''

    takes in square numpy matrix A, and numpy vector b

    returns soluton x as a numpy vector
    '''

    nrow,ncol = A.shape
    nb = b.size
    x = np.zeros(ncol)
    if((nrow!=ncol) or (nb!=nrow)):
        print("Exiting, dimensions don't agree")
        return x
    Augmented_mat = np.zeros((nrow,ncol+1))
    Augmented_mat[:,:-1] = A[:,:]
    Augmented_mat[:,-1] = b[:]
    #loop over the pivot rows
    for i in range(nrow-1):
        #partial pivoting would go here
        ####
        ###
        #
        pivot_element = Augmented_mat[i,i]
        pivot_row = Augmented_mat[i,:]
        for j in range(i+1,nrow):
            scaling_factor = -Augmented_mat[j,i]/pivot_element
            Augmented_mat[j,:] = Augmented_mat[j,:]+scaling_factor*pivot_row
   #now for back substitution
    for i in range(nrow-1,-1,-1):
        #store entry in the last column
        temp = Augmented_mat[i,-1]
        #loop over upper triangular portion
        for j in range(ncol-1,i,-1):
            temp-=x[j]*Augmented_mat[i,j]
        x[i] = temp/Augmented_mat[i,i]
    return x







A = np.array([[1.,-2.,1.],
                 [2.,-1.,2.],
                 [3.,-1.,2.]])
b = np.array([8.,10.,11.,])
x = naive_Gauss(A,b)
print("x = ")
print(x)
#compare with numpy's solver
print("x from numpy = ")
#numpy's solver
print(np.linalg.solve(A,b))
#numpy also has several other linear algebra routines
#np.dot(), can be used for two vectors, mat-vec, mat-mat
#exmaple
print("Ax should = b")
print(np.dot(A,x))
I = np.array([[1,0,0],
             [0,1,0],
             [0,0,1]]

)
#print should be same as matmul for mat-mat multiplication
print(np.dot(A,I))
print(np.matmul(A,I))
#calculate norms easily
print(np.linalg.norm(b,ord=2))
print(np.linalg.norm(b,ord=np.inf))
#calculate determinants
print(np.linalg.det(A))