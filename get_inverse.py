import numpy as np
from functools import reduce
import sys


def LU_deco_inverse(m):
	# well done
	dim = m.shape[0]
	E = np.mat(np.eye(dim))
	L = np.mat(np.eye(dim))
	U = m.copy()
	for i in range(dim):
		if abs(m[i,i]) < 1e-8:
			print("zero pivot encoUnted")
			sys.exit()

		L[i+1:,i] = U[i+1:,i] / U[i,i]
		E[i+1:,:] = E[i+1:,:] - L[i+1:,i]*E[i,:]
		U[i+1:,:] = U[i+1:,:] - L[i+1:,i]*U[i,:]

	print("LU decomposition matrix m to L,U:")
	print("L=",L,"\n","U=",U)

	for i in range(dim-1,-1,-1):
		(E[i,:], U[i,:]) = (E[i,:], U[i,:]) / U[i,i]
		E[:i,:] = E[:i,:] - U[:i,i]*E[i,:]
		U[:i,:] = U[:i,:] - U[:i,i]*U[i,:] # r_j = m_ji - r_j*r_i

	print("inverse of m via primary transformation:")
	print("E=",E)
	print("using the method 'I' of numpy to calculate the inverse of matrix m:")
	print("m_inv=",m.I)


if __name__ == '__main__':
	A = np.mat([[1.,1,1],[1,2,3],[1,5,1]])
	A_dim = A.shape[0]
	LU_deco_inverse(A)
  
### output: ###
# LU decomposition matrix m to L,U:
# L= [[ 1.  0.  0.]
#  [ 1.  1.  0.]
#  [ 1.  4.  1.]] 
#  U= [[ 1.  1.  1.]
#  [ 0.  1.  2.]
#  [ 0.  0. -8.]]
# inverse of m via primary transformation:
# E= [[ 1.625 -0.5   -0.125]
#  [-0.25   0.     0.25 ]
#  [-0.375  0.5   -0.125]]
# using the method 'I' of numpy to calculate the inverse of matrix m:
# m_inv= [[ 1.625 -0.5   -0.125]
#  [-0.25   0.     0.25 ]
#  [-0.375  0.5   -0.125]]
