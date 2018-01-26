import numpy as np
from numpy.linalg import norm

def is_strict_diagonal_doinant(matrix):
	diagonal = matrix.diagonal()
	row_sum = matrix.sum(axis=1).T - diagonal
	if (abs(diagonal) > abs(row_sum)).all():
		return True
	else:
		return False

def break_matrix(A):
	if is_strict_diagonal_doinant(A):
		L = np.mat(np.zeros(A.shape))
		U = L.copy()
		D = L.copy()

		for i in range(A.shape[0]):
			L[i+1:,i] = A[i+1:,i]
			U[i,i+1:] = A[i,i+1:]
			D[i,i] = A[i,i]

		return L,U,D
	else:
		print("A is not diagonal doinant!")
		return None

def richandson_iteration(A,b,tol=1e-6):
	iter_num = 0
	E = np.mat(np.eye(A.shape[0]))
	x = np.mat(np.zeros((A.shape[0],1)))
	while norm(A*x-b,ord=2)>tol:
		iter_num = iter_num + 1
		x = b+(E-A)*x
		# print(x)
	return x, iter_num

def jacobi_iteration(A,b,tol=1e-6):
	iter_num = 0
	L,U,D = break_matrix(A)
	D_inv = D.I
	x = np.mat(np.zeros((A.shape[0],1)))
	while norm(A*x-b,ord=2)>tol:
		iter_num = iter_num + 1
		x = D_inv*(b-(L+U)*x)
		# print(x)
	return x, iter_num

def gauss_seidel_iteration(A, b, tol=1e-6):
	iter_num = 0
	L,U,D = break_matrix(A)
	D_inv = D.I
	x = np.mat(np.zeros((A.shape[0],1)))
	while norm(A*x-b,ord=2)>tol:
		iter_num = iter_num + 1
		x = -(D+L).I*U*x + (L+D).I*b
		# print(x)
	return x, iter_num



if __name__ == "__main__":
	# A = np.mat("3.,1;1,2")
	# b = np.mat("5.;5")
	# # A = np.mat("1,2;3,1")
	# x,iter_num = jacobi_iteration(A,b)
	# print("jacobi","x=",x," iteration time = ",iter_num,sep="\n")

	# x,iter_num = gauss_seidel_iteration(A,b)
	# print("gauss_seidel","x=",x," iteration time = ",iter_num,sep="\n")

	# x,iter_num = richandson_iteration(A,b)
	# print("richandson","x=",x," iteration time = ",iter_num,sep="\n")

	A = np.mat([[1.,1/2,1/3],[1/3,1,1/2],[1/2,1/3,1]])
	b = np.mat([[11/18],[11/18],[11/18]])
	x,iter_num = jacobi_iteration(A,b)
	print("jacobi","x=",x," iteration time = ",iter_num,sep="\n")

	x,iter_num = gauss_seidel_iteration(A,b)
	print("gauss_seidel","x=",x," iteration time = ",iter_num,sep="\n")

	x,iter_num = richandson_iteration(A,b)
	print("richandson","x=",x," iteration time = ",iter_num,sep="\n")
