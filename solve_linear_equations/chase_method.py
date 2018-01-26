import numpy as np
import sys,time

def break_matrix(m):
	m_dim = m.shape
	a_dim = m_dim[0]
	A = m[:,:a_dim]
	b = m[:,a_dim]
	return A,b

def chasematrix(A,b):
	start_t = time.time()
	A_dim = A.shape[0]
	for j in range(A_dim-1):
		if abs(A[j,j]) < 1e-8:
			print("zero pivot encounted")
			sys.exit()
	y = b.copy()
	P = np.mat(np.tril(A,-1))
	Q = np.mat(np.eye(A_dim))
	P[0,0] = A[0,0]
	Q[0,1] = A[0,1]/P[0,0]
	Q[0,0] = 1
	for i in range(1,A_dim-1):
		P[i,i] = A[i,i] - A[i,i-1]*Q[i-1,i]
		Q[i,i+1] = A[i,i+1]/P[i,i]
	P[A_dim-1,A_dim-1] = A[A_dim-1,A_dim-1] - A[A_dim-1,A_dim-2]*Q[A_dim-2,A_dim-1]

	print("P=",P)
	print("Q=",Q)
	# print(P*Q==A)

	y = b.copy()
	y[0,0] = b[0,0] / P[0,0]
	for i in range(1,A_dim):
		y[i,0] = (b[i,0] - P[i,i-1]*y[i-1,0])/P[i,i]

	x = y.copy()
	# x[A_dim-1,0] = y[A_dim,0]
	for i in range(A_dim-2,-1,-1):
		x[i,0] = y[i,0]-Q[i,i+1]*x[i+1,0]

	# print(x)
	# print(A*x==b)
	end_t = time.time()
	using_t = "using time = "+str(round(end_t-start_t,8))+" s"
	return x,using_t




if __name__ == '__main__':
	m = np.mat("2.,-1,0,0,6;-1,3,-2,0,1;0,-1,2,-1,0;0,0,-3,5,1")
	A,b = break_matrix(m)
	print(A)
	print(b)
	x,using_t = chasematrix(A,b)
	print("By chasing method, the solution is x =",x,using_t,sep="\n")

