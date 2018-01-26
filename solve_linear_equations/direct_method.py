# LU_decomposition_f_improve.py 一样

import numpy as np
import sys,time

def break_matrix(m):
	m_dim = m.shape
	a_dim = m_dim[0]
	A = m[:,:a_dim]
	b = m[:,a_dim]
	print("A=",A)
	print("b=",b)
	return A,b

def gauss_solve(A,b):
	start_t = time.time()
	A_c,b_c = A.copy(),b.copy()
	A_c_dim = A_c.shape[0]
	mult = b_c.copy()
	for j in range(A_c_dim-1):
		if abs(A_c[j,j]) < 1e-8:
			print("zero pivot encounted")
			sys.exit()

		mult[:j+1] = 0
		mult[j+1:,0] = A_c[j+1:,j] / A_c[j,j]

		A_c[j+1:,:] = A_c[j+1:,:] - mult[j+1:]*A_c[j,:]
		b_c[j+1:,0] = b_c[j+1:,0] - mult[j+1:]*b_c[j,0]

	x = b_c.copy()
	for i in range(A_c_dim-1,-1,-1):
		b_c[i] = b_c[i] - A_c[i,i+1:]*x[i+1:,0]
		x[i] = b_c[i]/A_c[i,i]

	# print("x=",x)
	end_t = time.time()
	using_t = "using time = "+str(round(end_t-start_t,8))+" s"
	return x,using_t

def gauss_select(A,b):
	start_t = time.time()
	A_c,b_c = A.copy(),b.copy()
	A_c_dim = A_c.shape[0]
	for j in range(A_c_dim-1):
		if abs(A_c[j,j]) < 1e-8:
			print("zero pivot encounted")
			sys.exit()
		maxindex = j + abs(A_c[j:,j]).argmax()	# 选主元
		A_c[[j,maxindex],:] = A_c[[maxindex,j],:]	# 换行
		b_c[j,0],b_c[maxindex,0] = b_c[maxindex,0],b_c[j,0]

		# print(A_c)

		b_c[j,0] = b_c[j,0] / A_c[j,j]
		A_c[j,:] = A_c[j,:] / A_c[j,j]	# 归一化

		b_c[j+1:,0] = b_c[j+1:,0] - A_c[j+1:,j]*b_c[j,0]
		A_c[j+1:,:] = A_c[j+1:,:] - A_c[j+1:,j]*A_c[j,:]

	# print(A_c)
	# print(b_c)

	x = b_c.copy()
	for i in range(A_c_dim-1,-1,-1):	# A_dim 为3，故这应为减一
		b_c[i] = b_c[i] - A_c[i,i+1:]*x[i+1:]
		x[i] = b_c[i]/A_c[i,i]

	# print("x=",x)
	end_t = time.time()
	using_t = "using time = "+str(round(end_t-start_t,8))+" s"
	return x,using_t

def LU_decompo(A):
	A_dim = A.shape[0]
	L = A.copy()
	U = A.copy()
	for j in range(A_dim):
		if abs(A[j,j]) < 1e-8:
			print("zero pivot encounted")
			sys.exit()
		L[:j,j] = 0
		L[j:,j] = U[j:,j] / U[j,j]
		U[j+1:,:j+1] = 0
		U[j+1:,j+1:] = U[j+1:,j+1:] - L[j+1:,j]*U[j,j+1:]

	# print("U=",U)
	# print("L=",L)
	# print(L*U)
	return L, U

def back_up(b,L,U):
	b_dim = b.shape[0]
	c = b.copy()
	for i in range(b_dim):
		c[i] = c[i] - L[i,:i]*c[:i]
	# print("c=",c)

	x = b.copy()
	for i in range(b_dim-1,-1,-1):
		c[i] = c[i] - U[i,i+1:]*x[i+1:]
		x[i] = c[i]/U[i,i]
	# print("x=",x)
	return x

def LU_depo_solve(A,b):
	start_t = time.time()
	A_c,b_c = A.copy(),b.copy()
	L,U = LU_decompo(A_c)
	x = back_up(b_c,L,U)
	end_t = time.time()
	using_t = "using time = "+str(round(end_t-start_t,8))+" s"
	return x,using_t


if __name__ == '__main__':
	# m = np.mat([[1. ,2 ,-1,3.],[2,1,-2,3],[-3,1,1,-6]])
	m = np.mat([[6. ,-2 ,2 ,4 ,12],[12,-8,6,10,34],[3,-13,9,3,27],[-6,4,1,-18,-38]])
	A,b = break_matrix(m)
	# print("A=",A)

	x,using_t = gauss_solve(A,b)
	print("By using gauss elimination, the solution is x =",x,using_t,sep="\n")

	x,using_t = gauss_select(A,b)
	print("By using gauss selection main element, the solution is x =",x,using_t,sep="\n")

	x,using_t = LU_depo_solve(A,b)
	print("By using LU decomposition, the solution is x =",x,using_t,sep="\n")
