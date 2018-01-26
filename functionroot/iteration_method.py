import numpy as np
from numpy import sign
from sympy import *


def bisection_method(func, a, b, tol=1e-8):
	f_a = func.evalf(subs={x:a})
	f_b = func.evalf(subs={x:b})
	if sign(f_a)*sign(f_b) >= 0:
		print("f(a)*f(b)<0 not satisfied!")
		return None,None
	iter_num = 0
	while (b-a)/2>tol:
		iter_num = iter_num + 1
		c = (a+b)/2
		f_c = func.evalf(subs={x:c})
		if abs(f_c) < tol:
			break
		if sign(f_c)*sign(f_a)<0:
			b, f_b = c, f_c
		else:
			a, f_a = c, f_c
	xc = (a+b)/2
	print("bisection method:")
	print("In the error range of %.8f , the approximate solution is %.10f . \
		And the number of iterayions is %d" %(tol,xc,iter_num))
	return xc,iter_num

def newton_method(func,x0=0,tol=1e-8):
	d_func = diff(func,x)
	func_x0 = func.evalf(subs={x:x0})
	d_func_x0 = d_func.evalf(subs={x:x0})
	while abs(d_func_x0) < tol:
		x0 = x0 + 0.1
		func_x0 = func.evalf(subs={x:x0})
		d_func_x0 = d_func.evalf(subs={x:x0})
	iter_num = 1
	while iter_num:
		func_x0 = func.evalf(subs={x:x0})
		d_func_x0 = d_func.evalf(subs={x:x0})
		x1 = x0 - func_x0/d_func_x0
		func_x1 = func.evalf(subs={x:x1})
		if abs(func_x1) < tol:
			break
		iter_num = iter_num + 1
		x0 = x1
	print("newton iteration method:")
	print("In the error range of %.8f , the approximate solution is %.10f . \
		And the number of iterayions is %d" %(tol,x1,iter_num))
	return x1,iter_num

def simple_newton(func,x0=0,tol=1e-8):
	d_func = diff(func,x)
	func_x0 = func.evalf(subs={x:x0})
	d_func_x0 = d_func.evalf(subs={x:x0})
	while abs(d_func_x0) < tol:
		x0 = x0 + 0.1
		func_x0 = func.evalf(subs={x:x0})
		d_func_x0 = d_func.evalf(subs={x:x0})
	iter_num = 1
	while iter_num:
		func_x0 = func.evalf(subs={x:x0})
		x1 = x0 - func_x0/d_func_x0
		func_x1 = func.evalf(subs={x:x1})
		if abs(func_x1) < tol:
			break
		iter_num = iter_num + 1
		x0 = x1
	print("simple newton iteration method:")
	print("In the error range of %.8f , the approximate solution is %.10f . \
		And the number of iterayions is %d" %(tol,x1,iter_num))
	return x1,iter_num

def secant_method(func,x0,x1,tol=1e-8):
	iter_num = 1
	while iter_num:
		func_x0 = func.evalf(subs={x:x0})
		func_x1 = func.evalf(subs={x:x1})
		x2 = x1 - func_x1*(x1-x0)/(func_x1-func_x0)
		func_x2 = func.evalf(subs={x:x2})
		if abs(func_x2)<tol:
			break
		x0,x1 = x1,x2
		iter_num = iter_num + 1
	print("secant method:")
	print("In the error range of %.8f , the approximate solution is %.10f . \
		And the number of iterayions is %d" %(tol,x1,iter_num))
	return x1,iter_num

def steffensen_method(func,x0=0,tol=1e-8):
	func_x0 = func.evalf(subs={x:x0})
	iter_num = 1
	while iter_num:
		func_x0 = func.evalf(subs={x:x0})
		func_x00 = func.evalf(subs={x:x0+func_x0})
		x1 = x0 - func_x0**2/(func_x00-func_x0)
		func_x1 = func_x0.evalf(subs={x:x1})
		if abs(func_x1) < tol:
			break
		x0 = x1
		iter_num = iter_num + 1
	print("steffensen iteration method:")
	print("In the error range of %.8f , the approximate solution is %.10f . \
		And the number of iterayions is %d" %(tol,x1,iter_num))
	# print(tol,x1,iter_num)
	return x1,iter_num

if __name__ == "__main__":
	x = symbols("x", real=True)
	# 下面每行代表7数值实验中的每一个函数，这里以第5个为例
	# f = x**3 + x - 1
	# f = sin(x**2-1) - cos(x**(1/2)+1) + 1/3
	# f = exp(x**(1/2)-x)-x**2 - x
	# f = log(x**2+x+1/2)
	f = x**4 - 5.5*x**3 + 9*x**2 - 2*x - 4
	# f = x**4 - 1.0*x**3 + 0.25*x - 0.0625
	# f = 1/(x**2)+x-3
	# f = asin(x) + 0.5
	# f = sin(x) + tan(x) + 0.5
	# f = x + exp(x) - cos(x)
	# f = exp(x) - log(1+x**2)

	xc,iter_num = bisection_method(f,0, 3)
	xc,iter_num = newton_method(f,3)
	xc,iter_num = simple_newton(f,2.01)
	xc,iter_num = secant_method(f,3,2.8)
	xc,iter_num = steffensen_method(f,3)

	xc,iter_num = bisection_method(f,-0.8,0)
	xc,iter_num = newton_method(f,-1)
	xc,iter_num = simple_newton(f,-1)
	xc,iter_num = secant_method(f,-1,-0.8)
	xc,iter_num = steffensen_method(f,-0.6)
