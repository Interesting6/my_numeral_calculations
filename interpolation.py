import numpy as np
from numpy import sign
from sympy import *
from functools import reduce
import sys,time
import pandas as pd
import matplotlib.pyplot as plt



# 定义符号变量t
t = Symbol("t")



# 1. Lagrange插值
def Lagrange(x,y,xi=1):
    t = Symbol("t")
    m,n = len(x),len(y)
    x = np.array(x)
    y = np.array(y)
    xi = np.array([xi],dtype="float64")
    if m!=n:
        print("error input,vector x,y 's demension must be consistent")
        sys.exit()
    l_arr = [0]*n
    for i in range(n):
        temp = np.delete(x,i)
        l_arr[i] = (t-temp)/(x[i]-temp)
        l_arr[i] = reduce(lambda x,y:x*y,l_arr[i])
    l_arr = np.array(l_arr)
    func = np.dot(l_arr, y)
    yi = func.evalf(subs={t:xi})
    print("P(t)= ",func)
    print("P(xi)= ",yi)
    return func,yi


# 用抛物线插值计算根号115
x = [100,121,144]
y = [10,11,12]
x1 = 115.
func11,y1 = Lagrange(x,y,x1)

np.sqrt(115)

my_sqrt = lambdify(t,func11)

simplify(func11)

x = np.linspace(100,150)
my_y = my_sqrt(x)
sys_y = np.sqrt(x)
plt.plot(x,my_y,"b.",label="$my_sqrt(x)$")
plt.plot(x,sys_y,"r",label="$sqrt(x)$")
plt.legend()
plt.show()

# 用lagrange插值sin
x = np.arange(0,8)
y = np.sin(x)
func12,yi, = Lagrange(x,y,1.5)

x = np.linspace(0,8)
my_sin = lambdify(t,func12)
my_y = my_sin(x)
sys_y = np.sin(x)
plt.plot(x,my_y,"b.",label="$my sin(x)$")
plt.plot(x,sys_y,"r",label="$sin(x)$")
plt.legend()
plt.show()



# 2. Newton插值
def Newton(x,y,xi=1):
    t = Symbol("t")
    m,n = len(x),len(y)
    if m!=n:
        print("error input,vector x,y 's demension must be consistent")
        sys.exit()
    x,fx = np.array(x),np.array(y)
    cs = np.eye(n)
    cs[:,0] = fx
    fx_t = fx.copy()
    p_ = 1
    p_arr = np.array([p_])
    for k in range(n-1):
        cs[k+1:,k+1] = (fx_t[k+1:n]-fx_t[k:n-1])/(x[k+1:n]-x[:-1-k])
        fx_t = cs[:,k+1]
        if k>0:
            p_ = p_*(t-x[k-1])
            p_arr = np.append(p_arr,p_)
    # print(cs)

    func_cs = np.diag(cs)
    func_cs = np.delete(func_cs,-1)

    func = np.dot(p_arr,func_cs)
    print("P(t) =",func)
    yi = func.evalf(subs={t:xi})
    print("P(xi) =",yi)
    return func,yi

# 双曲正弦函数f(x)=sh(x)的插值
x = [0.4,0.55,0.65,0.80,0.90,1.05]
y = np.sinh(x)
print(y)

func2,y1 = Newton(x,y,0.596)

np.sinh(0.596)

my_sinh = lambdify(t,func2)

print(my_sinh(0.7))
print(np.sinh(0.7))

x = np.linspace(0,2)
my_y = my_sinh(x)
sys_y = np.sinh(x)

plt.plot(x,my_y,"b.",label="$my sinh(x)$")
plt.plot(x,sys_y,"r",label="$sinh(x)$")
plt.legend()
plt.show()

# 用newton插值exp(x)
x = np.arange(10)
y = np.exp(x)
print("x=",x,"y=",y,sep=" ")

func22,y1 = Newton(x,y,4.8)
np.exp(4.8)

my_exp = lambdify(t,func22)

x = np.linspace(0,10)
my_y = my_exp(x)
sys_y = np.exp(x)
plt.plot(x,my_y,"b.",label="$my exp(x)$")
plt.plot(x,sys_y,"r",label="$exp(x)$")
plt.legend()
plt.show()



# 3. 三次样条插值
# 三对角元系数矩阵的追赶法
def chasematrix(A,b):
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

    y = b.copy()
    y[0,0] = b[0,0] / P[0,0]
    for i in range(1,A_dim):
        y[i,0] = (b[i,0] - P[i,i-1]*y[i-1,0])/P[i,i]

    x = y.copy()
    for i in range(A_dim-2,-1,-1):
        x[i,0] = y[i,0]-Q[i,i+1]*x[i+1,0]
    return x

def third_spline(x,y,xi=1):
    t = Symbol("t")
    x,fx = np.array(x),np.array(y)
    h = np.diff(x)
    u = 2*(np.roll(h,1)+h)
    u = np.delete(u,0)
    b = 6*np.diff(fx)/h
    v = np.diff(b)
    A = np.diag(u)
    h = np.delete(h,0)
    for i in range(0,len(v)-1):
        A[i,i+1] = h[i]
        A[i+1,i] = h[i]
    A = np.mat(A)
    v = np.mat(np.reshape(v,(len(v),1)))
    z = chasematrix(A,v)
    z = np.array(z).flatten()
    z = np.concatenate(([0],z))
    h = np.diff(x)
    cs = [0]*4
    func = []
    for i in range(1,len(z)-1):
        cs[0] = z[i]/(6*h[i])*(x[i+1]-t)**3
        cs[1] = z[i+1]/(6*h[i])*(t-x[i])**3
        cs[2] = (y[i+1]/h[i] - z[i+1]*h[i]/6)*(t-x[i])
        cs[3] = (y[i]/h[i] - z[i]*h[i]/6)*(x[i+1]-t)
        func_t = reduce(lambda x,y:x+y,cs)
        func.append(func_t)
        # print(func_t)
        if xi>=x[i] and xi<x[i+1]:
            yi = func_t.evalf(subs={t:xi})
            print("P(t) =",func_t)
            print("P(xi) =",yi)
            xindex = i

    # print(func)
    return func,yi,xindex


# f(x)=x^(1/2)的插值
x = [1,4,9,16,25,36]
y = [1,2,3,4,5,6]
func3,yi,xindex = third_spline(x,y,17)

np.sqrt(17)

plt.figure(figsize=(16,8)) 
for i in range(1,4):
    my_sqrt3 = lambdify(t,func3[i-1])
    x_ = np.linspace(x[i],x[i+1])
    my_y = my_sqrt3(x_)
    sys_y = np.sqrt(x_)
    plt.plot(x_,my_y,"b.",label="$my sqrt(x)$"+str(i))
    plt.plot(x_,sys_y,"r",label="$sinh(x)$")
plt.legend()
plt.show()

# f(x)=cos(x)的插值
x = np.arange(0,12,2)
y = np.cos(x)
func32,yi,xindex = third_spline(x,y,5.4)

np.cos(5.4)

plt.figure(figsize=(16,8)) 
for i in range(1,4):
    my_cos32 = lambdify(t,func32[i-1])
    x_ = np.linspace(x[i],x[i+1])
    my_y = my_cos32(x_)
    sys_y = np.cos(x_)
    plt.plot(x_,my_y,"b.",label="$my cos(x)$"+str(i))
    plt.plot(x_,sys_y,"r",label="$cos(x)$")
plt.legend()
plt.show()









