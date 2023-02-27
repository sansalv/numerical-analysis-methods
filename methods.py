import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

from sympy import *
import sympy as sp

"""
Some common methods for numerical analysis.
I wrote these during my university course Numerical analysis to help me better understand the methods and to help me do the "calculate by hand" iteration exercises.

TODO: Comment and document these methods properly
"""

def lu(A):
	A = A.astype('float64')
	n = A.shape[0]
	U = A.copy()
	L = np.eye(n, dtype=np.double)
    #Loop over rows
	for i in range(n):    
        #Eliminate entries below i with row operations 
        #on U and reverse the row operations to 
        #manipulate L
		factor = U[i+1:, i] / U[i, i]
		L[i+1:, i] = factor
		U[i+1:] -= factor[:, np.newaxis] * U[i]
	return L, U
    


def cholesky(A):
	"""Performs a Cholesky decomposition of A, which must 
	be a symmetric and positive definite matrix. The function
	returns the lower variant triangular matrix, L."""
	n = len(A)

	# Create zero matrix for L
	L = [[0.0] * n for i in range(n)]
	L = np.array(L)
	
	# Perform the Cholesky decomposition
	for i in range(n):
		for k in range(i+1):
			tmp_sum = sum(L[i,j] * L[k,j] for j in range(k))
			if (i == k):
				L[i,k] = sqrt(A[i,i] - tmp_sum)
			else:
				L[i,k] = (1.0 / L[k,k] * (A[i,k] - tmp_sum))
	return L
	
#def jacob(A, b, x):


def dlu(A):
	n = len(A)
	A = A.astype('float64')
	
	D = np.zeros((n, n))
	L = np.zeros((n, n))
	U = np.zeros((n, n))
	
	for i in range(n):
		D[i,i] = A[i,i]
		for j in range(n):
			if i > j:
				L[i,j] = A[i,j]
			elif i < j:
				U[i,j] = A[i,j]
	return D, L, U

def jacob(A, b, x, r):
	D, L, U = dlu(A)
	D_inv = np.linalg.inv(D)
	H = -D_inv @ (L+U)
	c = D_inv @ b
	k_fixed = -1
	for k in range(r):
		if np.Inf in x:
			k_fixed = i
			break
		else:
			x = H@x + c
	return x, k_fixed
	
	
def gauss_seidel(A, b, x, r):
	x = x.astype('double')
	A = A.astype('double')
	b = b.astype('double')
	D, L, U = dlu(A)
	D_inv = np.linalg.inv(D)
	k_fixed = -1
	n = len(A)
	print('x0=' + str(x))
	for k in range(r):
		print('*************************')
		print('iteration #' + str(k+1) + ':')
		print('-------------')
		if np.Inf in x:
			k_fixed = k
			break
		else:
			for i in range(n):
				print('i=' + str(i) + ':')
				s1 = sum(A[i,j] * x[j] for j in range(0,i,1))
				s2 = sum(A[i,j] * x[j] for j in range(i+1,n,1))
				x[i] = -D_inv[i,i]*(s1 + s2 - b[i])
				print(x)
	print('*************************')
	return x, k_fixed
	
def newton(f, x0, r):
	f_der = np.polyder(f, 1)
	x = x0
	for k in range(r):
		x = x - f(x)/f_der(x)
	return x
	
def secant(f, x0, x1, r):
	for k in range(r):
		if f(x1)-f(x0) != 0:
			x2 = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
			x0 = x1
			x1 = x2
		else:
			break
	return x2
	
def lambdaMu(a,b):
	golden = (-1+5**0.5)/2
	l = a + (1-golden)*(b-a)
	m = a + golden*(b-a)
	return l, m
	
def goldenRatioMethod(a,b,f,r):
	golden = (-1+5**0.5)/2
	if r != np.Inf:
		print('*******************************************')
		print('Input gap =  ' + str((round(a,2),round(b,2))))
		print('with values: ' + str((round(f(a),2),round(f(b),2))))
		for i in range(r):
			a1, b1 = lambdaMu(a,b)
			print('*******************************************')
			print(str(i+1) + '. iteration:')
			print('lambda =      ' + str(round(a1,6)))
			print('mu =          ' + str(round(b1,6)))
			print('f(lambda) =   ' + str(round(f(a1),6)))
			print('f(mu) =       ' + str(round(f(b1),6)))
			if f(a1) < f(b1):
				b = b1
			else:
				a = a1
			print('New gap =    ' + str((round(a,2),round(b,2))))
			print('with values: ' + str((round(f(a),2),round(f(b),2))))
		print('*******************************************')
		return (a,b), (f(a),f(b))
	else:
		while (round(f(a),10) != round(f(b),10)) or (round(a,10) != round(b,10)):
			a1, b1 = lambdaMu(a,b)
			if f(a1) < f(b1):
				b = b1
			else:
				a = a1
		return (a,b), (f(a),f(b))
		
def linearFit(xy):
	xi = [a[0] for a in xy]
	yi = [a[1] for a in xy]
	
	N = len(xy)
	s = np.sum(xy, axis=0)
	sx2 = np.sum([a[0]**2 for a in xy])
	sxy = np.sum([a[0]*a[1] for a in xy])
	
	a = np.array([[N, s[0]], [s[0], sx2]])
	b = np.array([s[1], sxy])
	
	a0, a1 = np.linalg.solve(a, b)
	
	y_lin = [a0+a1*t for t in xi]
	
	plt.scatter(xi,yi)
	plt.plot(xi, y_lin)
	plt.show()
	
	return a0, a1
	
def printACsWithVacuum():

	var('a b')
	
	T1 = 1/sp.sqrt(2)*Matrix([[0,1,0],[1,0,1],[0,1,0]])
	T2 = 1j/sp.sqrt(2)*Matrix([[0,-1,0],[1,0,-1],[0,1,0]])
	T3 = Matrix([[1,0,0],[0,0,0],[0,0,-1]])

	v11 = Matrix([a,0,0])
	v12 = Matrix([0,a,0])
	v13 = Matrix([0,0,a])
	v21 = Matrix([0,a,b])
	v22 = Matrix([a,0,b])
	v23 = Matrix([a,b,0])
	
	
	l = [v11, v12, v13, v21, v22, v23]
	
	for v in l:
		pprint(v)
		print('')
		pprint(T1.multiply(v))
		pprint(T2.multiply(v))
		pprint(T3.multiply(v))
		print('----------')
	
	t11 = 2*T1.multiply(T1)
	t22 = 2*T2.multiply(T2)
	t33 = 2*T3.multiply(T3)
	t12 = (T1.multiply(T2)+T2.multiply(T1))
	t13 = (T1.multiply(T3)+T3.multiply(T1))
	t23 = (T2.multiply(T3)+T3.multiply(T2))
	
	lt = [t11, t22, t33, t12, t13, t23]
	lt_name = ['t11 = {T1,T1}', 't22 = {T2,T2}', 't33 = {T3,T3}', 't12 = {T1,T2}', 't13 = {T1,T3}', 't23 = {T2,T3}']
	
	i = 0
	print('------------------')
	for t in lt:
		print(lt_name[i]+ ' = ')
		pprint(t)
		i += 1
		print('------------------')
	print('************************************************')
	M_list = []
	for v in l:
		M = Matrix([[0,0,0],[0,0,0],[0,0,0]])
		m_list = []
		for t in lt:
			m_ij = v.dot(t.multiply(v))
			m_list.append(m_ij)
		
		M[0,0] = m_list[0]
		M[1,1] = m_list[1]
		M[2,2] = m_list[2]
		
		M[0,1] = m_list[3]
		M[0,2] = m_list[4]
		M[1,2] = m_list[5]
		
		M[1,0] = M[0,1]
		M[2,0] = M[0,2]
		M[2,1] = M[1,2]
		
		M_list.append(M)
	i = 0
	for v in l:
		print('------------------------------------------------')
		print('vacuum =')
		pprint(v)
		print('')
		pprint(M_list[i])
		i += 1
	 














	
	
	
	
