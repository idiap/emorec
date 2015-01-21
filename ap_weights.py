#    Copyright (c) 2014 Idiap Research Institute, http://www.idiap.ch/
#    Written by Nikolaos Pappas <nikolaos.pappas@idiap.ch>,
#
#    This file is part of EMORec.
#
#    EMORec is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 as
#    published by the Free Software Foundation.
#
#    EMORec is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Emorec. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from scipy import sparse
from scipy import linalg 
from numpy import dot, matrix
from sklearn import linear_model
from numpy.linalg import norm
from scipy.sparse import lil_matrix
from sklearn.base import BaseEstimator
from constrained_rls import cRLS 
from sklearn.metrics import mean_absolute_error	


class APWeights(BaseEstimator):

	def __init__(self, iterations, l1=.5, l2=.5, l3=10, reg=None):
		self.iterations = iterations 
		self.l1 = l1					
		self.l2 = l2  					
		self.l3 = l3					
		self.f1 = None
		self.f2 = None
		self.f3 = None  


	def fit(self, X, Y):
		M = X[0].get_shape()[1]      # number of features
		N = len(X)                   # number of instances 
		F = np.random.ranf((1,M))    # hyperplane to be learned
 		H = matrix(np.zeros((N,M)))  # bag representations
 		P = []
		Y_w = []
		X_w = []
		converged = False
		prev_error = 999999
		it = 0
		print "-"*100

		print "L1: %f" % self.l1
		print "L2: %f" % self.l2
		print "L3: %f" % self.l3
		print "M: %d" % M
		print "N: %d" % N

		print
		print "[+] Training..." 
		while(not converged and it < self.iterations):
			for i, Xi in enumerate(X): 
				if it == 0:
					if X_w == []:
						X_w = Xi
					else:						
						X_w = sparse.vstack([X_w, Xi ]) 
					P.append(np.ones((1,X[i].get_shape()[0]))) 
					Y_w.append([])
				Xi = Xi.tocsr()
				if self.f2: 
					HC = matrix(self.f2.predict(Xi)).T 
				else:
					HC = Xi.dot(F.T).T
				self.f1 = cRLS(alpha=self.l1)
				P[i] = self.f1.fit(HC,Y[i],P[i])  
				Y_w[i] = self.f1.coef_
				cur_p = sparse.csr_matrix(self.f1.coef_)
				H[i] = cur_p.dot(Xi).todense()

			self.f2 = linear_model.Ridge(alpha=self.l2)
			self.f2.fit(H,Y) 
			pred = self.f2.predict(H)
			cur_error = mean_absolute_error(pred,Y)
			print "iteration %d -> (MAE: %f) " % (it, cur_error)
			self.coef_ = self.f2.coef_
			if prev_error - cur_error < 0.000001:
				converged = True
				self.coef_ = self.f2.coef_
			prev_error = cur_error
			it += 1 
		Y_w = np.hstack(Y_w) 
		print "Training f3..."
		self.f3 = linear_model.Ridge(alpha=self.l3)  
		self.f3.fit(X_w,Y_w)		

		self.P = P
		self.H = H
		print "--/end"
		
		return F


	def predict(self, X):
		M = X[0].get_shape()[1]       
		N = len(X)				   	
		H = matrix(np.zeros((N,M)))
 		W = []
		for i, instances in enumerate(X):
			Xi = instances 
			weights = matrix(self.f3.predict(Xi)).view(np.ndarray)[0]
			nweights = weights/sum(weights)
			W.append(nweights)
			H[i] =  dot(weights, Xi.todense())[0] 
		Y = self.f2.predict(H)
		self.P_test = W
		return Y

