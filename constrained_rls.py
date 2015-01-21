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
#    along with EMORec. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import fmin_slsqp
from numpy.linalg import norm
from numpy import dot, matrix
from scipy import sparse

class cRLS:
	def __init__(self, alpha=0.0): 
		self.alpha = alpha	
		self.coef_ = []
		
	def loss(self, coef_, X, y):
 		return  pow(y - (dot(X, coef_)), 2) + self.alpha*norm(coef_,2)

 	def eq_con(self, coef_, *params):
 		return np.array([sum(coef_) - 1.])

	def fit(self, X, Y, P): 
		X = X.view(np.ndarray)[0] 
		self.coef_ = P.view(np.ndarray)[0] 
		bounds = []
		for c in self.coef_:
			bounds.append( (0.00001,1.00001) )
		out = fmin_slsqp(self.loss, self.coef_, args=(X, Y),
						f_eqcons=self.eq_con, iter=100,
						bounds=bounds, iprint=0, full_output=0)
		self.coef_ = out
		return matrix(self.coef_)
