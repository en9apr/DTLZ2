#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Mono-Surrogate Approaches to Multi-Objective Bayesian Optimisation
================================================================================
:Author:
   Alma Rahat   <A.A.M.Rahat@exeter.ac.uk>
:Date:
   19 April 2017
:Copyright:
   Copyright (c)  Alma Rahat, University of Exeter, 2017
:File:
   mono_surrogate.py
"""

# imports
import numpy as np
import itertools
from evoalgos.performance import FonsecaHyperVolume as FH
import time
try: 
    from BO_base import BayesianOptBase
    from surrogate import Surrogate
except:
    from .BO_base import BayesianOptBase
    from .surrogate import Surrogate


class MonoSurrogate(BayesianOptBase):
    """
    Mono-surrogate base class.
    """

    def __init__(self, func, n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None, \
                    kern=None, ref_vector=None):
        """This constructor creates the mono-surrogate base class.
        
        Parameters.
        -----------
        func (method): the method to call for actual evaluations
        n_dim (int): number of decision space dimensions
        n_obj (int): number of obejctives
        lower_bounds (np.array): lower bounds per dimension in the decision space
        upper_boudns (np.array): upper bounds per dimension in the decision space
        obj_sense (np.array or list): whether to maximise or minimise each 
                objective (ignore this as it was not tested). 
                keys. 
                    -1: minimisation
                     1: maximisation
        args (tuple): tuple of arguments the objective function requires
        kwargs (dictionary): dictionary of keyword argumets to pass to the 
                            objective function. 
        X (np.array): initial set of decision vectors.
        Y (np.array): initial set of objective values (associated with X).
        ref_vector (np.array): reference vector in the objective space.
        """
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                                obj_sense, args, kwargs, X, Y,\
                                ref_vector=ref_vector)
        self.kernel = kern
        
    def get_toolbox(self, xtr, skwargs, cfunc=None, \
                        cargs=(), ckwargs={}, verbose=True):
        ytr = self.scalarise(xtr, kwargs=skwargs)
        self.current_hv = self.current_hpv()
        surr = Surrogate(xtr, ytr, self.kernel.copy(), verbose=verbose)
        return self.init_deap(surr.expected_improvement, obj_sense=1, cfunc=cfunc, cargs=cargs, \
                        ckwargs=ckwargs, lb=self.lower_bounds, ub=self.upper_bounds)
        
class HypI(MonoSurrogate):

    def __init__ (self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None,\
                    kern=None, ref_vector=None):
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                                obj_sense, args, kwargs, X, Y, kern=kern,\
                                ref_vector=ref_vector)
        
    def scalarise(self, x, kwargs={}):
        '''
        ref_vector=None, approximate_ref=False
        Scalarise a multi-objective problem using hyper-volume contribution.
        
        S_i is the set of solutions in shell i. The common area is the hypervolume
        of the next shell. 
        F(x_k, S_i, S_(i+1)) = H({x_k} U S_(i+1)) 
        
        parameters. 
        -----------
        x (numpy array): 
        '''
        start = time.time()
        ref_vector = kwargs.get('ref_vector', None)
        approximate_ref = kwargs.get('approximate_ref', False)
        y = self.m_obj_eval(x)
        self.X = x
        n_data = x.shape[0]
        h = np.zeros(n_data)
        if approximate_ref:
            ref_vector = np.max(y, axis=0) + 0.1 * (np.max(y, axis=0) - np.min(y, axis=0))
            print("New Reference vector: ", ref_vector)
        y, comp_mat = self.get_dom_matrix(y, ref_vector)
        # shell hpv calculations
        shells = []
        h_shells = []
        loc_comp_mat = comp_mat.copy()
        hpv = FH(ref_vector)
        del_inds = []        
        while True:
            fr_inds = self.get_front(y, loc_comp_mat, del_inds)
            if fr_inds.shape[0] == 0:
                break
            shells.append(fr_inds)
            h_shells.append(hpv.assess_non_dom_front(y[fr_inds]))
            del_inds = np.concatenate([fr_inds, del_inds], axis=0)
            loc_comp_mat[:,fr_inds] = loc_comp_mat[fr_inds, :] = -1
        n_shells = len(shells)
        for i in range(n_shells-1):
            for j in shells[i]:
                comp_row = comp_mat[j]
                # find dominated next shell indices
                nondominated = np.where(comp_row[shells[i+1]] == 3)[0]
                nfr = np.concatenate([[j], shells[i+1][nondominated]])
                h[j] = hpv.assess_non_dom_front(y[nfr])
        print('Total time: ', (time.time() - start)/60.0)
        return np.reshape(h, (-1, 1))        

        
class MSD(MonoSurrogate):
    
    def __init__ (self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None,\
                    kern=None, ref_vector=None):
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                                obj_sense, args, kwargs, X, Y, kern=kern,\
                                ref_vector=ref_vector)
    
    def scalarise(self, x, kwargs={}):
        """
        Min-max distance from the Pareto front.
        """
        start = time.time()
        y = self.m_obj_eval(x)
        self.X = x
        n_data = x.shape[0]
        h = np.zeros(n_data)
        y, comp_mat = self.get_dom_matrix(y)
        front_inds = self.get_front(y, comp_mat)
        for i in range(n_data):
            if i not in front_inds:
                dist = [np.sum(y[k]-y[i]) for k in front_inds]
                h[i] = np.min(dist)
        print('Total time: ', (time.time() - start)/60.0)
        return np.reshape(h, (-1, 1))
        
        
class DomRank(MonoSurrogate):
    
    def __init__ (self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None,\
                    kern=None, ref_vector=None):
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                                obj_sense, args, kwargs, X, Y, kern=kern,\
                                ref_vector=ref_vector)

    def scalarise(self, x, kwargs={}):
        """
        H[i] = # that dominates i / (n - 1), where
        n = # of solutions
        """
        start = time.time()
        y = self.m_obj_eval(x)
        self.X = x
        n_data = x.shape[0]
        #print('Total data points:', n_data)
        h = np.zeros(n_data)
        y, comp_mat = self.get_dom_matrix(y)
        front_inds = self.get_front(y, comp_mat)
        for i in range(n_data):
            if i not in front_inds:
                row = comp_mat[i,:]
                count = np.where(row == 1)[0].shape[0]
                count = count / (n_data - 1)
                h[i] = count 
        h = 1 - h
        print('Total time: ', (time.time() - start)/60.0)
        return np.reshape(h, (-1, 1))
        

class ParEGO(MonoSurrogate):
   
    def __init__ (self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None,\
                    kern=None, ref_vector=None):
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                        obj_sense, args, kwargs, X, Y, kern=kern,\
                        ref_vector=ref_vector)
                        
                        
    def normalise(self, y):
        """
        normalise cost functions.
        according to parego the known or estimated limits are used. 
        here we use limits estimated from data.
        """
        min_y = np.min(y, axis=0)
        max_y = np.max(y, axis=0)
        return (y - min_y)/(max_y - min_y)
        
    def get_lambda(self, s, n_obj):
        """
        s is the total number of vectors
        """
        try:
            self.l_set
        except:
            l = [np.arange(s+1, dtype=int) for i in range(n_obj)]
            self.l_set = np.array([np.array(i) \
                    for i in itertools.product(*l) if np.sum(i) == s])/s
            print("Number of scalarising vectors: ", self.l_set.shape[0])
        ind = np.random.choice(np.arange(self.l_set.shape[0], dtype=int))
        return self.l_set[ind]
        
    def scalarise(self, x, kwargs={}):
        """
        transform cost functions with augmented chebyshev: parego style.
        """
        s = kwargs.get('s', 5)
        rho = kwargs.get('rho', 0.05)
        y = self.m_obj_eval(x)
        self.X = x
        y_norm = self.normalise(y)
        lambda_i = self.get_lambda(s, y.shape[1])
        new_y = np.max(y_norm * lambda_i, axis=1) + (rho * np.dot(y, lambda_i))
        return np.reshape(-new_y, (-1, 1))
        
class EGO(MonoSurrogate):
   
    def __init__ (self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=-1, args = (), kwargs={}, X=None, Y=None,\
                    kern=None, ref_vector=None):
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                        obj_sense, args, kwargs, X, Y, kern=kern,\
                        ref_vector=ref_vector)
                                
    def get_toolbox(self, xtr, skwargs, cfunc=None, \
                        cargs=(), ckwargs={}, verbose=True):
        ytr = self.scalarise(xtr, kwargs=skwargs)
        self.current_hv = self.current_hpv()
        surr = Surrogate(xtr, ytr, self.kernel.copy(), verbose=verbose)
        self.surr = surr
        return self.init_deap(surr.expected_improvement, obj_sense=self.obj_sense[0], \
                        cfunc=cfunc, cargs=cargs, ckwargs=ckwargs, \
                        lb=self.lower_bounds, ub=self.upper_bounds)
        
    def scalarise(self, x, kwargs={}):
        """
        single objective, hence send back whatever it is
        """
        y = self.m_obj_eval(x)
        self.X = x
        return y
        
        
