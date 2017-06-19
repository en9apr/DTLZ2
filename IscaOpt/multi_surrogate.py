#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Multi-Surrogate Approaches to Multi-Objective Bayesian Optimisation
================================================================================
:Author:
   Alma Rahat   <A.A.M.Rahat@exeter.ac.uk>
:Date:
   19 April 2017
:Copyright:
   Copyright (c)  Alma Rahat, University of Exeter, 2017
:File:
   multi_surrogate.py
"""
# imports
import numpy as np
from scipy.special import erf as ERF
from scipy.stats import norm as NORM
try:
    from surrogate import MultiSurrogates
except:
    from .surrogate import MultiSurrogates
import _csupport as CS
from evoalgos.performance import FonsecaHyperVolume as FH
try:
    from BO_base import BayesianOptBase
except:
    from .BO_base import BayesianOptBase


class MultiSurrogate(BayesianOptBase):

    def __init__(self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None, \
                    kern=None, ref_vector=None):
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                                obj_sense, args, kwargs, X, Y,\
                                ref_vector=ref_vector)
        self.kernels = kern
        self.hpv = FH(self.ref_vector)
        
    def get_toolbox(self, xtr, skwargs, cfunc=None, \
                        cargs=(), ckwargs={}, verbose=True):
        self.budget = skwargs.get('budget', 250)
        self.update(xtr)
        return self.init_deap(self.scalarise, cfunc=cfunc, cargs=cargs, \
                        ckwargs=ckwargs)
        
    def build_models(self, xtr, ytr, kernels):
        self.xtr = xtr
        self.ytr = ytr
        models = MultiSurrogates(xtr, ytr, kernels)
        return models
    
    def update(self, x_new):
        # set/update xtr and ytr
        self.xtr = x_new
        self.ytr = self.m_obj_eval(x_new)
        assert self.xtr.shape[0] == self.ytr.shape[0]
        yt, comp_mat = self.get_dom_matrix(self.ytr)
        self.comp_mat = comp_mat
        # update budget count
        self.b_count = self.budget - len(self.xtr) - 1 
        # update + optimise models
        self.models = self.build_models(self.xtr, self.ytr, \
                            [kern.copy() for kern in self.kernels])
        # current pf
        self.pfr_inds = self.get_front(self.ytr, self.comp_mat)
        # current hv
        self.current_hv = self.current_hpv()#self.hpv.assess_non_dom_front(self.ytr[self.pfr_inds])
        # epsilon for sms-ego
        n_pfr = len(self.pfr_inds)
        print(n_pfr)
        c = 1 - (1/ 2**self.n_obj)
        self.epsilon = (np.max(self.ytr, axis=0) - np.min(self.ytr, axis=0))\
                        /(n_pfr + (c * self.b_count))
                        
    
        
class MPoI(MultiSurrogate):

    def __init__(self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None, \
                    kern=None, ref_vector=None):
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                                obj_sense, args, kwargs, X, Y,\
                                kern=kern, ref_vector=ref_vector)
        
    def scalarise(self, x, cfunc=None, cargs=(), ckwargs={}):
        '''
        Calculate the minimum probability of dominance compared to current 
        Pareto front f.
        
        parameters:
        -----------
        f: n x d array of n Pareto front elements (objective values)
        u: m x d array of means (m tentative solutions)
        s: m x d array of standard deviations
        '''
        n_sols = x.shape[0]
        if cfunc is not None:
            if not cfunc(x, *cargs, **ckwargs):
                return np.zeros((1,1))# penalise
        yp, stdp = self.models.predict(x)
        y = self.ytr[self.pfr_inds]
        res = np.zeros((yp.shape[0], 1))
        sqrt2 = np.sqrt(2)
        for i in range(yp.shape[0]):
            # probability that u dominates any f
            m = (yp[i] - y)/(sqrt2 * stdp[i])
            pdom = 1 - np.prod(0.5 * (1 + ERF(m)), axis=1)
            res[i] = np.min(pdom)
        return res
        
class SMSEGO(MultiSurrogate):
    
    def __init__(self, func, \
                    n_dim, n_obj, lower_bounds, upper_bounds, \
                    obj_sense=[-1, -1], args = (), kwargs={}, X=None, Y=None, \
                    kern=None, ref_vector=None):
        super().__init__(func, n_dim, n_obj, lower_bounds, upper_bounds, \
                                obj_sense, args, kwargs, X, Y,\
                                kern=kern, ref_vector=ref_vector)
        self.gain = - NORM.ppf(0.5 * (0.5**(1/self.n_obj)))

    def compare_add_solution(self, y, ytest, obj_sense):
        result = np.ones(y.shape[0])
        for i in range(y.shape[0]):
            result[i] = CS.compare_solutions(y[i], ytest, self.obj_sense)
            if result[i] == 0:
                return y
        inds = np.where(result==3)[0]
        try:
            return np.concatenate([y[inds], ytest])
        except ValueError:
            print("Likely error in y: ", y[inds])
            return ytest
                        
    def penalty(self, y, y_test):
        yt = y_test - (self.epsilon * self.obj_sense)
        l = [-1 + np.prod(1 + y_test - y[i]) \
                if CS.compare_solutions(y[i], yt, self.obj_sense) == 0\
                else 0 for i in range(y.shape[0])]
        return (max([0, max(l)]))
        
    def scalarise(self, x, cfunc=None, cargs=(), ckwargs={}):
        '''
        Calculate the minimum probability of dominance compared to current 
        Pareto front f.
        
        parameters:
        -----------
        f: n x d array of n Pareto front elements (objective values)
        u: m x d array of means (m tentative solutions)
        s: m x d array of standard deviations
        '''
        n_sols = len(x)
        if cfunc is not None:
            if not cfunc(x, *cargs, **ckwargs):
                return np.ones((1,1))*-100 # heavily penalise
        # predictions
        yp, stdp = self.models.predict(x)
        # lower confidence bounds
        yl = yp - (self.gain * np.multiply(self.obj_sense, stdp))
        pen = self.penalty(self.ytr[self.pfr_inds], yl)
        if pen > 0:
            return np.array([-pen])            
        # new front
        yn = self.compare_add_solution(self.ytr[self.pfr_inds], yl, self.obj_sense)
        return np.array([self.hpv.assess_non_dom_front(yn) - self.current_hv])


