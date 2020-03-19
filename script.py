from IPython.display import clear_output
# example set up
import numpy as np
# import optimiser codes
import IscaOpt

from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce

settings = {\
    'n_dim': 2,\
    'n_obj': 2,\
    'lb': np.zeros(2),\
    'ub': np.ones(2),\
    'ref_vector': [2.5]*2,\
    'method_name': 'HypI',\
    'budget':40,\
    'n_samples':21,\
    'visualise':True,\
    'multisurrogate':False}

# APR added test function
def DTLZ2_BO_Function(decision_vector):
    """DTLZ2 multiobjective function. It returns a tuple of *obj* values. 
    The individual must have at least *obj* elements.
    From: K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective 
    Optimization Test Problems. CEC 2002, p. 825 - 830, IEEE Press, 2002.
    
    :math:`g(\\mathbf{x}_m) = \\sum_{x_i \in \\mathbf{x}_m} (x_i - 0.5)^2`
    
    :math:`f_{\\text{DTLZ2}1}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\prod_{i=1}^{m-1} \\cos(0.5x_i\pi)`
    
    :math:`f_{\\text{DTLZ2}2}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\sin(0.5x_{m-1}\pi ) \\prod_{i=1}^{m-2} \\cos(0.5x_i\pi)`
    
    :math:`\\ldots`
    
    :math:`f_{\\text{DTLZ2}m}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\sin(0.5x_{1}\pi )`
    
    Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m` is a
    vector of the remaining attributes :math:`[x_m~\\ldots~x_n]` of the
    individual in :math:`n > m` dimensions.
    """
    obj=2
    xc = decision_vector[:obj-1]
    xm = decision_vector[obj-1:]
    g = sum((xi-0.5)**2 for xi in xm)
    f = [(1.0+g) *  reduce(mul, (cos(0.5*xi*pi) for xi in xc), 1.0)]
    f.extend((1.0+g) * reduce(mul, (cos(0.5*xi*pi) for xi in xc[:m]), 1) * sin(0.5*xc[m]*pi) for m in range(obj-2, -1, -1))

    return f
# APR added test function



# function settings
from deap import benchmarks as BM
fun = BM.dtlz2
args = (2,) # number of objectives as argument

# optimise
res = IscaOpt.Optimiser.EMO(DTLZ2_BO_Function, settings=settings)
clear_output()