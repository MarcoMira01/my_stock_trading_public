import numpy as np
import random
import math
from scipy.stats import norm
import pymle.models as py_mod
from pymle.core.TransitionDensity import *
from pymle.sim.Simulator1D import Simulator1D

#--------------------------------------------------------------#
# Sample mean computation
#--------------------------------------------------------------#
def sample_mean( x: np.array ):

        mean  = 0

        # Mean
        mean = sum(x) / len(x)
        
        return mean

#--------------------------------------------------------------#
# Sample variance computation
#--------------------------------------------------------------#
def sample_variance( x: np.array ):

        variance  = 0

        # Mean
        mean = sample_mean( x )
        
        # Variance
        variance = np.add( x , -mean )
        variance = np.power( variance , 2 )
        variance = np.sum(variance) / (len(x)-1)      # estimator of (sample) variance
        
        return variance

#--------------------------------------------------------------#
# Gaussian kernel
#--------------------------------------------------------------#
def dnorm( x , mean = 0 , sd = 1 ):

    result = norm.pdf( x , loc=mean , scale=sd )
    return result

#--------------------------------------------------------------#
# Montecarlo simulation
#--------------------------------------------------------------#
def Montecarlo_simulation( X0: float , T: int , dt: float ,
    model_sim: py_mod , N: int , seed ):

    # X0 = initial value of process
    # T  = length of the simulation (number of samples in a row)
    # seed: seed of the random simulation, set to None to get new results each time


    X = np.zeros([N,T+1], dtype=float)
    for i in range(N):
        simulator  = Simulator1D(S0=X0, M=T, dt=dt, model=model_sim).set_seed(seed=seed)
        sample = np.round(simulator.sim_path(),3)
        # sample_sim = resolve_nan_1D( sample_sim )
        for k in range(len(sample)):
            X[i][k] = sample[k]

    return X

#--------------------------------------------------------------#
# Resolve nan in array
#--------------------------------------------------------------#
def resolve_nan_1D( x: np.array ):
    w = np.where(np.isnan(x))
    if len(w[0] != 0):
        w = w[0]
        for j in range(len(w)):
            x[w[j]] = x[w[j]-1]

    return x

#--------------------------------------------------------------#
# Resolve nan in matrix
#--------------------------------------------------------------#
def resolve_nan_2D( x: np.ndarray ):
    for i in range(len(x[:,0])):
        w = np.where(np.isnan(x[i,:]))
        if len(w[0] != 0):
            w = w[0]
            for j in range(len(w)):
                x[i,w[j]] = x[i,w[j]-1]

    return x 

#########################################################################
#########################################################################
#--------------------------------------------------------------#
# CKLS model: one step integration with Euler-Maruyama scheme
#--------------------------------------------------------------#
def CKLS_EulerMaruyama( X0: float , dt: float , params: np.array ):
     # X0 = initial value
     # dt = sampling time
     # params = array of parameters [ alpha beta sigma gamma ]

     alpha = params[0]
     beta  = params[1]
     sigma = params[2]
     gamma = params[3]

     X = X0 + (alpha+beta*X0)*dt + sigma*np.power( X0 , gamma )*math.sqrt(dt)*random.gauss( 0 , 1 )

     return X

def CKLS_process( X0: float , T: float , dt: float , params: np.array ):
     
    # X0 = initial value
    # T  = number of samples 
    # dt = sampling time
    # params = array of parameters [ alpha beta sigma gamma ]

    X = np.zeros( (T+1,1) )
    X[0] = X0
    for i in range(T):
         X[i+1] = CKLS_EulerMaruyama( X[i] , dt , params )

    return X