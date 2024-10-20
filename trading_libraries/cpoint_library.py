import numpy as np
from scipy.stats import norm
import math
from stochastic_library import *

#--------------------------------------------------------------#
# Nadaraya-Watson kernel-weighted average with Gaussian kernel
#--------------------------------------------------------------#
def ND_kernel_average_gauss( Xq: float , X_obs: np.array , Y_obs: np.array , bw: float ):

    num = 0
    den = 0
    for i in range( len(X_obs) ):
        pdf_i = dnorm( Xq , X_obs[i] , bw )   # normalized Gaussian that sums to 1
        num   = num + ( pdf_i*Y_obs[i] )
        den   = den + pdf_i
    
    Yq = num/den                              # Nadaraya-Watson kernel-weighted average

    return Yq

#--------------------------------------------------------------#
# Non-parametric drift estimation
#--------------------------------------------------------------#
def nonparametric_drift( X_obs: np.array , dt: float ):

    n = len(X_obs)
    bw = pow( n , -1/5 ) * math.sqrt( sample_variance(X_obs) )  # See Scott or Silverman's rule of thumbs

    np_drift = np.zeros((len(X_obs)-1,1))
    Y_obs    = np.zeros((len(X_obs)-1,1))

    for i in range( len(np_drift) ):
        Y_obs[i] = (X_obs[i+1]-X_obs[i])/dt

    for i in range( len(np_drift) ):
        Xq    = X_obs[i]
        np_drift[i] = ND_kernel_average_gauss( Xq , X_obs[:-1] , Y_obs , bw )

    return np_drift

#### TO IMPLEMENT BANDWITH CROSS VALIDATION COMPUTATION ####

#--------------------------------------------------------------#
# Change point estimation
#--------------------------------------------------------------#
def cpoint( X_obs: np.array , dt: float ):
     
    #--------------------------------------------------------------#
    # Standardized residuals 
    #--------------------------------------------------------------#
    Z     = np.zeros((len(X_obs)-1,1))
    drift = nonparametric_drift( X_obs , dt )

    lenZ = len(Z)
    for i in range( lenZ ):
        Z[i] = ( X_obs[i+1]-X_obs[i]-drift[i]*dt )/math.sqrt(dt)

    #--------------------------------------------------------------#
    # Estimation
    #--------------------------------------------------------------#
    maxD    = 0
    argmaxD = 0
    D  = np.zeros((lenZ,1))
    S  = np.zeros((lenZ,1))
    Sn = sum( np.power( Z , 2 ) )
    for i in range( lenZ ):
        S[i] = sum( np.power( Z[0:i] , 2 ) )
        D[i] = i/lenZ - S[i]/Sn

        if ( maxD < D[i] ):
            maxD    = D[i]
            argmaxD = i

    k0     = argmaxD+1
    theta1 = S[k0]/k0
    theta2 = S[lenZ-k0]/(lenZ-k0)

    return k0 , theta1 , theta2