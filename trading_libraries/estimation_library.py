import numpy as np
from scipy.optimize import minimize 

########################################################################################
######################################### QMLE #########################################
########################################################################################
#--------------------------------------------------------------#
# Quasi MLE objective function = logaritmic function
#--------------------------------------------------------------#
def qmle_obj_fcn( params: np.array , X: np.array , dt: float ):

    # X  = stochastic process samples
    # dt = sampling time
    # params = array of parameters [ alpha beta sigma gamma ]

    alpha = params[0]
    beta  = params[1]
    sigma = params[2]
    gamma = params[3]

    lenX = len(X)

    Jln = 0
    for i in range(lenX-1):

        term_1 = 2*np.log( sigma*np.power( X[i] , gamma ) )
        term_2 = np.power( X[i+1]-X[i] - (alpha+beta*X[i])*dt , 2 )/( dt*np.power( sigma*np.power( X[i] , gamma ) , 2 ) )

        Jln = Jln + term_1 + term_2

    return Jln/2

#--------------------------------------------------------------#
# Quasi MLE estimation function
#--------------------------------------------------------------#
def qmle_estimation( X: np.array , dt: float , theta_0: np.array , theta_min: np.array , theta_max: np.array ):

    alpha_min = theta_min[0]
    beta_min  = theta_min[1]
    sigma_min = theta_min[2]
    gamma_min = theta_min[3]

    alpha_max = theta_max[0]
    beta_max  = theta_max[1]
    sigma_max = theta_max[2]
    gamma_max = theta_max[3]

    # options = {'maxiter': 250, 'gtol': 1e-06, 'xtol': 1e-04, 'verbose': 1}
    options = {'maxiter': 250, 'gtol': 1e-06, 'xtol': 1e-05, 'verbose': 1}

    bnds = ( ( alpha_min , alpha_max ) , ( beta_min , beta_max ) , ( sigma_min , sigma_max ) , ( gamma_min , gamma_max ) )

    theta_qmle = minimize( qmle_obj_fcn , theta_0 , args = ( X , dt ) , bounds = bnds , method = 'trust-constr' , tol=1e-6 , options = options )

    return theta_qmle

########################################################################################
######################################## LASSO #########################################
########################################################################################
#--------------------------------------------------------------#
# LASSO objective function = Hessian of QMLE
#--------------------------------------------------------------#
def lasso_obj_fcn( params:np.array , params_tilde: np.array , penalties: np.array , delta: np.array , X: np.array , dt: float , H: np.array ):

    # X  = stochastic process samples
    # dt = sampling time
    # params = array of parameters [ alpha beta sigma gamma ]

    alpha = params[0];             beta = params[1];             sigma = params[2];             gamma = params[3]
    alpha_tilde = params_tilde[0]; beta_tilde = params_tilde[1]; sigma_tilde = params_tilde[2]; gamma_tilde = params_tilde[3]
    alpha_0 = penalties[0];        beta_0 = penalties[1];        sigma_0 = penalties[2];        gamma_0 = penalties[3]
    delta_1  = delta[0];           delta_2 = delta[1];           delta_3 = delta[2];            delta_4 = delta[3]

    alpha_n = alpha_0*np.power( np.abs( alpha_tilde ) , -delta_1 )
    beta_n  = beta_0*np.power(   np.abs( beta_tilde ) , -delta_2 )
    sigma_n = sigma_0*np.power( np.abs( sigma_tilde ) , -delta_3 )
    gamma_n = gamma_0*np.power( np.abs( gamma_tilde ) , -delta_4 )

    t_alpha = alpha_n*np.abs( alpha ); t_beta = beta_n*np.abs( beta ); t_sigma = sigma_n*np.abs( sigma ); t_gamma = gamma_n*np.abs( gamma )
    J = np.dot( (params-params_tilde) , np.dot( H , (params-params_tilde) )) + t_alpha + t_beta + t_sigma + t_gamma

    return J

#--------------------------------------------------------------#
# Hessian computation for LASSO
#--------------------------------------------------------------#
def lasso_hessian( params: np.array , Xi: float , Delta_X: float , dt: float ):

    alpha = params[0]
    beta  = params[1]
    sigma = params[2]
    gamma = params[3]

    Hi = np.zeros( [ 4 , 4 ] )

    Hi[0,0] = dt*np.power( Xi , -2*gamma )*np.power( sigma , -2 )
    Hi[0,1] = dt*Xi*np.power( Xi , -2*gamma )*np.power( sigma , -2 )
    Hi[0,2] = 2*( Delta_X - dt*( alpha+beta*Xi ) )*np.power( Xi , -2*gamma )*np.power( sigma , -3 )
    Hi[0,3] = 2*( Delta_X - dt*( alpha+beta*Xi ) )*np.log( Xi )*np.power( Xi , -2*gamma )*np.power( sigma , -2 )

    Hi[1,0] = dt*Xi*np.power( Xi , -2*gamma )*np.power( sigma , -2 )
    Hi[1,1] = dt*np.power( Xi , 2 )*np.power( Xi , -2*gamma )*np.power( sigma , -2 )
    Hi[1,2] = 2*( Delta_X - dt*( alpha+beta*Xi ) )*Xi*np.power( Xi , -2*gamma )*np.power( sigma , -3 )
    Hi[1,3] = 2*( Delta_X - dt*( alpha+beta*Xi ) )*Xi*np.log( Xi )*np.power( Xi , -2*gamma )*np.power( sigma , -2 )

    Hi[2,0] = 2*( Delta_X - dt*( alpha+beta*Xi ) )*np.power( Xi , -2*gamma )*np.power( sigma , -3 )
    Hi[2,1] = 2*( Delta_X - dt*( alpha+beta*Xi ) )*Xi*np.power( Xi , -2*gamma )*np.power( sigma , -3 )
    Hi[2,2] = -np.power( sigma , -2 ) + 3*np.power( ( Delta_X - dt*( alpha+beta*Xi ) ) , 2 )*np.power( Xi , -2*gamma )*np.power( sigma , -4 )*np.power( dt , -1 )
    Hi[2,3] = 2*np.power( ( Delta_X - dt*( alpha+beta*Xi ) ) , 2 )*np.log( Xi )*np.power( Xi , -2*gamma )*np.power( sigma , -3 )*np.power( dt , -1 )

    Hi[3,0] = 2*( Delta_X - dt*( alpha+beta*Xi ) )*np.log( Xi )*np.power( Xi , -2*gamma )*np.power( sigma , -2 )
    Hi[3,1] = 2*( Delta_X - dt*( alpha+beta*Xi ) )*Xi*np.log( Xi )*np.power( Xi , -2*gamma )*np.power( sigma , -2 )
    Hi[3,2] = 2*np.power( ( Delta_X - dt*( alpha+beta*Xi ) ) , 2 )*np.log( Xi )*np.power( Xi , -2*gamma )*np.power( sigma , -3 )*np.power( dt , -1 )
    Hi[3,3] = 2*np.power( ( Delta_X - dt*( alpha+beta*Xi ) ) , 2 )*np.power( np.log( Xi ) , 2 )*np.power( Xi , -2*gamma )*np.power( sigma , -2 )*np.power( dt , -1 )

    return Hi

#--------------------------------------------------------------#
# LASSO estimation function
#--------------------------------------------------------------#
def lasso_estimation( X: np.array , dt: float , theta_0: np.array , theta_tilde: np.array , theta_min: np.array , theta_max: np.array , 
                     penalties: np.array , delta: np.array ):

    alpha_min = theta_min[0]
    beta_min  = theta_min[1]
    sigma_min = theta_min[2]
    gamma_min = theta_min[3]

    alpha_max = theta_max[0]
    beta_max  = theta_max[1]
    sigma_max = theta_max[2]
    gamma_max = theta_max[3]

    lenX = len(X)

    H = np.zeros( [ 4 , 4 ] )
    for i in range(lenX-1):

        Delta_X = X[i+1]-X[i]

        Hi = lasso_hessian( theta_tilde , X[i] , Delta_X , dt )

        H = H + Hi

    options = {'maxiter': 250, 'gtol': 1e-06, 'xtol': 1e-05, 'verbose': 1}

    bnds = ( ( alpha_min , alpha_max ) , ( beta_min , beta_max ) , ( sigma_min , sigma_max ) , ( gamma_min , gamma_max ) )
    
    theta_lasso = minimize( lasso_obj_fcn , theta_0 , args = ( theta_tilde , penalties , delta , X , dt , H ) , bounds = bnds , method = 'trust-constr' , tol=1e-6 , options = options )

    return theta_lasso