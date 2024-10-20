import datetime as datetime
from gekko import GEKKO
import numpy as np
import pymle.models as py_mod
from stochastic_library import *

#--------------------------------------------------------------#
# Stochastic MPC
#--------------------------------------------------------------#
def SMPC_control( horizon_length: int , nr_of_sim: int , sims: np.ndarray ,
    initial_value: float , risk_free_fact: float , risk_aversion_coef: float , inv_limit: float , trd_type: str ):

    m = GEKKO(remote=False)   # use local CPU, set true to use remote server from gekko

    # m.DIAGLEVEL     = 1           # return messages
    # m.options.IMODE = 8
    m.options.SOLVER  = 1           # change solver (1=APOPT,3=IPOPT)

    V0   = initial_value
    r_f  = risk_free_fact
    beta = inv_limit

    # Define the model variables
    # V = account value
    # u = amount of shares to buy (long position) or sell (short position)
    V = m.Array( m.Var , (nr_of_sim,horizon_length+1) )     # from 0 to N
    # u = m.Array( m.MV , (1) )                  # from 0 to N-1
    u = m.MV()

    for i in range(nr_of_sim):
        for k in range(horizon_length+1):
            V[i,k].LOWER = 100

    u.STATUS = 1            # enable the algorithm to vary it

    if ( trd_type.lower() == 's' ):
        u.UPPER = 0           # only short position
    elif ( trd_type.lower() == 'l' ):
        u.LOWER = 0           # only long position
    
    # OTHERWISE: both long and short positions

    #------------------------------------------------------------------------#
    # Optimization 
    #------------------------------------------------------------------------#
    # Constraints
    for i in range(nr_of_sim):
        m.Equations( [ V[i,0] == V0 ] )
        for k in range(horizon_length):
            m.Equations( [ V[i,k+1] == V[i,k]*(1+r_f) + u*( sims[i][k+1]-(1+r_f)*sims[i][k] ) ] )
            m.Equation( u*sims[i][k] <= beta*V[i,k] )
            m.Equation( -u*sims[i][k] <= beta*V[i,k] )


    # Objective function
    a1 = 1/nr_of_sim
    a2 = risk_aversion_coef/(2*(nr_of_sim**2))
    a3 = -risk_aversion_coef/(2*nr_of_sim)

    # Expected value
    m.Maximize( a1*sum( [ V[i,horizon_length] for i in range(nr_of_sim) ] ) )

    # Variance
    m.Maximize( a2*(sum( [ V[i,horizon_length] for i in range(nr_of_sim) ] ))**2 )
    m.Maximize( a3*sum( [ V[i,horizon_length]**2 for i in range(nr_of_sim) ] ) )

    m.solve(disp=False)

    ctrl_input = np.array(u.value)
    ctrl_input = np.round(ctrl_input.item(),2)

    return ctrl_input

#--------------------------------------------------------------#
# Investment computation
#--------------------------------------------------------------#
def Stochastic_investment( nr_of_itr: int , inv_thd: float , asset_initial_value: float , sim_length: int , dt: float ,
    model_sim: py_mod , nr_of_sim: int , seed , account_initial_value: float , risk_free_fact: float , risk_aversion_coef: float , 
    inv_limit: float ):

    S0 = asset_initial_value
    T  = sim_length
    V0 = account_initial_value

    u = np.zeros((nr_of_itr,1))
    inv_nr = 0
    sims_list = [ ]
    for i in range( nr_of_itr ):
        sims = Montecarlo_simulation( S0 , T , dt , model_sim , nr_of_sim , seed )
        u[i] = SMPC_control( T , nr_of_sim , sims , V0 , risk_free_fact , risk_aversion_coef , inv_limit , 'l' )
        
        sims_list.append(sims)

        # Only long for the moment
        if ( u[i] > 0 ):
            inv_nr += 1
    
    perc_inv = ( inv_nr/nr_of_itr )*100
    if ( perc_inv > inv_thd ):
        u_ret = np.round( sum(u)/nr_of_itr , 2 )
    else:
        u_ret = 0.0

    return u_ret , u , sims_list , perc_inv

#--------------------------------------------------------------#
# Expected value of account at the end of the strategy period
#--------------------------------------------------------------#
def account_expected_value( horizon_length: int , nr_of_sim: int , sims: np.ndarray ,
    initial_value: float , risk_free_fact: float , u_invst: float ):

    a1 = 1/nr_of_sim

    V0   = initial_value
    r_f  = risk_free_fact

    V = np.zeros( (nr_of_sim,horizon_length+1) )

    for i in range(nr_of_sim):
        V[i,0] = V0
        for k in range(horizon_length):
            V[i,k+1] = V[i,k]*(1+r_f) + u_invst*( sims[i][k+1]-(1+r_f)*sims[i][k] )

    # Expected value
    V_exp = a1*sum( [ V[i,horizon_length] for i in range(nr_of_sim) ] )

    return np.round( V_exp , 3 )

#--------------------------------------------------------------#
# Expected value of account at the end of the strategy period
#--------------------------------------------------------------#
def EMA_crossing_strategy( EMA_smooth: np.array , EMA_fast: np.array , EMA_slow: np.array ):

    flag_EMA_crossing = np.zeros( ( len(EMA_smooth),1 ) )
    for i in range( 1 , len(EMA_smooth) ):
        if ( EMA_fast[i-1] < EMA_slow[i-1] ) and ( EMA_fast[i] >= EMA_slow[i] ):
            flag_EMA_crossing[i] = 1

    return flag_EMA_crossing