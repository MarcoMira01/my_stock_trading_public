import yfinance as yf_database
import pandas as pd
import datetime as datetime
import copy
import numpy as np

#--------------------------------------------------------------#
# Download samples and compute adjusted values
#--------------------------------------------------------------#
def initialize_data( ticker ):

    # ticker = ticker name of the chosen asset

    #--------------------------------------------------------------#
    # Download the selected assets info
    #--------------------------------------------------------------#
    asset = yf_database.download( ticker )
    
    #--------------------------------------------------------------#
    # Create new DataFrame with additional columns:
    #  Date | Open | Close | High | Low | Volume | Adj Open | Adj Close | Adj High | Adj Low 
    #--------------------------------------------------------------#
    l = list( asset.columns )
    l.insert( 0 , 'Date' )
    l.extend( [ 'Adj Open' , 'Adj High' , 'Adj Low' ] )

    asset_ext = pd.DataFrame( index = list(range(len(asset))) , columns = l )

    #--------------------------------------------------------------#
    # Copy dates
    #--------------------------------------------------------------#
    d = asset.index.date
    asset_ext['Date'] = list(d)

    #--------------------------------------------------------------#
    # Copy existing columns
    #--------------------------------------------------------------#
    c = list( asset.columns )
    asset_ext[c] = copy.copy( asset[c].values )

    #--------------------------------------------------------------#
    # Compute adjusted prices factor
    #--------------------------------------------------------------#
    k_adj = asset_ext['Adj Close']/asset_ext['Close']
    k_adj = k_adj.values[0:]

    #--------------------------------------------------------------#
    # Compute the other adjusted prices
    #--------------------------------------------------------------#
    asset_ext['Adj Open'] = k_adj*asset_ext['Open']
    asset_ext['Adj High'] = k_adj*asset_ext['High']
    asset_ext['Adj Low']  = k_adj*asset_ext['Low']

    return np.round(asset_ext,3)

#--------------------------------------------------------------#
# Find asset corresponding to date
#--------------------------------------------------------------#
def asset_search_for_date( asset: pd.DataFrame , date_to_find: datetime.date ):

    exit_flag = False
    idx       = -1
    while ( exit_flag == False ) and ( idx < len(asset) ):
        idx = idx+1
        if ( asset['Date'].values[idx] >= date_to_find ):
            idx_date = idx
            exit_flag = True
        
    if ( exit_flag == False ):
        raise Exception('The date you are looking for does not exist in the database!')
    else:
        return idx_date

#--------------------------------------------------------------#
# Select asset samples between two dates
#--------------------------------------------------------------#
def reduce_asset( asset: pd.DataFrame , start_date: datetime.date , end_date: datetime.date ):

    idx_start = asset_search_for_date( asset , start_date )
    idx_end   = asset_search_for_date( asset , end_date )
    asset_red = pd.DataFrame(asset.iloc[idx_start:idx_end+1])
    asset_red.index = asset_red.index-idx_start

    return asset_red

#--------------------------------------------------------------#
# Trailing Stop-Loss (SL) indicator
#--------------------------------------------------------------#
def Trailing_SL_Indicator( data: pd.Series , ATR_indicator: pd.Series , multiplier: float ):

    # chosen_value: Close, Open, Adj Close, Adj Open
    # len(data) must be = len(ATR_indicator)

    # Initilization
    temp        = np.empty( ( len(data) , 1 ) )
    temp[:]     = np.nan
    Trailing_SL = [ ]
    for idx in range(len(temp)):
        Trailing_SL.append( temp[idx].item() )

    SL = multiplier*ATR_indicator[13]
    Trailing_SL[13] = data[13]-SL

    for idx in range(14,len(data)):

        SL = multiplier*ATR_indicator[idx]
        # print(idx)
        if ( data[idx] > Trailing_SL[idx - 1] ) and ( data[idx - 1] > Trailing_SL[idx - 1] ):
            # print('a')
            Trailing_SL[idx] = max(Trailing_SL[idx - 1], data[idx] - SL)

        elif ( data[idx] < Trailing_SL[idx - 1] ) and ( data[idx - 1] < Trailing_SL[idx - 1] ):
            # print('b')
            Trailing_SL[idx] = min(Trailing_SL[idx - 1], data[idx] + SL)
            
        elif ( data[idx] > Trailing_SL[idx - 1] ) and ( data[idx - 1] < Trailing_SL[idx - 1] ):
            # print('c')
            Trailing_SL[idx] = data[idx] - SL
            
        elif ( data[idx] < Trailing_SL[idx - 1] ) and ( data[idx - 1] > Trailing_SL[idx - 1] ):
            # print('d')
            Trailing_SL[idx] = data[idx] + SL

    return np.round(Trailing_SL,3)