import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#---------------------------------------------------------------#
# Simple candlestick chart
#---------------------------------------------------------------#
def plot_candlestick_chart( asset: pd.DataFrame , ticker: str ):

    # New figure
    cndlstk_cht = make_subplots( rows = 1, cols = 1)

    # Build the candlestick chart
    cndlstk_cht.append_trace(
            go.Candlestick(
                x     = asset['Date'],
                open  = asset['Adj Open'],
                high  = asset['Adj High'],
                low   = asset['Adj Low'],
                close = asset['Adj Close'],
                increasing_line_color = 'green',
                decreasing_line_color = 'red',
                name       = ticker,
                showlegend = True ) , row = 1, col = 1 
        )
    
    # Layout
    layout = go.Layout(
        plot_bgcolor = '#efefef',
        # Font Families
        font_family = 'Monospace',
        font_color  = '#000000',
        font_size   = 20,
        xaxis = dict(
            rangeslider = dict(
                visible = False
            )
        ),
        width = 1000,
        height = 600,
    )
    # Update options and show plot
    cndlstk_cht.update_layout(layout)

    cndlstk_cht.show( )

#---------------------------------------------------------------#
# Candlestick chart with indicators   
#---------------------------------------------------------------#
def plot_chart_indicators( asset: pd.DataFrame , ticker: str ):

    # New figure
    chart = make_subplots( rows = 2, cols = 1 , shared_xaxes = True )

    # Build the candlestick chart
    chart.append_trace(
            go.Candlestick(
                x     = asset['Date'],
                open  = asset['Adj Open'],
                high  = asset['Adj High'],
                low   = asset['Adj Low'],
                close = asset['Adj Close'],
                increasing_line_color = 'green',
                decreasing_line_color = 'red',
                name       = ticker,
                showlegend = True ) , row = 1, col = 1 
        )

    # EMA fast
    chart.append_trace(
            go.Scatter(
                x           = asset['Date'],
                y           = asset['EMA_fast'],
                line        = dict(color='red', width=2),
                name        = 'EMA_fast',
                showlegend  = True,
            ) , row = 1, col = 1,
        )
    
    # EMA slow
    chart.append_trace(
            go.Scatter(
                x           = asset['Date'],
                y           = asset['EMA_slow'],
                line        = dict(color='green', width=2),
                name        = 'EMA_slow',
                showlegend  = True,
            ) , row = 1, col = 1,
        )
    
    # BB high
    chart.append_trace(
        go.Scatter(
            x           = asset['Date'],
            y           = asset['BB_high'],
            line        = dict(color='blue', width=2),
            name        = 'BB_h',
            showlegend  = True,
        ) , row = 1, col = 1,
    )

    # BB low
    chart.append_trace(
        go.Scatter(
            x           = asset['Date'],
            y           = asset['BB_low'],
            line        = dict(color='blue', width=2),
            name        = 'BB_l',
            showlegend  = True,
        ) , row = 1, col = 1,
    )

    # MACD line
    chart.append_trace(
        go.Scatter(
            x           = asset['Date'],
            y           = asset['MACD'],
            line        = dict(color='green', width=2),
            name        = 'MACD',
            showlegend  = True,
        ), row = 2, col = 1
    )

    # MACD signal
    chart.append_trace(
        go.Scatter(
            x           = asset['Date'],
            y           = asset['MACD_signal'],
            line        = dict(color='red', width=2),
            name        = 'MACD sig.',
            showlegend  = True,
        ), row = 2, col = 1
    )
    
    # MACD histogram
    # Colorize the histogram values
    colors = np.where( asset['MACD_H'] < 0, 'red', 'green' )
    # Plot the histogram
    chart.append_trace(
        go.Bar(
            x            = asset['Date'],
            y            = asset['MACD_H'],
            name         = 'MACD hist.',
            showlegend   = True,
            marker_color = colors,
        ), row = 2, col = 1
    )

    # Layout
    layout = go.Layout(
        plot_bgcolor = '#efefef',
        # Font Families
        font_family = 'Monospace',
        font_color  = '#000000',
        font_size   = 20,
        xaxis = dict(
            rangeslider = dict(
                visible = False
            )
        ),
        width = 1000,
        height = 600,
    )
    # Update options and show plot
    chart.update_layout(layout)

    chart.show( )