import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf


@st.cache_data(ttl=300)
def get_live_portfolio_data(tickers):
    """Get live portfolio data for given tickers"""
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(tickers, period="2d", auto_adjust=True, progress=False)
        if data.empty or len(data) < 2:
            return pd.DataFrame()
            
        if isinstance(tickers, str) or len(tickers) == 1:
            ticker = tickers[0] if isinstance(tickers, list) else tickers
            close_prices = data[['Close']].rename(columns={'Close': ticker})
        else:
            close_prices = data['Close']
            
        close_prices = close_prices.dropna(axis=1, how='any')
        if close_prices.empty:
            return pd.DataFrame()
            
        last_prices = close_prices.iloc[-1]
        prev_close = close_prices.iloc[-2]
        
        live_data = pd.DataFrame({
            'Current Price': last_prices,
            'Previous Close': prev_close
        })
        
        live_data['Day Change Price'] = live_data['Current Price'] - live_data['Previous Close']
        live_data['Day Change %'] = (live_data['Day Change Price'] / live_data['Previous Close']) * 100
        
        return live_data.fillna(0)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_historical_data(tickers, period_days):
    """Get historical data for given tickers and period"""
    if not tickers:
        return None, None
    try:
        data = yf.download(
            tickers, 
            period=f"{period_days}d", 
            interval="1d", 
            auto_adjust=True, 
            progress=False
        )
        
        if data.empty:
            return None, None
            
        if isinstance(tickers, str) or len(tickers) == 1:
            ticker = tickers[0] if isinstance(tickers, list) else tickers
            price_data = data[['Close']].rename(columns={'Close': ticker})
        else:
            price_data = data['Close']
            
        price_data = price_data.dropna(axis=1, how='all')
        if price_data.empty:
            return None, None
            
        log_returns = np.log(price_data / price_data.shift(1)).dropna()
        return price_data, log_returns
    except Exception:
        return None, None


@st.cache_data(ttl=300)
def get_index_data():
    """Get index data for NIFTY 50 and SENSEX"""
    try:
        return yf.download(
            tickers=['^NSEI', '^BSESN'], 
            period='5d', 
            auto_adjust=True, 
            progress=False
        )['Close']
    except Exception:
        return None