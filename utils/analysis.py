import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from config.constants import NIFTY_200_TICKERS


@st.cache_data(ttl=600)
def get_technical_analysis(ticker_symbol):
    """Perform technical analysis on a given ticker"""
    try:
        data = yf.Ticker(ticker_symbol).history(period="1y", auto_adjust=True)
        if data.empty or len(data) < 200:
            return 0, [], ["Not enough historical data."]

        # Add technical indicators
        data.ta.rsi(append=True)
        data.ta.macd(append=True)
        data.ta.ema(length=50, append=True)
        data.ta.ema(length=200, append=True)
        data.ta.bbands(append=True)
        data.ta.adx(append=True)
        data.ta.supertrend(append=True)

        latest = data.iloc[-1]
        bullish, bearish = [], []

        # RSI Analysis
        if pd.notna(latest.get('RSI_14')) and latest['RSI_14'] < 40:
            bullish.append("RSI has room to grow (< 40)")
        elif pd.notna(latest.get('RSI_14')) and latest['RSI_14'] > 70:
            bearish.append("RSI is Overbought (> 70)")

        # MACD Analysis
        if pd.notna(latest.get('MACD_12_26_9')) and pd.notna(latest.get('MACDs_12_26_9')):
            if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']:
                bullish.append("MACD Line > Signal Line (Bullish Crossover)")
            else:
                bearish.append("MACD Line < Signal Line (Bearish Crossover)")

        # EMA Cross Analysis
        if pd.notna(latest.get('EMA_50')) and pd.notna(latest.get('EMA_200')):
            if latest['EMA_50'] > latest['EMA_200']:
                bullish.append("Golden Cross: 50d EMA > 200d EMA")
            else:
                bearish.append("Death Cross: 50d EMA < 200d EMA")

        # Bollinger Bands Analysis
        if pd.notna(latest.get('BBL_20_2.0')) and pd.notna(latest.get('BBU_20_2.0')):
            if latest['Close'] < latest['BBL_20_2.0']:
                bullish.append("Price Near/Below Lower Bollinger Band")
            elif latest['Close'] > latest['BBU_20_2.0']:
                bearish.append("Price Near/Above Upper Bollinger Band")

        # ADX Analysis
        if pd.notna(latest.get('ADX_14')) and latest['ADX_14'] > 25:
            if latest.get('DMP_14') > latest.get('DMN_14'):
                bullish.append(f"Strong Uptrend (ADX: {latest['ADX_14']:.0f})")
            else:
                bearish.append(f"Strong Downtrend (ADX: {latest['ADX_14']:.0f})")

        # Supertrend Analysis
        if pd.notna(latest.get('SUPERTd_7_3.0')):
            if latest['SUPERTd_7_3.0'] == 1:
                bullish.append("Supertrend is Bullish (Uptrend)")
            else:
                bearish.append("Supertrend is Bearish (Downtrend)")

        score = len(bullish) - len(bearish)
        return score, bullish, bearish
    except Exception:
        return 0, [], ["Error in technical analysis."]


@st.cache_data(ttl=3600)
def get_fundamental_analysis(ticker_symbol):
    """Perform fundamental analysis on a given ticker"""
    try:
        info = yf.Ticker(ticker_symbol).info
        strengths, weaknesses = [], []

        # Financial metrics
        pe = info.get('trailingPE')
        peg = info.get('pegRatio')
        pb = info.get('priceToBook')
        pm = info.get('profitMargins')
        roe = info.get('returnOnEquity')
        de = info.get('debtToEquity')

        # P/E Ratio Analysis
        if pe and pe < 25:
            strengths.append(f"Healthy P/E Ratio ({pe:.2f})")
        elif pe and pe > 60:
            weaknesses.append(f"High P/E Ratio ({pe:.2f})")

        # PEG Ratio Analysis
        if peg and peg < 1.2 and peg > 0:
            strengths.append(f"Good PEG Ratio ({peg:.2f}) - Value for Growth")
        elif peg and peg > 2:
            weaknesses.append(f"High PEG Ratio ({peg:.2f}) - Potentially Overvalued for Growth")

        # P/B Ratio Analysis
        if pb and pb < 3:
            strengths.append(f"Low P/B Ratio ({pb:.2f})")
        elif pb and pb > 7:
            weaknesses.append(f"High P/B Ratio ({pb:.2f})")

        # Profit Margin Analysis
        if pm and pm > 0.10:
            strengths.append(f"High Profit Margin ({pm:.2%})")
        elif pm and pm < 0.05:
            weaknesses.append(f"Low Profit Margin ({pm:.2%})")

        # Return on Equity Analysis
        if roe and roe > 0.15:
            strengths.append(f"Strong Return on Equity ({roe:.2%})")
        elif roe and roe < 0.10:
            weaknesses.append(f"Weak Return on Equity ({roe:.2%})")

        # Debt to Equity Analysis
        if de is not None and de < 100:
            strengths.append(f"Low Debt-to-Equity ({de/100:.2f})")
        elif de is not None and de > 200:
            weaknesses.append(f"High Debt-to-Equity ({de/100:.2f})")

        score = len(strengths) - len(weaknesses)
        return score, strengths, weaknesses
    except Exception:
        return 0, [], ["Could not retrieve fundamental data."]


def generate_signal(tech_score, fund_score, sentiment_score):
    """Generate trading signal based on analysis scores"""
    final_score = (tech_score * 1.5) + fund_score
    
    if sentiment_score > 0.1:
        final_score += 1
    if sentiment_score < -0.1:
        final_score -= 1

    if final_score >= 5:
        return "STRONG BUY"
    elif final_score >= 2:
        return "BUY"
    elif final_score <= -5:
        return "STRONG SELL"
    elif final_score <= -2:
        return "SELL"
    else:
        return "HOLD"


@st.cache_data(ttl=43200)
def screen_stocks(ticker_list_df):
    """Screen stocks from NIFTY 200 for opportunities"""
    all_results = []
    progress_bar = st.progress(0, text="Downloading batch market data for NIFTY 200...")
    
    try:
        all_data = yf.download(
            NIFTY_200_TICKERS, 
            period="1y", 
            auto_adjust=True, 
            progress=False, 
            group_by='ticker'
        )
        if all_data.empty:
            st.error("Failed to download market data for screening.")
            return None, None
    except Exception as e:
        st.error(f"Market data download failed: {e}")
        return None, None

    for i, ticker in enumerate(NIFTY_200_TICKERS):
        progress_bar.progress(
            (i + 1) / len(NIFTY_200_TICKERS), 
            text=f"Scanning {ticker.replace('.NS', '')}..."
        )
        
        try:
            hist = all_data[ticker]
            if hist.empty or len(hist) < 200:
                continue

            # Add technical indicators
            hist.ta.rsi(append=True)
            hist.ta.ema(length=50, append=True)
            hist.ta.ema(length=200, append=True)
            hist.ta.adx(append=True)
            hist.ta.supertrend(append=True)

            latest = hist.iloc[-1]
            info = yf.Ticker(ticker).info

            rise_score, fall_score = 0, 0
            reasons_rise, reasons_fall = [], []

            # Bullish signals
            is_uptrend = (latest.get('EMA_50', 0) > latest.get('EMA_200', 0) and 
                         latest.get('SUPERTd_7_3.0') == 1)
            is_strong_trend = latest.get('ADX_14', 0) > 20
            not_overbought = latest.get('RSI_14', 100) < 70
            good_roe = info.get('returnOnEquity', 0) > 0.15

            if is_uptrend:
                rise_score += 2
                reasons_rise.append("Confirmed Uptrend (EMA & Supertrend)")
            if is_strong_trend:
                rise_score += 1
                reasons_rise.append("Strong Trend Momentum (ADX > 20)")
            if not_overbought:
                rise_score += 1
                reasons_rise.append("RSI Not Overbought")
            if good_roe:
                rise_score += 1
                reasons_rise.append("Strong ROE (>15%)")

            # Bearish signals
            is_downtrend = (latest.get('EMA_50', 0) < latest.get('EMA_200', 0) and 
                           latest.get('SUPERTd_7_3.0') == -1)
            is_overbought = latest.get('RSI_14', 0) > 70
            high_debt = info.get('debtToEquity', 0) > 200

            if is_downtrend:
                fall_score += 2
                reasons_fall.append("Confirmed Downtrend (EMA & Supertrend)")
            if is_overbought:
                fall_score += 1
                reasons_fall.append("Overbought RSI (>70)")
            if high_debt:
                fall_score += 1
                reasons_fall.append("High Debt Burden (D/E > 2)")

            company_name = (ticker_list_df.loc[ticker, 'Security Name'] 
                          if ticker in ticker_list_df.index 
                          else ticker.replace('.NS', ''))
            
            all_results.append({
                'Ticker': ticker,
                'Company Name': company_name,
                'Rise Score': rise_score,
                'Fall Score': fall_score,
                'Reasons Rise': " • ".join(reasons_rise) if reasons_rise else "No strong signals",
                'Reasons Fall': " • ".join(reasons_fall) if reasons_fall else "No strong signals",
                'Current Price': latest.get('Close', 0)
            })
        except Exception:
            continue
            
    progress_bar.empty()
    
    if not all_results:
        return None, None

    results_df = pd.DataFrame(all_results)
    top_rise = (results_df[results_df['Rise Score'] >= 3]
               .sort_values(by='Rise Score', ascending=False)
               .head(5))
    top_fall = (results_df[results_df['Fall Score'] >= 2]
               .sort_values(by='Fall Score', ascending=False)
               .head(5))
    
    return top_rise, top_fall