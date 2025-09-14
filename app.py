import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import pandas_ta as ta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings
import textwrap
from datetime import datetime
import io
import time
import requests

from utils.data_loader import load_ticker_list
from utils.market_data import (
    get_live_portfolio_data, 
    get_historical_data, 
    get_index_data
)
from utils.news_service import (
    get_general_market_news, 
    get_portfolio_news, 
    get_news_and_sentiment
)
from utils.analysis import (
    get_technical_analysis, 
    get_fundamental_analysis, 
    generate_signal, 
    screen_stocks
)
from utils.ui_helpers import load_css, display_index_tickers
from utils.portfolio_manager import read_log_from_memory, write_log_to_memory
from config.constants import NEWS_API_KEY, NIFTY_200_TICKERS

warnings.filterwarnings('ignore')

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Quantitative Risk & Analysis Dashboard")

# Load CSS styling
load_css()

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {'RELIANCE.NS': 10, 'TCS.NS': 15, 'HDFCBANK.NS': 20}
if 'suggestions_results' not in st.session_state:
    st.session_state.suggestions_results = None
if 'suggestions_log' not in st.session_state:
    st.session_state.suggestions_log = pd.DataFrame()

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "Portfolio", "Analysis", "Suggestions", "Methodology"],
        icons=["speedometer2", "briefcase-fill", "graph-up-arrow", "lightbulb-fill", "book-half"],
        menu_icon="compass-fill", 
        default_index=0
    )
    st.info("⚠️ This tool is for educational purposes and not financial advice.")

# Dashboard Page
if selected == "Dashboard":
    title_col, index_col = st.columns([2, 1.5])
    with title_col:
        st.title("📈 Quantitative Risk Dashboard")
    with index_col:
        display_index_tickers()

    if not st.session_state.portfolio:
        st.warning("Your portfolio is empty. Add assets in the 'Portfolio' page to see your dashboard.")
        st.stop()

    with st.expander("⚙️ Configure Risk Parameters"):
        p_col1, p_col2 = st.columns(2)
        conf_level = p_col1.slider("VaR Confidence Level", 0.90, 0.99, 0.95, 0.01, key="conf_level")
        hist_days = p_col2.number_input("Historical Lookback (Days)", 100, 2000, 500, key="hist_days")

    price_data, returns_data = get_historical_data(list(st.session_state.portfolio.keys()), hist_days)

    if price_data is None or price_data.empty or returns_data is None or returns_data.empty:
        st.error("Could not fetch sufficient market data to perform analysis. Please check tickers or try again later.")
        st.stop()

    latest_timestamp = price_data.index[-1].strftime('%d-%b-%Y')
    st.caption(f"Market data is based on the closing prices of: **{latest_timestamp}**")

    latest_prices = price_data.iloc[-1]
    portfolio_value = np.nansum([st.session_state.portfolio.get(t, 0) * latest_prices.get(t, 0) for t in st.session_state.portfolio])
    weights = pd.Series({t: st.session_state.portfolio.get(t, 0) * latest_prices.get(t, 0) for t in price_data.columns if t in st.session_state.portfolio})

    if weights.sum() > 0:
        weights /= weights.sum()

    portfolio_returns = returns_data[weights.index].dot(weights)

    var_value_percentile = portfolio_returns.quantile(1 - conf_level)
    var_in_currency = portfolio_value * var_value_percentile

    pnl, pnl_pct = 0, 0
    if len(price_data) >= 2:
        yesterday_prices = price_data.iloc[-2]
        yesterday_value = np.nansum([st.session_state.portfolio.get(t, 0) * yesterday_prices.get(t, 0) for t in st.session_state.portfolio])
        if yesterday_value > 0:
            pnl = portfolio_value - yesterday_value
            pnl_pct = (pnl / yesterday_value) * 100

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Risk & Return Summary")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Total Portfolio Value", value=f"₹{portfolio_value:,.2f}")
    col2.metric(
        label=f"{int(conf_level*100)}% Historical VaR (1-Day)", 
        value=f"₹{-var_in_currency:,.2f}", 
        help="Historical Value at Risk (VaR) estimates the maximum potential loss based on the worst-case daily returns from your chosen historical period."
    )
    col3.metric(label="Today's Profit / Loss", value=f"₹{pnl:,.2f}", delta=f"{pnl_pct:.2f}% vs Yesterday")

    st.divider()
    
    st.subheader("Portfolio Holdings")
    holdings = [
        {
            "Ticker": t.replace('.NS', ''), 
            "Shares": s, 
            "Price": latest_prices.get(t, 0), 
            "Value": s * latest_prices.get(t, 0)
        } 
        for t, s in st.session_state.portfolio.items()
    ]
    if holdings:
        holdings_df = pd.DataFrame(holdings)
        holdings_df['Allocation'] = (holdings_df['Value'] / portfolio_value) * 100 if portfolio_value > 0 else 0
        st.dataframe(
            holdings_df, 
            hide_index=True, 
            use_container_width=True, 
            column_config={
                "Price": st.column_config.NumberColumn(format="₹%.2f"), 
                "Value": st.column_config.NumberColumn(format="₹%.2f"), 
                "Allocation": st.column_config.ProgressColumn("Allocation", format="%.2f%%", min_value=0, max_value=100)
            }
        )

# Portfolio Page
elif selected == "Portfolio":
    st.title("💼 Portfolio Manager")
    ticker_list = load_ticker_list()
    if ticker_list is None:
        st.error("Could not load stock list. Portfolio manager is unavailable.")
        st.stop()

    st.subheader("Manage Portfolio")
    with st.form("manage_portfolio_form"):
        col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
        selected_label = col1.selectbox(
            "Stock", 
            options=ticker_list['label'].tolist(), 
            index=None, 
            placeholder="Type to search... (e.g., Reliance Industries)"
        )
        shares_input = col2.number_input("Quantity", min_value=1, value=10)
        
        with col3:
            st.write(""); st.write("")
            add_submitted = st.form_submit_button("Add", use_container_width=True, type="primary")
        with col4:
            st.write(""); st.write("")
            remove_submitted = st.form_submit_button("Remove", use_container_width=True)
            
        if add_submitted and selected_label:
            ticker = ticker_list[ticker_list['label'] == selected_label].index[0]
            st.session_state.portfolio[ticker] = st.session_state.portfolio.get(ticker, 0) + int(shares_input)
            st.toast(f"✅ Added {shares_input} shares of {ticker.replace('.NS', '')}!", icon="🎉")
            time.sleep(1)
            st.rerun()
            
        if remove_submitted and selected_label:
            ticker = ticker_list[ticker_list['label'] == selected_label].index[0]
            current_shares = st.session_state.portfolio.get(ticker)
            if current_shares is None:
                st.error(f"You do not own any shares of {ticker.replace('.NS', '')}.")
            elif shares_input >= current_shares:
                del st.session_state.portfolio[ticker]
                st.toast(f"✅ Removed all shares of {ticker.replace('.NS', '')}.", icon="🗑️")
                time.sleep(1)
                st.rerun()
            else:
                st.session_state.portfolio[ticker] -= int(shares_input)
                st.toast(f"✅ Removed {shares_input} shares of {ticker.replace('.NS', '')}.", icon="➖")
                time.sleep(1)
                st.rerun()

    st.divider()
    st.subheader("📋 Current Holdings")
    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Use the form above to add stocks.")
    else:
        tickers_in_portfolio = list(st.session_state.portfolio.keys())
        live_data = get_live_portfolio_data(tickers_in_portfolio)
        portfolio_items = []
        
        for ticker, shares in st.session_state.portfolio.items():
            company_name = ticker_list.loc[ticker, 'Security Name'] if ticker in ticker_list.index else "Unknown"
            current_price = live_data.loc[ticker, 'Current Price'] if not live_data.empty and ticker in live_data.index else 0
            day_change = live_data.loc[ticker, 'Day Change %'] if not live_data.empty and ticker in live_data.index else 0
            portfolio_items.append({
                'Ticker': ticker.replace('.NS', ''), 
                'Company Name': company_name, 
                'Shares': shares, 
                'Current Price': current_price, 
                'Total Value': shares * current_price, 
                "Day's Change %": day_change
            })
            
        portfolio_df = pd.DataFrame(portfolio_items)
        st.dataframe(
            portfolio_df, 
            use_container_width=True, 
            hide_index=True, 
            column_config={
                "Company Name": st.column_config.TextColumn("Company", width="large"), 
                "Current Price": st.column_config.NumberColumn("Price", format="₹%.2f"), 
                "Total Value": st.column_config.NumberColumn("Value", format="₹%.2f"), 
                "Day's Change %": st.column_config.NumberColumn("Day Change", format="%.2f%%")
            }
        )

    st.markdown("---")
    st.subheader("📰 Portfolio & Market News")
    news_tab1, news_tab2 = st.tabs(["My Portfolio News", "General Market News"])
    
    with news_tab1:
        if not st.session_state.portfolio:
            st.info("Add stocks to your portfolio to see related news here.")
        else:
            portfolio_news_df = get_portfolio_news(list(st.session_state.portfolio.keys()))
            if not portfolio_news_df.empty:
                md_table = "| Date | Ticker | Article | Source |\n|---|---|---|---|\n"
                for _, row in portfolio_news_df.head(20).iterrows():
                    md_table += f"| {row['Date']} | {row['ticker']} | [{row['Title']}]({row['Link']}) | {row['Source']} |\n"
                st.markdown(md_table, unsafe_allow_html=True)
            else:
                st.info("No recent news found for your portfolio stocks.")
                
    with news_tab2:
        market_news_df = get_general_market_news()
        if not market_news_df.empty:
            md_table = "| Date | Article | Source |\n|---|---|---|\n"
            for _, row in market_news_df.head(20).iterrows():
                md_table += f"| {row['Date']} | [{row['Title']}]({row['Link']}) | {row['Source']} |\n"
            st.markdown(md_table, unsafe_allow_html=True)
        else:
            st.info("Could not fetch general market news.")

# Analysis Page
elif selected == "Analysis":
    st.title("🔬 Comprehensive Stock Analysis")
    ticker_list = load_ticker_list()
    if ticker_list is None:
        st.error("Could not load the stock list. The analysis page is unavailable.")
        st.stop()

    label = st.selectbox(
        "Select a Stock for a Detailed Report", 
        ticker_list['label'].tolist(), 
        index=None, 
        placeholder="Search by name or ticker..."
    )
    
    if label:
        ticker = ticker_list[ticker_list['label'] == label].index[0]
        company_name = ticker_list.loc[ticker, 'Security Name']
        
        with st.spinner(f'Performing comprehensive analysis for **{ticker.replace(".NS", "")}**...'):
            tech_score, bullish, bearish = get_technical_analysis(ticker)
            fund_score, strengths, weaknesses = get_fundamental_analysis(ticker)
            news_df, sentiment, sentiment_text = get_news_and_sentiment(ticker, company_name)
            signal = generate_signal(tech_score, fund_score, sentiment)

        st.header(f"Analysis Report: {label}")
        signal_color_map = {"BUY": "buy", "STRONG BUY": "buy", "SELL": "sell", "STRONG SELL": "sell", "HOLD": "hold"}
        signal_color = signal_color_map.get(signal, "hold")
        st.markdown(
            f'<div class="signal-base signal-{signal_color}" style="margin-bottom:20px;">Overall Signal: {signal}</div>', 
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Technical Analysis (Score: {tech_score})")
            with st.container(border=True):
                if not bullish and not bearish:
                    st.write("No strong technical signals detected.")
                else:
                    for point in bullish:
                        st.markdown(f"<p style='color: #3FB950; margin: 0;'>+ {point}</p>", unsafe_allow_html=True)
                    for point in bearish:
                        st.markdown(f"<p style='color: #F85149; margin: 0;'>- {point}</p>", unsafe_allow_html=True)
                        
        with col2:
            st.subheader(f"Fundamental Analysis (Score: {fund_score})")
            with st.container(border=True):
                if not strengths and not weaknesses:
                    st.write("No strong fundamental signals detected.")
                else:
                    for strength in strengths:
                        st.markdown(f"<p style='color: #3FB950; margin: 0;'>+ {strength}</p>", unsafe_allow_html=True)
                    for weakness in weaknesses:
                        st.markdown(f"<p style='color: #F85149; margin: 0;'>- {weakness}</p>", unsafe_allow_html=True)

        with st.expander("How to Read This Analysis 🤔", expanded=True):
            st.markdown(textwrap.dedent('''
            This report generates a signal by looking at the stock from two main angles:

            #### 1. Technical Analysis (Chart Analysis)
            This is like being a detective for stock charts. We analyze patterns in the stock's price history to gauge its current momentum and potential future direction. A high score suggests the chart patterns are favorable.

            - **How the Score is Calculated**: We check 6 key technical indicators. The final score is simply **(Number of Bullish Signals) - (Number of Bearish Signals)**, which can range from **-6 to +6**.
                - **Bullish (+1 point)**: A signal suggesting the price may go up (e.g., RSI is low, a 'Golden Cross' occurs).
                - **Bearish (-1 point)**: A signal suggesting the price may go down (e.g., RSI is too high, a 'Death Cross' occurs).

            #### 2. Fundamental Analysis (Company's Health)
            This is like checking the company's annual report card. We look at its financial health—like profits, sales, and debt—to determine if it is a strong, well-managed company. A high score means the company's finances look solid.

            - **How the Score is Calculated**: We check 6 key financial metrics. The score is **(Number of Strengths) - (Number of Weaknesses)**, ranging from **-6 to +6**.
                - **Strength (+1 point)**: A good financial sign (e.g., low P/E ratio, high profit margin).
                - **Weakness (-1 point)**: A potential red flag (e.g., very high debt, low return on equity).

            ---
            The **Overall Signal** combines these two scores and also considers the latest news sentiment (positive, negative, or neutral) to provide a final, actionable recommendation.
            '''))

    st.markdown("---")
st.subheader("📊 Portfolio Batch Analysis")
if not st.session_state.portfolio:
    st.warning("Your portfolio is empty. Add stocks using the Portfolio page to enable this feature.")
elif st.button("🚀 Analyze All Portfolio Stocks", use_container_width=True, type="primary"):
    results = []
    tickers_to_analyze = list(st.session_state.portfolio.keys())
    progress_bar = st.progress(0, text="Downloading batch historical data...")

    all_hist_data = yf.download(
        tickers_to_analyze,
        period="1y",
        auto_adjust=True,
        progress=False,
        group_by='ticker'
    )

    total_stocks = len(tickers_to_analyze)
    for i, ticker in enumerate(tickers_to_analyze):
        progress_bar.progress((i + 1) / total_stocks, text=f"Analyzing {ticker.replace('.NS','')}...")
        company_name = ticker_list.loc[ticker, 'Security Name'] if ticker in ticker_list.index else ticker
        
        tech_score, _, _ = get_technical_analysis(ticker)
        fund_score, _, _ = get_fundamental_analysis(ticker)
        _, sentiment, _ = get_news_and_sentiment(ticker, company_name)
        signal = generate_signal(tech_score, fund_score, sentiment)
        
        yoy_change = 0.0
        try:
            hist_data = all_hist_data[ticker]
            if not hist_data.empty and len(hist_data) > 1:
                start_price = hist_data['Close'].iloc[0]
                latest_price = hist_data['Close'].iloc[-1]
                if start_price > 0:
                    yoy_change = ((latest_price - start_price) / start_price) * 100
        except (KeyError, IndexError):
            yoy_change = 0.0

        results.append({
            'Ticker': ticker.replace('.NS',''),
            'Signal': signal,
            'Tech Score': tech_score,
            'Fund Score': fund_score,
            '1Y Change %': f"{yoy_change:+.2f}%"
        })
    progress_bar.empty()
    st.session_state['batch_results'] = pd.DataFrame(results)

if 'batch_results' in st.session_state:
    results_df = st.session_state['batch_results'].copy()
    
    # Apply styling function for better mobile display
    def style_dataframe(df):
        def color_score(val):
            """Color code the scores"""
            try:
                if isinstance(val, str) and '%' in val:
                    # Handle percentage values
                    num_val = float(val.replace('%', '').replace('+', ''))
                    color = '#28a745' if num_val > 0 else '#dc3545' if num_val < 0 else '#6c757d'
                elif isinstance(val, (int, float)):
                    # Handle numeric scores
                    color = '#28a745' if val > 0 else '#dc3545' if val < 0 else '#6c757d'
                else:
                    return ''
                return f'color: {color}; font-weight: bold'
            except:
                return ''
        
        def color_signal(val):
            """Color code the signals"""
            if 'STRONG BUY' in str(val):
                return 'color: white; font-weight: bold; padding: 4px; border-radius: 4px'
            elif 'BUY' in str(val):
                return 'color: white; font-weight: bold; padding: 4px; border-radius: 4px'
            elif 'HOLD' in str(val):
                return 'color: white; font-weight: bold; padding: 4px; border-radius: 4px'
            elif 'SELL' in str(val):
                return 'color: white; font-weight: bold; padding: 4px; border-radius: 4px'
            else:
                return 'color: white; font-weight: bold; padding: 4px; border-radius: 4px'
        
        # Apply styles
        styled_df = df.style.applymap(color_score, subset=['Tech Score', 'Fund Score', '1Y Change %'])
        styled_df = styled_df.applymap(color_signal, subset=['Signal'])
        
        return styled_df

    # Display the styled dataframe
    st.dataframe(
        style_dataframe(results_df),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("📈 Ticker", width="small"),
            "Signal": st.column_config.TextColumn("🎯 Overall Signal", width="medium"),
            "Tech Score": st.column_config.TextColumn("📊 Technical Score", width="small"),
            "Fund Score": st.column_config.TextColumn("💰 Fundamental Score", width="small"),
            "1Y Change %": st.column_config.TextColumn("📈 1Y Change %", width="medium")
        }
    )

    with st.expander("How to Read This Analysis 🤔"):
        st.markdown(textwrap.dedent('''
        This table gives you a quick summary of each stock in your portfolio:

        - **Technical Score**: Looks at chart patterns. A positive score is green, negative is red. Calculated from 6 indicators, so the score is between **-6 and +6**.
        - **Fundamental Score**: Checks the company's financial health. A positive score is green, negative is red. Calculated from 6 metrics, so the score is between **-6 and +6**.
        - **1Y Change %**: Shows the stock's percentage price change over the last year.
        - **Overall Signal**: The final recommendation based on all of the above.
        📱 **Mobile Tip**: You can scroll horizontally on the table if needed, and tap on rows for better visibility.
        '''))

# Suggestions Page
elif selected == "Suggestions":
    st.title("💡 Market Opportunities Scanner")
    st.warning("⚠️ **Disclaimer:** This is an experimental feature for informational purposes only. These suggestions are **not** financial advice.")

    ticker_list = load_ticker_list()
    if ticker_list is None:
        st.error("Could not load the stock list.")
        st.stop()

    if st.button("🔍 Scan NIFTY 200 for Opportunities", use_container_width=True, type="primary"):
        with st.spinner("🔬 Analyzing NIFTY 200 stocks... This may take a few minutes."):
            st.session_state.suggestions_results = screen_stocks(ticker_list)
        if st.session_state.suggestions_results is not None:
            top_rise, top_fall = st.session_state.suggestions_results
            log_data = []
            if top_rise is not None and not top_rise.empty:
                log_rise = top_rise.copy()
                log_rise['Suggestion'] = 'Rise'
                log_data.append(log_rise)
            if top_fall is not None and not top_fall.empty:
                log_fall = top_fall.copy()
                log_fall['Suggestion'] = 'Fall'
                log_data.append(log_fall)

            if log_data:
                suggestions_to_log = pd.concat(log_data)
                suggestions_to_log.rename(columns={'Current Price': 'Initial Price'}, inplace=True)
                suggestions_to_log['Suggestion Date'] = pd.to_datetime(datetime.utcnow())
                final_log = suggestions_to_log[['Suggestion Date', 'Ticker', 'Suggestion', 'Initial Price']]
                if write_log_to_memory(final_log):
                    st.success("✅ Suggestions saved to Performance Tracker!")
        st.rerun()

    if st.session_state.suggestions_results:
        top_rise, top_fall = st.session_state.suggestions_results
        st.divider()
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("📈 Top Potential Risers")
            if top_rise is not None and not top_rise.empty:
                for _, row in top_rise.iterrows():
                    with st.container(border=True):
                        st.markdown(
                            f"**{row['Ticker'].replace('.NS', '')}** - <span style='color: #8B949E;'>{row['Company Name']}</span>", 
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f"<span style='color: #3FB950;'>Rise Score: {row['Rise Score']}/5</span> | Price: ₹{row['Current Price']:.2f}", 
                            unsafe_allow_html=True
                        )
                        st.caption(f"Reasons: {row['Reasons Rise']}")
            else:
                st.info("No strong bullish candidates found based on the current criteria.")
                
        with col2:
            st.subheader("📉 Top Potential Decliners")
            if top_fall is not None and not top_fall.empty:
                for _, row in top_fall.iterrows():
                    with st.container(border=True):
                        st.markdown(
                            f"**{row['Ticker'].replace('.NS', '')}** - <span style='color: #8B949E;'>{row['Company Name']}</span>", 
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            f"<span style='color: #F85149;'>Fall Score: {row['Fall Score']}/4</span> | Price: ₹{row['Current Price']:.2f}", 
                            unsafe_allow_html=True
                        )
                        st.caption(f"Reasons: {row['Reasons Fall']}")
            else:
                st.info("No strong bearish signals detected based on the current criteria.")

# Methodology Page
elif selected == "Methodology":
    st.title("🧠 Methodology Explained")
    with st.expander("💡 Signal Generation Algorithm", expanded=True):
        st.info("The final signal is created by combining all analysis scores into a single number.")
        st.markdown("A higher weight is given to **Technical Analysis** to reflect its importance in market momentum. The formula used is:")
        st.latex(r'''
        \text{Final Score} = (\text{Technical Score} \times 1.5) + \text{Fundamental Score} + \text{Sentiment Adjustment}
        ''')
        st.markdown("""
        - **Sentiment Adjustment**: 
            - **+1** if overall news sentiment is Positive.
            - **-1** if overall news sentiment is Negative.
            - **0** if overall news sentiment is Neutral.
        
        This **Final Score** is then mapped to a recommendation like *Strong Buy* (if score ≥ 5) or *Sell* (if score ≤ -2).
        """)
    with st.expander("⚖️ Value at Risk (VaR) - Historical Method"):
        st.markdown("""
        **Historical Simulation VaR** is a straightforward method to estimate risk. It doesn't use complex statistical assumptions, but instead answers a simple question: 
        
        > *"Based on the daily returns of the last X days, what was the most I could have lost in a single day, 95% of the time?"*

        Think of it like a **financial weather forecast**. It looks at past storms (bad days) to estimate the worst-case scenario for tomorrow, helping you prepare for potential losses.
        """)
    with st.expander("📊 Technical Analysis Indicators"):
        st.markdown('''
        This analysis focuses on **chart patterns and market momentum** to predict short-to-medium term price movements. We use indicators popular in the Indian market:
        - **Supertrend**: A trend-following indicator used to identify clear uptrends and downtrends.
        - **ADX (Average Directional Index)**: Measures the *strength* of a trend, helping to filter out weak or sideways markets.
        - **RSI & MACD**: Standard momentum indicators for identifying overbought/oversold conditions and trend reversals.
        - **EMA Cross (50/200)**: The 'Golden Cross' (bullish) and 'Death Cross' (bearish) are used to signal major long-term trend changes.
        ''')
    with st.expander("🏦 Fundamental Analysis Metrics"):
        st.markdown('''
        This analysis assesses the **financial health and intrinsic value** of a company. The metrics are adjusted for the Indian market, which often features high-growth companies:
        - **P/E Ratio**: Uses a more lenient threshold to avoid incorrectly flagging growth stocks as overvalued.
        - **PEG Ratio**: Balances P/E with earnings growth, offering better insight into the valuation of growth stocks.
        - **Profit Margin & ROE**: Standard metrics to assess a company's profitability and efficiency.
        - **Debt-to-Equity**: Assesses financial leverage and risk.
        ''')
    with st.expander("📰 AI-Powered News Sentiment"):
        st.markdown("""
        **FinBERT Integration**: This dashboard uses a specialized AI model trained on financial text (**FinBERT**) to analyze the sentiment of recent news headlines.
        
        It classifies headlines as **Positive**, **Negative**, or **Neutral**, providing a real-time pulse on market perception that influences the final trading signal. News is sourced via a dedicated News API for reliability.
        """)