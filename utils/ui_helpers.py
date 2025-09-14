import streamlit as st
import textwrap
from .market_data import get_index_data


def load_css():
    """Load custom CSS styling for the application"""
    st.markdown(textwrap.dedent('''
    <style>
        .data-card {
            background: linear-gradient(135deg, #161B22 0%, #1A1F26 100%);
            border: 1px solid #30363D; border-radius: 12px;
            padding: 25px; margin-bottom: 20px; height: 100%;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        h1, h2, h3, h5 { color: #58A6FF; }
        h3 { border-bottom: 2px solid #30363D; padding-bottom: 10px; }
        .index-ticker { 
            display: flex; 
            justify-content: flex-end; 
            align-items: center; 
            gap: 20px; 
            padding-top: 10px; 
        }
        .index-name { font-size: 0.9em; color: #8B949E; }
        .index-value { font-size: 1.1em; font-weight: bold; color: #C9D1D9; }
        .index-change-green { color: #3FB950; font-size: 1.1em; }
        .index-change-red { color: #F85149; font-size: 1.1em; }
        .signal-base { 
            padding: 15px; 
            border-radius: 12px; 
            text-align: center; 
            font-size: 1.5em; 
            font-weight: bold; 
        }
        .signal-buy { 
            background: linear-gradient(135deg, rgba(63, 185, 80, 0.2) 0%, rgba(63, 185, 80, 0.1) 100%); 
            border: 2px solid #3FB950; 
            color: #3FB950; 
        }
        .signal-sell { 
            background: linear-gradient(135deg, rgba(248, 81, 73, 0.2) 0%, rgba(248, 81, 73, 0.1) 100%); 
            border: 2px solid #F85149; 
            color: #F85149; 
        }
        .signal-hold { 
            background: linear-gradient(135deg, rgba(139, 148, 158, 0.2) 0%, rgba(139, 148, 158, 0.1) 100%); 
            border: 2px solid #8B949E; 
            color: #8B949E; 
        }
    </style>
    '''), unsafe_allow_html=True)


def display_index_tickers():
    """Display live index ticker information"""
    index_data = get_index_data()
    if index_data is not None and not index_data.empty:
        st.markdown('<div class="index-ticker">', unsafe_allow_html=True)
        
        for ticker in ['^NSEI', '^BSESN']:
            if ticker in index_data.columns and len(index_data[ticker].dropna()) >= 2:
                name = "NIFTY 50" if ticker == '^NSEI' else "SENSEX"
                latest, prev = index_data[ticker].dropna().iloc[-2:]
                change = latest - prev
                pct_change = (latest - prev) / prev * 100
                color = "green" if change >= 0 else "red"
                
                st.markdown(
                    f'<span class="index-name">{name}</span> '
                    f'<span class="index-value">{latest:,.2f}</span> '
                    f'<span class="index-change-{color}">{change:+.2f} ({pct_change:+.2f}%)</span>', 
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)