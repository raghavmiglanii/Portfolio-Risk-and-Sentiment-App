import streamlit as st
import pandas as pd
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config.constants import NEWS_API_KEY


@st.cache_resource
def load_sentiment_model():
    """Load the FinBERT sentiment analysis model"""
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model


@st.cache_data(ttl=1800)
def get_general_market_news():
    """Fetch general Indian market news using NewsAPI"""
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWS_API_KEY_HERE":
        st.error("Please add your NewsAPI key to the script to fetch news.")
        return pd.DataFrame()
    
    url = (f"https://newsapi.org/v2/everything?"
           f"q=India business finance&"
           f"language=en&"
           f"sortBy=publishedAt&"
           f"apiKey={NEWS_API_KEY}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') == 'ok' and data.get('articles'):
            articles = data['articles']
            df = pd.DataFrame(articles)
            df = df[['title', 'source', 'url', 'publishedAt']]
            df['Source'] = df['source'].apply(lambda s: s['name'])
            df.rename(columns={
                'title': 'Title', 
                'url': 'Link', 
                'publishedAt': 'Timestamp'
            }, inplace=True)
            df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
            return df[['Date', 'Title', 'Link', 'Source']].drop_duplicates(
                subset=['Title']
            ).reset_index(drop=True)
        else:
            st.warning(f"Could not fetch general news. Reason: {data.get('message', 'No articles found')}")
            return pd.DataFrame()
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching general news: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while processing general news: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def get_portfolio_news(portfolio_tickers):
    """Fetch news for specific stocks using NewsAPI"""
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWS_API_KEY_HERE":
        return pd.DataFrame()
    if not portfolio_tickers:
        return pd.DataFrame()

    all_articles = []
    for ticker in portfolio_tickers:
        query = ticker.replace('.NS', '')
        url = (f"https://newsapi.org/v2/everything?"
               f"q={query}&"
               f"language=en&"
               f"searchIn=title,description&"
               f"sortBy=publishedAt&"
               f"pageSize=5&"
               f"apiKey={NEWS_API_KEY}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if data.get('status') == 'ok':
                for article in data.get('articles', []):
                    article['ticker'] = query
                    all_articles.append(article)
        except (requests.exceptions.RequestException, KeyError):
            continue
            
    if not all_articles:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_articles)
    df['Source'] = df['source'].apply(lambda s: s['name'])
    df.rename(columns={
        'title': 'Title', 
        'url': 'Link', 
        'publishedAt': 'Timestamp'
    }, inplace=True)
    df['Date'] = pd.to_datetime(df['Timestamp']).dt.date
    df = df.sort_values(by='Timestamp', ascending=False)
    return df[['Date', 'ticker', 'Title', 'Link', 'Source']].drop_duplicates(
        subset=['Title']
    ).reset_index(drop=True)


@st.cache_data(ttl=3600)
def get_news_and_sentiment(ticker_symbol, company_name):
    """Get news and sentiment analysis for a specific ticker"""
    try:
        news_df = get_portfolio_news([ticker_symbol])
        if news_df.empty:
            return pd.DataFrame(), 0, "No recent news found."

        headlines = news_df['Title'].tolist()
        tokenizer, model = load_sentiment_model()

        inputs = tokenizer(
            headlines, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        with torch.no_grad():
            logits = model(**inputs).logits
        scores = torch.nn.functional.softmax(logits, dim=1)

        if not news_df.empty:
            news_df['Positive'] = scores[:, 0].tolist()
            news_df['Negative'] = scores[:, 1].tolist()
            news_df['Neutral'] = scores[:, 2].tolist()

            avg_sentiment = (news_df['Positive'] - news_df['Negative']).mean()
            if avg_sentiment > 0.1:
                summary = "Positive"
            elif avg_sentiment < -0.1:
                summary = "Negative"
            else:
                summary = "Neutral"
        else:
            avg_sentiment = 0
            summary = "Neutral"

        return news_df, avg_sentiment, summary
    except Exception as e:
        return pd.DataFrame(), 0, f"News analysis error: {e}"