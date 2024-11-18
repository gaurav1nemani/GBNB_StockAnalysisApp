import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
import yfinance as yf
from stocknews import StockNews

# Set up page configuration
st.set_page_config(layout="wide")
st.title('Get Bulls Not Bears')

# Sidebar inputs
st.sidebar.header('Enter the stock:')
ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
ticker = st.sidebar.selectbox("Ticker", ticker_list)

start_date = st.sidebar.date_input(
    'Start Date', value=datetime.today() - timedelta(days=365)
)
end_date = st.sidebar.date_input('End Date')

# Download stock data
if start_date < end_date:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    csv = stock_data.to_csv(index=True)

    # Sidebar buttons
    if st.sidebar.button("Update Data"):
        st.success("Data Updated Successfully!")
    st.sidebar.download_button(
        "Download Data as CSV",
        data=csv,
        file_name=f'Stockdata_{datetime.now().date()}.csv'
    )
else:
    st.error("Start date must be earlier than end date.")

# Menu for navigation
menu = option_menu(
    menu_title=None,
    options=["Summary", "Chart", "Financials", "Monte Carlo Simulation", "News", "Portfolio Management"],
    icons=["journal-text", "graph-up-arrow", "cash-coin", "bezier2", "news"],
    default_index=0,
    orientation="horizontal"
)

# Menu logic
if menu == "Summary":
    st.header("Company Information")
    ticker_info = yf.Ticker(ticker).info
    st.write(ticker_info.get('longBusinessSummary', 'No summary available.'))

    # Display Key Statistics
    st.subheader("Key Statistics")
    key_statistics = {
        'Market Cap': ticker_info.get('marketCap'),
        'Trailing P/E': ticker_info.get('trailingPE'),
        'Forward P/E': ticker_info.get('forwardPE'),
        'Dividend Yield': ticker_info.get('dividendYield'),
        'Revenue': ticker_info.get('totalRevenue'),
        'EBITDA': ticker_info.get('ebitda'),
    }
    stats_df = pd.DataFrame(list(key_statistics.items()), columns=["Metric", "Value"])
    st.table(stats_df)

    # Plot Adjusted Close and Volume
    if not stock_data.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=stock_data.index, y=stock_data['Adj Close'], name='Adjusted Close'),
            secondary_y=True
        )
        fig.add_trace(
            go.Bar(x=stock_data.index, y=stock_data['Volume'], name='Volume'),
            secondary_y=False
        )
        fig.update_layout(title="Stock Data Overview", template="plotly_white")
        st.plotly_chart(fig)

elif menu == "Chart":
    chart_type = st.selectbox("Select Chart Type", ["Line Chart", "Candlestick Chart"])
    if chart_type == "Line Chart":
        fig = px.line(
            stock_data,
            x=stock_data.index,
            y='Adj Close',
            title="Line Chart - Adjusted Close"
        )
        st.plotly_chart(fig)
    elif chart_type == "Candlestick Chart":
        fig = go.Figure(data=[
            go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Adj Close']
            )
        ])
        st.plotly_chart(fig)

elif menu == "Financials":
    st.header("Financial Statements")
    # Placeholder for financials logic
    st.warning("Financial data functionality is under development.")

elif menu == "Monte Carlo Simulation":
    st.header("Monte Carlo Simulation")
    if not stock_data.empty:
        random_seed = st.number_input("Random Seed", min_value=1, max_value=1000, value=42)
        time_horizon = st.slider("Time Horizon (days)", min_value=30, max_value=365, step=30)
        simulations = st.slider("Number of Simulations", min_value=100, max_value=1000, step=100)

        np.random.seed(random_seed)
        close_price = stock_data['Adj Close']
        daily_return = close_price.pct_change().dropna()
        daily_volatility = np.std(daily_return)

        simulation_df = pd.DataFrame()
        for _ in range(simulations):
            prices = [close_price[-1]]
            for _ in range(time_horizon):
                future_price = prices[-1] * (1 + np.random.normal(0, daily_volatility))
                prices.append(future_price)
            simulation_df[_] = prices

        st.line_chart(simulation_df)
    else:
        st.error("No stock data available for simulation.")


elif menu=="News":
    st.header(f'News of {ticker}')
    sn=StockNews(ticker, save_news=False)
    news_data=sn.read_rss()
    for i in range(20):
        st.subheader(f'News Article {i+1}')
        st.write(news_data['published'][i])
        st.write(news_data['title'][i])
        st.write(news_data['summary'][i])
        sentiment_title_data=(news_data['sentiment_title'][i])
        st.write(f'Title Sentiment{sentiment_title_data}')
        sentiment_data=news_data['sentiment_summary'][i]
        st.write(f'News Sentiment{sentiment_data}')

elif menu=="Portfolio Management":
    selected_stock=st.multiselect('Select the stocks to the portfolio: (recommended max. 10)', ticker_list)
    
    stocks_industries_list = { "Technology": ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMD", "INTU", "ADBE", "CSCO", "ORCL", "INTC", "IBM", "CRM"], 
                        "Health Care": ["JNJ", "PFE", "MRK", "ABBV", "UNH", "LLY", "VTRS", "GILD", "BIIB", "REGN", "AMGN", "ABT", "BMY"], 
                        "Consumer Discretionary": ["AMZN", "TGT", "HD", "NKE", "COST", "MCD", "SBUX", "WMT", "LOW", "MOS", "PG", "PEP"], 
                        "Financials": ["JPM", "BAC", "C", "GS", "WFC", "COF", "AXP", "USB", "BK", "AFL", "TRV", "ALL", "DFS"], 
                        "Industrials": ["MMM", "GE", "HON", "UTX", "DOW", "CAT", "BA", "ETN", "DE", "ITT", "HMC", "LMT", "RTX"], 
                        "Materials": ["LIN", "EMN", "DOW", "MMM", "CF", "LYB", "PCP", "ALB", "NEM", "PPG", "MMM", "ALXN", "MMM"], 
                        "Real Estate": ["SPG", "ARE", "PSA", "REG", "STAG", "O", "WELL", "BXP", "DLR", "EQR", "PLD", "REIT", "SLG"], 
                        "Utilities": ["D", "DUK", "NEE", "AES", "CMS", "EXC", "PG", "ED", "SRE", "WEC", "EIX", "DUK", "AES"], 
                        "Communication Services": ["GOOGL", "META", "NFLX", "T", "CMCSA", "DIS", "VZ", "ATVI", "CHTR", "FOXA", "NFLX", "DIS"], 
                        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "HAL", "OXY", "MRO", "CVX", "XOM", "COP"]
                        }
    selected_industry=st.multiselect('Select the industries of stocks to the portfolio: (recommended max. 10)', stocks_industries_list)

    if selected_stock !='':
        for ticker in selected_stock:
            selected_stock_weightage[ticker]=st.slider('Weightage of stock: '+str(ticker))
            stock_data[ticker]=yf.download(ticker, start=start_date, end=end_date)
            selected_daily_return=stock_data[ticker].pct_change()



###################################################THEME & FORMATING################################
# [theme]
# base="light"
# primaryColor="#e8aefd"
# secondaryBackgroundColor="#dae3ff"
# textColor="#000000"
# font="serif"

