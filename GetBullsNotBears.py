################################################################
##      GET BULL NOT BEARS    ## STOCK ANALYSIS APP           ##
################################################################


import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_option_menu import option_menu
import datetime as dt
from datetime import datetime, timedelta
import yfinance as yf
import requests
import stocknews
from stocknews import StockNews

icon_placeholder= "https://raw.githubusercontent.com/gaurav1nemani/GBNB/main/Assets/GBNB%20Icon.png"
sidebar_icon_placeholder="https://raw.githubusercontent.com/gaurav1nemani/GBNB/main/Assets/GBNB%20Icon.png"

#reference: 1. Minh Notes, 2.Streamlit Documentation and Streamlit Community, 3. youtube: financial programing with Ritvick, CFA, 4. Youtube: Coding is Fun, 5. Youtube: Intrendias, 6. MBD & Ara ,7. app icon attribution: <a href="https://www.flaticon.com/free-icons/stock-market" title="stock market icons">Stock market icons created by Freepik - Flaticon</a>
st.set_page_config(page_title="Get Bulls Not Bears", page_icon=icon_placeholder, layout="wide")

# Sidebar icon
st.sidebar.image(sidebar_icon_placeholder, width=30)

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 10px;">
        <img src="{icon_placeholder}" alt="App Icon" style="height: 50px;"> 
        <h1 style="margin: 0;">Get Bulls Not Bears</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# # Sidebar with ticker input and app icon
# st.sidebar.markdown(
#     f"""
#     <div style="display: flex; align-items: center; gap: 10px;">
#         <img src="{sidebar_icon_placeholder}" alt="Sidebar Icon" style="height: 30px;"> 
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

#Make the side bar with the ticker, start date, end date with update button
st.sidebar.header('Enter the stock:')
global ticker
ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
ticker = st.sidebar.selectbox("Ticker", ticker_list)

global start_date
start_date = st.sidebar.date_input('Start Date', value=datetime.today()-timedelta(days=365), format="DD/MM/YYYY")

global end_date
end_date=st.sidebar.date_input('End Date', format="DD/MM/YYYY")

stock_data=yf.download(ticker, start=start_date, end=end_date)
csv=stock_data.to_csv(index=True)

st.sidebar.button("Update Data", type="primary")
st.sidebar.download_button("Download Data as CSV", data=csv, file_name=f'Stockdata_{datetime.now().date()}.csv')

#make the menu option
menu=option_menu(
    menu_title=None,
    options=["Summary", "Chart", "Financials", "Monte Carlo Simulation","News", "Portfolio Analysis"],
    icons=["journal-text", "graph-up-arrow", "cash-coin", "bezier2", "newspaper", "briefcase"],
    default_index=0,
    orientation="horizontal"
)

#Summary Page
if menu=="Summary":

    st.header("Company Information")
    st.write(yf.Ticker(ticker).info.get('longBusinessSummary'))
    
    key_info=yf.Ticker(ticker).info
    col1, col2=st.columns([0.4,0.6], gap="small", vertical_alignment="top")

    with col1:
        #Key Statistics
        st.header("Key Statistics")
        key_statistics1 = { 
        'Market Cap': key_info.get('marketCap'), 
        'Enterprise Value': key_info.get('enterpriseValue'), 
        'Trailing P/E': key_info.get('trailingPE'), 
        'Forward P/E': key_info.get('forwardPE'), 
        'PEG Ratio': key_info.get('pegRatio'), 
        'Price to Book': key_info.get('priceToBook'), 
        'Dividend Rate': key_info.get('dividendRate'), 
        'Dividend Yield': key_info.get('dividendYield'), 
        'Revenue': key_info.get('totalRevenue'), 
        'EBITDA': key_info.get('ebitda'), 
        'Net Income': key_info.get('netIncomeToCommon'), 
        'Free Cash Flow': key_info.get('freeCashflow') 
        }
        key_statistics_df1 = df(list(key_statistics1.items()), columns=['Metric', 'Value'])
        key_statistics_df1 = key_statistics_df1.set_index(key_statistics_df1.columns[0])
        st.write(key_statistics_df1)

    with col2:
        # Area plot and bar plot for stock data
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        area_plot = go.Scatter(x=stock_data.index, y=stock_data['Adj Close'].squeeze(),
                                fill='tozeroy', fillcolor='rgba(133, 133, 241, 0.2)', showlegend=False)
        fig.add_trace(area_plot, secondary_y=True)

        bar_plot = go.Bar(x=stock_data.index, 
                        y=stock_data['Volume'].squeeze(), 
                        marker_color=np.where(stock_data['Close'].pct_change() < 0, 'red', 'green'),
                        showlegend=False
                    )
        fig.add_trace(bar_plot, secondary_y=False)

        fig.update_xaxes(
            rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(label="MAX", step="all")
                    ])
                )
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Adjusted Price",
            template='plotly_white',
            margin=dict(t=10, b=10)
        )
        fig.update_yaxes(secondary_y=False)

        st.plotly_chart(fig)
    
    st.subheader("Top 5 Share Holders")
    share_holders = yf.Ticker(ticker).major_holders 
    st.table(share_holders)

#Chart Page
elif menu=="Chart":

    chart_type=st.selectbox("Select the chart style: ", options=["Line Chart", "Candelstick Chart"])
    if chart_type=="Line Chart":
        if ticker !='':
            line_chart_figure=px.line(
                x=stock_data.index, 
                y=stock_data['Adj Close'].squeeze()
            )
            line_chart_figure.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(label="MAX", step="all")
                    ])
                ))
            line_chart_figure.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Adjusted Price"
            )
        st.plotly_chart(line_chart_figure)

    if chart_type=="Candelstick Chart":
        if ticker != '':
            fig_candle = make_subplots(specs=[[{"secondary_y": True}]])
            candlestick_figure = go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'].squeeze(),
                high=stock_data['High'].squeeze(),
                low=stock_data['Low'].squeeze(),
                close=stock_data['Adj Close'].squeeze()
                )
            fig_candle.update_layout(height=800)
            fig_candle.add_trace(candlestick_figure, secondary_y=True)

            fig_candle.update_xaxes(
                rangeslider_visible=False,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(label="MAX", step="all")
                    ])
                ))

            volume_plot = go.Bar(x=stock_data.index, y=stock_data['Volume'].squeeze(), marker_color=np.where(stock_data['Close'].pct_change() < 0, 'red', 'green'),
                            showlegend=False)
            fig_candle.add_trace(volume_plot, secondary_y=False)
            fig_candle.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Adjusted Price"
            )

            #Add moving average line
            moving_avg=stock_data['Adj Close'].rolling(window=50).mean()
            moving_plot = go.Scatter(
                x=stock_data.index, 
                y=moving_avg.squeeze(), 
                mode='lines',
                name='50 day moving average',
                line=dict(color='blue'),
                showlegend=True
            )
            fig_candle.add_trace(moving_plot, secondary_y=True)

        st.plotly_chart(fig_candle)

#Financials Page
elif menu=="Financials":
    
    stock=yf.Ticker(ticker)
    
    options_list=["Balance Sheet","Cash Flow Statement","Income Statement"]
    get_type_financial=st.selectbox("Select type of financial data: ", options=options_list)
    get_annual_quarterly=st.selectbox("Select annual or quarterly: ", options=["Annual","Quarterly"])

    if get_type_financial=="Balance Sheet":
        st.subheader('Balance Sheet')
        if get_annual_quarterly=="Annual":
            bs_data=stock.balance_sheet
            st.write(bs_data)
        elif get_annual_quarterly=="Quarterly":
            bs_data=stock.quarterly_balance_sheet
            st.write(bs_data)
    
    if get_type_financial=="Cash Flow Statement":
        st.subheader('Cash Flow Statement')
        if get_annual_quarterly=="Annual":
            cf_data=stock.cashflow
            st.write(cf_data)
        elif get_annual_quarterly=="Quarterly":
            cf_data=stock.quarterly_cashflow
            st.write(cf_data)

    if get_type_financial=="Income Statement":
        st.subheader('Income Statement')
        if get_annual_quarterly=="Annual":
            ist_data=stock.financials
            st.write(ist_data)
        elif get_annual_quarterly=="Quarterly":
            ist_data=stock.quarterly_financials
            st.write(ist_data)

#Monte Carlo Simulation Page
elif menu=="Monte Carlo Simulation":
    
    def get_randomseed():
        random_seed=st.number_input("Select a random seed (1-1000): ", min_value=1, max_value=1000)
        return random_seed
    
    def get_timehorizon():
        time_horizon=st.selectbox("Select time horizon (in days): ", options=[30, 60, 90])
        return time_horizon
    
    def get_nbrsimulations():
        nbr_simulations=st.selectbox("Select the number of simulations (200, 500, 1000): ", options=[200, 500, 1000])
        return nbr_simulations
    
    random_seed=get_randomseed()
    time_horizon=get_timehorizon()
    nbr_simulations=get_nbrsimulations()

    def get_montecarlo(stock_data, get_randomseed, get_timehorizon, get_nbrsimulations):
        
        np.random.seed(random_seed)
        close_price = stock_data['Adj Close']
        daily_return = close_price.pct_change()
        daily_volatility = np.std(daily_return)

        simulation_df = pd.DataFrame()

        for i in range(nbr_simulations):
    
            next_price = []
            last_price = close_price.iloc[-1]
    
            for j in range(time_horizon):
                future_return = np.random.normal(0, daily_volatility)

                future_price = last_price * (1 + future_return)

                next_price.append(future_price)
                last_price = future_price
    
            next_price_df = pd.Series(next_price).rename('sim' + str(i))
            simulation_df = pd.concat([simulation_df, next_price_df], axis=1)

        plt.figure(figsize=(15, 10))

        plt.plot(simulation_df)
        plt.axhline(y=close_price.iloc[-1].squeeze(), color='black',linewidth=1.5)
        plt.title('Monte Carlo simulation for AAPL stock price in next 200 days')
        plt.xlabel('Day')
        plt.ylabel('Price')
        plt.legend(['Current stock price is: ' + str(np.round(close_price.iloc[-1], 2))])

        
        st.pyplot(plt)
        
        #Add VAR Value
        ending_price = simulation_df.iloc[-1:, :].squeeze()
        ending_price = pd.to_numeric(ending_price, errors='coerce').dropna()
        st.write(ending_price)
        future_price_95ci = np.percentile(ending_price, 5)
        VaR = close_price.iloc[-1] - future_price_95ci
        st.write('Value at Risk (VaR) at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')
    
    get_montecarlo(stock_data, random_seed, time_horizon, nbr_simulations)

#News Page
elif menu=="News":
    st.header(f'News of {ticker}')
    sn=StockNews(ticker, save_news=False)
    news_data=sn.read_rss()
    for i in range(20):
        st.subheader(f'News Article {i+1}')
        st.write(news_data['published'][i])
        st.write(news_data['title'][i])
        st.write(news_data['summary'][i])

        #Get Sentiment scores for future sentiment analysis
        sentiment_title_data=(news_data['sentiment_title'][i])
        st.write(f'Title Sentiment: {sentiment_title_data}')
        sentiment_data=news_data['sentiment_summary'][i]
        st.write(f'News Sentiment: {sentiment_data}')

#Portfolio Analysis Page
elif menu=="Portfolio Analysis":
    selected_stock=st.multiselect('Select the stocks to the portfolio: (recommended max. 10)', ticker_list)
        
    if selected_stock !='':
        selected_stock_data = {} 
        for ticker in selected_stock:
            data = yf.download(ticker, period="1y")
            if not data.empty:
                selected_stock_data[ticker] = data

        sp500_data = yf.download('^GSPC', period="1y") 
        sp500_data['Daily Return'] = sp500_data['Adj Close'].pct_change() 
        sp500_data['Cumulative Return'] = (1 + sp500_data['Daily Return']).cumprod()
        
        fig=go.Figure()

        for ticker, data in selected_stock_data.items(): 
            data['Daily Return'] = data['Adj Close'].pct_change() 
            data['Cumulative Return'] = (1 + data['Daily Return']).cumprod()

        for ticker, data in selected_stock_data.items():
            fig.add_trace(go.Scatter( 
                        x=data.index, 
                        y=data['Cumulative Return'], 
                        mode='lines', 
                        name=ticker 
                    ))
        
        fig.add_trace(go.Scatter(
            x=sp500_data.index,
            y=sp500_data['Cumulative Return'],
            mode='lines',
            name='S&P 500',
            line=dict(color='black')
        ))
        fig.update_layout( title='1-Year Return Comparison of Selected Stocks', 
                    xaxis_title='Date', 
                    yaxis_title='Cumulative Return', 
                    legend_title='Stocks' 
                    )

        st.plotly_chart(fig)


# References:-

# 1. Professor Minh, Professor Juliana and Professor Fernando (Financial Programming) MBD Class 2024, IESEG School of Management, Lille
# 2. https://stackoverflow.com/questions/44449139/change-date-format-mm-dd-yyyy-to-mm-dd-yyyy-in-macro-vba
# 3. Streamlit Documentation and Community
# 4. Numpy Documentation and Community
# 5. Pandas Documentation and Community
# 6. Matplotlib Documentation and Community
# 7. Plotly Documentation and Community
# 8. StockNews API Documentation and Community
# 9. youtube: financial programing with Ritvick, CFA
# 10. Youtube: Coding is Fun
# 11. Youtube: Intrendias
# 12. Arajem Aboudi & Paul Gaillard, MBD 2024, IESEG School of Management (Thank you for all the help! :) )
# 13. MBD Class 2024, IESEG School of Management
