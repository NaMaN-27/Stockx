import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import scipy.stats as stats
import streamlit as st
from conditional_prob import conditional_prob_calc

#streamlit configs 
st.set_page_config(layout="wide") 


#path="C:/Users/naman/OneDrive/Desktop/placement/Stockx/db/ticker.xlsx"
# Load downloaded CSV from NSE
#df = pd.read_excel(path)
#tickers = df['Symbol'].tolist()
#NAN in pandas is float
#tickers_ns = [str(symbol) + '.NS' for symbol in tickers if pd.notna(symbol)]
#

selected_value=""

st.title("STOCKX")
col1, col2 = st.columns(2)


with col1:
    #getting input of stock
    ticker = st.text_input("ENTER THE NAME OF STOCK")


with col2:
    if ticker :
        try :

            with st.spinner("Generating plot..."):
                #downloading historical data fro the stock 
                df = yf.download(ticker,period='10y',interval='1d')
                df.columns = df.columns.get_level_values('Price')
                weekly_data = df.resample('W').agg({'Close':['first','last']})
                weekly_data.columns = ['start_of_week_price','end_of_week_price']
                weekly_data['start_of_week'] = weekly_data.index - pd.offsets.Week(weekday=0) + pd.offsets.BDay(0)
                weekly_data['end_of_week'] = weekly_data.index - pd.offsets.Week(weekday=4) + pd.offsets.BDay(0)
                weekly_data['PriceChange'] = weekly_data['end_of_week_price'] - weekly_data['start_of_week_price']
                weekly_data['PercentChange'] = (weekly_data['PriceChange'] / weekly_data['start_of_week_price'])*100

                percent_changes = weekly_data['PercentChange'].dropna()

                # Plot the histogram
                figure=plt.figure(figsize=(10, 6))
                sns.histplot(percent_changes, bins=30, kde=True, stat="density")

                (mu, sigma) = norm.fit(percent_changes)
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x,mu,sigma)
                plt.plot(x,p,'k',linewidth=2)
                plt.axvline(mu,color='r', linestyle='dashed', linewidth=2)
                plt.axvline(mu+sigma, color='g', linestyle='dashed', linewidth=2)
                plt.axvline(mu - sigma, color='g', linestyle='dashed', linewidth=2)
                plt.axvline(mu + 2*sigma, color='b', linestyle='dashed', linewidth=2)
                plt.axvline(mu - 2*sigma, color='b', linestyle='dashed', linewidth=2)
                # Add labels for the lines
                plt.text(mu, plt.ylim()[1] * 0.9, 'mean', color='r', ha='center')
                plt.text(mu + sigma, plt.ylim()[1] * 0.8, '+1σ', color='g', ha='center')
                plt.text(mu - sigma, plt.ylim()[1] * 0.8, '-1σ', color='g', ha='center')
                plt.text(mu + 2*sigma, plt.ylim()[1] * 0.7, '+2σ', color='b', ha='center')
                plt.text(mu - 2*sigma, plt.ylim()[1] * 0.7, '-2σ', color='b', ha='center')

                title = "Fit results: mu = %.2f,  sigma = %.2f" % (mu, sigma)
                plt.title(title)

                plt.xlabel('Percent Change')
                plt.ylabel('Density')
                st.pyplot(figure)
            st.success("Plot generated successfully!")

            selected_value = st.slider("CHOOSE EXPECTED MOVEMENT", min_value=1, max_value=10, step=1)

        
        except Exception as e:
            st.error(f"Error: {e}")

    if selected_value :
        probability = norm.sf(selected_value, loc=mu, scale=sigma)
        st.write(f"The probability of {ticker} stock going up more than {selected_value}% in one week is {round(probability*100,1):.1f}%")

if ticker :
    prob_df, count_df = conditional_prob_calc(weekly_data)

    with col1:

        labels = ["<-4%", "-4% to -3%", "-3% to -2%", "-2% to -1%", "-1% to 0%", 
            "0% to 1%", "1% to 2%", "2% to 3%", "3% to 4%", ">4%"]

        extra_cols = ["Negative","Positive", ">1%", ">2%", ">3%"]

        all_columns = labels + extra_cols
        heatmap_data = prob_df[all_columns].astype(float).round(2)
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=ax)

        ax.set_title("Probability of This Week's Return Given Last Week", fontsize=14)
        ax.set_xlabel("This Week")
        ax.set_ylabel("Previous Week")

        # Show in Streamlit
        st.pyplot(fig)
        
        st.write("Transition Probability Matrix")
        st.dataframe(prob_df)

        st.write("Count Matrix")
        st.dataframe(count_df)