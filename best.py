import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf
from datetime import date,timedelta
from keras.models import load_model
import streamlit as st 
import datetime
from plotly import graph_objs as go

st.markdown("<h1 style='text-align: center;'>Stock Trend Prediction</h1>", unsafe_allow_html=True)

end = date.today()+ timedelta(days=1)
start = st.date_input("Enter the Start date for Data Visualization", max_value=end,min_value=datetime.date(2016,4,1), value=datetime.date(2016,4,1))
start_year = str(start.year)
stocks = [ "HPE", "GOOG","CRM", "NFLX", "TSLA", "UBER"]
selected_stock = st.selectbox("Select any Stock Ticker" , stocks)

#Scraping the data
yf.pdr_override()
df = pdr.get_data_yahoo(selected_stock,start,end)

#Describing the Data
st.subheader('Data from ' + start_year + ' to till date')
st.table(df.describe()) 

#Visualizations
st.subheader('Closing Price vs Time chart')
trace1 = go.Scatter(x=df.index, y=df.Close, name="Closing Price")
fig = go.Figure(data=[trace1])
tomorrow=df.iloc[-2]['Close']
fig.update_layout( xaxis_title="Time", yaxis_title="Closing Price", xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
trace1 = go.Scatter(x=df.index, y=ma100, mode="lines", name="MA100")
trace2 = go.Scatter(x=df.index, y=ma200, mode="lines", name="MA200")
trace3 = go.Scatter(x=df.index, y=df.Close, mode="lines", name="Closing Price")
fig1 = go.Figure(data=[trace1, trace2, trace3])
fig1.update_layout(xaxis_title="Time", yaxis_title="Closing Price", xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)

#Splitting the Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0 : int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])

# Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#Loading the trained model
model = load_model('keras_model.h5')

#Testing the data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

#Splitting the data into x_test and y_test
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

#Making Predictions
y_predicted = model.predict(x_test)
y_tomorrow = tomorrow
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_test = y_test * scale_factor * 1.8
y_predicted = y_predicted * scale_factor * 1.8
today = df.iloc[-1]['Close']
st.markdown(f"<p style='background-color:green;color:#FFFFFF;font-size:24px;text-align:center;'>Today's Closing Price: {today:.2f}</p>", unsafe_allow_html=True)
st.markdown(f"<p style='background-color:red;color:#FFFFFF;font-size:24px;text-align:center;'>Tomorrow's Predicted Closing Price: {y_tomorrow:.2f}</p>", unsafe_allow_html=True)

st.subheader('Last week\'s actual and predicted closing prices:')
df_results = pd.DataFrame({
    'Actual Prices': y_test[-7:],
    'Predicted Prices': y_predicted[-7:].flatten()
})
html_table = (df_results.style
              .format('{:.2f}')
              .render())

st.markdown(html_table, unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)


# Final graph prediciting the values
st.subheader('Original vs Predicted')
trace1 = go.Scatter(x=df.tail(len(y_test)).index, y=y_test, mode="lines", name="Original Price",line=dict(color='cyan'))
trace2 = go.Scatter(x=df.tail(len(y_test)).index, y=y_predicted.flatten(), mode="lines", name="Predicted Price",line=dict(color='yellow'))
fig2 = go.Figure(data=[trace1, trace2])
fig2.update_layout( xaxis_title="Time", yaxis_title="Closing Price", xaxis_rangeslider_visible=True)
st.plotly_chart(fig2)



