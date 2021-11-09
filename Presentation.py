import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import plotly.figure_factory as ff
from plotly import graph_objs as go
import statsmodels.api as sm
import streamlit as st
import plotly.express as px

st.title("User Account Balance Prediction")

@st.cache
def load_data():
    data = pd.read_csv('CGI_dataset.csv')
    data = data[(data.transaction_date > "2020-01-01")]
    data = data.set_index('transaction_date')
    return data

def show_data(data):
    # st.subheader("MA model example ")
    
    fig = go.Figure(data = go.Table(header = dict(values = ['Date','balance'],
                                                  fill_color = '#FD8E72'),
                    cells = dict(values = [data.index,data.balance])))
    fig.update_layout(margin = dict(l = 5, r = 5, b = 10, t = 10), height = 200, 
                      font=dict(
                                family="Courier New, monospace",
                                size=12,
                                color="Black"
                            ))
    st.write(fig)

@st.cache
def plot_balance_data(data):
    st.subheader("How the data looks like?")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data.index, y = data.balance, name = 'account_balance'))
    fig.layout.update(margin = dict(l = 5, r = 5, b = 1, t = 1),xaxis_rangeslider_visible = True
                      ,xaxis_title="Date", yaxis_title = "Account Balance")
    st.plotly_chart(fig)


def plot_moving_average(data):
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data.index, y = data.moving_average, name = 'account_balance'))
    fig.layout.update(margin = dict(l = 5, r = 5, b = 1, t = 1),xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

@st.cache
def balance_acf(data):
    # df = data.copy()
    # df['integration'] = data['balance'].diff(12)
    acf = pd.DataFrame(sm.tsa.stattools.acf(data.balance[12:],nlags = 100), columns = ['acf'])
    # st.subheader('Correlation')
    # fig = sm.graphics.tsa.plot_acf(data.balance, lags=30)
    
    st.subheader("Auto Correlation(ACF)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = acf.index, y = acf.acf, name = 'Correlation', mode = "lines + markers"))
    fig.layout.update(margin = dict(l = 20, r = 20, b = 70, t = 30),xaxis_rangeslider_visible = True
                      ,xaxis_title = "Lag", yaxis_title= "Significance Level",
                      font=dict(
                                family="Courier New, monospace",
                                size=12,
                                color="Black"
                            ))
    st.plotly_chart(fig)
    # fig = autocorrelation_plot(data.balance)
    # sm.graphics.tsa.plot_acf(data.balance, lags=10)
    # st.plotly_chart(fig)
    # plt.show()
    # pass

@st.cache 
def balance_pacf(data):
    # st.subheader('Partial Correlation')
    # fig = sm.graphics.tsa.plot_pacf(data.balance, lags=100)
    # # fig = autocorrelation_plot(data.balance)
    # # sm.graphics.tsa.plot_acf(data.balance, lags=10)
    # st.plotly_chart(fig)
    pacf = pd.DataFrame(sm.tsa.stattools.pacf(data.balance,nlags = 100), columns = ['pacf'])
    # st.subheader('Correlation')
    # fig = sm.graphics.tsa.plot_acf(data.balance, lags=30)
    st.write("**Partial Auto Correlation(PACF)**")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = pacf.index, y = pacf.pacf, name = 'partial auto correlation',mode = "lines + markers"))
    fig.layout.update(margin = dict(l = 20, r = 20, b = 70, t = 30),xaxis_rangeslider_visible = True
                      ,xaxis_title = "Lag", yaxis_title= "Significance Level",
                      font=dict(
                                family="Courier New, monospace",
                                size=12,
                                color="Black"
                            ))
    st.plotly_chart(fig)
    # plt.show()
    # pass

    
    
@st.cache
def plot_MA_table(data):
    MA_example = pd.DataFrame(columns = ["Original_Balance","Predicted_Balance","Correction"])
    MA_example['Original_Balance'] = [1000,2000]
    MA_example['Predicted_Balance'] = [990,2005]
    MA_example['Correction'] = [10,-5]
    
    st.subheader("MA model example ")
    
    fig = go.Figure(data = go.Table(header = dict(values = ["Original_Balance","Predicted_Balance","Correction"],
                                                  fill_color = '#FD8E72'),
                    cells = dict(values = [MA_example.Original_Balance,MA_example.Predicted_Balance,MA_example.Correction])))
    fig.update_layout(margin = dict(l = 5, r = 5, b = 1, t = 1), height = 80)
    st.write(fig)
    
    st.write("Our MA model will be something like **prediction = current_prediction + c1(10) + c2(-5)**")

@st.cache
def differnce_acf(data):
    df = data.copy()    
    df['integration'] = data['balance'].diff(1)
    acf = pd.DataFrame(sm.tsa.stattools.acf(df.integration[1:],nlags = 100), columns = ['acf'])

    st.write("**Auto Correlation(ACF) of Differneced Data**")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = acf.index, y = acf.acf, name = 'partial auto correlation',mode = "lines + markers"))
    fig.layout.update(margin = dict(l = 20, r = 20, b = 70, t = 30),xaxis_rangeslider_visible = True
                      ,xaxis_title = "Lag", yaxis_title= "Significance Level",
                      font=dict(
                                family="Courier New, monospace",
                                size=12,
                                color="Black"
                            ))
    st.plotly_chart(fig)
    # plt.show()
    # pass
    

def train_model(data):
    
    st.subheader("Training the model and Explainability of it.")
    st.write("**Input Data**")
    st.markdown("- **Training Data: Jan 2020 - Dec 2020** \
                \n - **Testing Data: Jan 2021**")

    st.markdown("**The parameters of the model**\
             \n- **AR: 1**\
             \n- **I : 1**\
             \n- **MA: 0** \
             \n- **Seasonal AR: 0**\
             \n- **Seasonal I: 1**\
             \n- **Seasonal MA: 2**\
             \n- **Seasonality: 1 month**")
             
    st.write("The parameters of the model were chosen with a grid search approach.")    
    
    train_data = data.balance[(data.index > "2020-01-01") & (data.index <= "2020-12-31")]
    

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(train_data, order=(1,1,0), seasonal_order = (0,1,2,12))
    # model = ARIMA(train_data, order=(4,2,2))
    model_fit = model.fit()
    
    st.write("**How do we explain what is our actual model?**")
    st.write("From the parameters of the model, we can generate the equation of our model.")
    parameters = pd.DataFrame(model_fit.params, columns = ['Coefficients'])
    st.write(parameters)
    
    
    return model_fit


def model_quality(data,model_fit):
    st.subheader("Model fit on the training Data")
    residuals = pd.DataFrame(model_fit.resid, columns = ['residuals'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = residuals.index, y = residuals.residuals))
    fig.layout.update(margin = dict(l = 20, r = 20, b = 70, t = 30),xaxis_rangeslider_visible = True
                      ,xaxis_title = "Date of Prediction", yaxis_title= "Residual (Original - Predicted)",
                      font=dict(
                                family="Courier New, monospace",
                                size=12,
                                color="Black"
                            ))
    st.plotly_chart(fig)
    # summary stats of residuals
    st.write("**Summarizing The Model Residuals**")
    st.write(residuals.describe())
    
def predict_data(data,model_fit):
    test_data = data.balance[(data.index >= "2021-01-01")][:30]
    
    y_pred = model_fit.predict(start = test_data.index.min() ,end = test_data.index.max())
    
    residuals = test_data - y_pred
    # st.write("ploting results")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data.index, y=y_pred,
                        name='predicted values'))
    fig.add_trace(go.Scatter(x=test_data.index, y=test_data,
                        name='original values'))
    fig.layout.update(margin = dict(l = 5, r = 5, b = 1, t = 20),width = 850, xaxis_rangeslider_visible = True
                      ,xaxis_title = "Date", yaxis_title= "Account Balance",
                      font=dict(
                                family="Courier New, monospace",
                                size=12,
                                color="Black"
                            ))
    st.plotly_chart(fig)
    
    df = pd.DataFrame(columns = ['original','prediction'])
    df['original'] = test_data#.reshape(-1)
    df['prediction'] = list(y_pred)#.reshape(-1)
    
    st.write("**Prediction Summary**")
    st.write(df.describe())
    
@st.cache    
def intergration(data):
    # df = data.copy()
    # df['integration'] = data['balance'].diff(12).diff(12)
    # # df['integration'] = data['integration'].diff(12)
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x = df.index[24:], y = df.integration[24:]))
    # fig.layout.update(margin = dict(l = 20, r = 20, b = 70, t = 30),xaxis_rangeslider_visible = True
    #                   ,xaxis_title = "Date", yaxis_title= "Two Difference of (Lag 12)",
    #                   font=dict(
    #                             family="Courier New, monospace",
    #                             size=12,
    #                             color="Black"
    #                         ))
    # st.plotly_chart(fig)
    
    df = data.copy()
    df['integration'] = data['balance'].diff(1).diff(1)
    # df['integration'] = data['integration'].diff(12)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df.index[2:], y = df.integration[2:]))
    fig.layout.update(margin = dict(l = 20, r = 20, b = 70, t = 30),xaxis_rangeslider_visible = True
                      ,xaxis_title = "Date", yaxis_title= "Two Difference of (Lag 1)",
                      font=dict(
                                family="Courier New, monospace",
                                size=12,
                                color="Black"
                            ))
    st.plotly_chart(fig)
    
def main():
    data = load_data()
        
    st.header('User Account Balance Data')
    # st.write("First 50 rows from the data available of user's daily account balance ")
    show_data(data)
    # st.write(data.head(50))
    
    
    st.write("Understanding the data")
    st.write(data.describe())
    
    
    plot_balance_data(data)
    
    st.write("**What can we analyse from this visualization?**")
    st.markdown("- The data has a specific pattern repiting multiple time. We might be able to use past data to predict future values. \
                \n- Data shows a constant upward trend. \
                \n- Data seems to have some seasonal changes which occurs after a specific interval") 
    
    st.write("Based on this observations ARIMA model can be a better fit for this data.")
    
    st.header("About Arima Model")
    st.write("Arima model is a statistical model which stand for Autoregressive Itegrated Moving Average Model.")
    st.write("**ARIMA = Autoregression(AR) + Moving Average Error(MA) + Differentiated Values(I) + Seasonal relations of (AR+MA+I)**")
    
    st.write("Let's understand each part of ARIMA separately and check how data satisfies each term.")
    
    st.subheader("Auto_regression(AR) in ARIMA model")
    st.write("Auto_regression models means that the data has a strong relation with its past lags. i.e. past data can explain current data points with a simple mathematical equation.")
    
    balance_pacf(data)
    
    st.write("**How to determine past dependence from PACF?**")
    st.markdown("* PACF shows the significance of relation between past data and present data.\
                \n * Based on the graph previous days data (**Lag 1**) has a strong relation with today's date.")
    
    st.subheader("Integral(I) in ARIMA model")
    st.write("**I** in the **ARIMA** model simply means differencing current value with an old value.\
             **I** is necessary when there is a trend in the data.")
    balance_acf(data)
    st.markdown("- The constant drop in **ACF** implies that there is a constant trend in the data.")
    st.write("**Why it is necessary to take difference?**")
    st.markdown("\
                - Our model shows upward trend and the mean of the data constantly changes towards a higher value. \
             \n - For statistical model it is necessary to remove the trend and make the time series stationary i.e. mean and variance should be constant. \
             \n - Also, differencing can reveal a seasonality in the data as well.")
    
    intergration(data)
    
    st.subheader("Moving Average(MA) in ARIMA model")
    
    st.write("**What is a moving average model?**")
    st.markdown("- In a simple terms, when a model takes past errors into the account and use those values in the prediction of current value. It is a moving average model.")
    plot_MA_table(data)
    
    st.write("To understand the order of MA, we will check the ACF of the differneced data taken with every 12th value.")
    differnce_acf(data)
    st.write("**Till what data should we use the MA relation for our model?**")
    st.markdown("- ACF graph of the data with differncing with the past value can shows significance of the number of past data. \
                \n - ACF shows a constant pattern without shutting down. Based on the plot a seasonal MA model can be effect.")
    
    st.subheader("Seasonality in ARIMA")
    st.write("**What is Seasonality?**")
    st.markdown("- Seasonality of the data means after which observation the data repeats its pattern. \
                \n- Using the past value of the same date can explain current data well.")
    
    
    
    model = train_model(data)
    
    model_quality(data,model)
    
    st.subheader("Model Predictions")
    predict_data(data,model)
    
    st.subheader("Summary")

if __name__ == '__main__':
    main()
# slow decay in residual means d is required
# add pacf of balance to explain arima model
# add table of value and correction to explain moving average model
# ACF for order of MA and pacf for the order of ar
# I differenciation of the data
# show graph of seasonal repitition for S in Sarima and explain that seasonal part is same as non seasonal 
#                                   but we take ar and ma of that seasonal component (m observation back)
