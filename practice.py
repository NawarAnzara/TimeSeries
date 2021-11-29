# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 11:11:45 2021

@author: NawarAnzara
"""
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64
from pmdarima.arima.utils import nsdiffs  
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset

from pmdarima.model_selection import train_test_split
header = st.container()


with header:
    st.title('TechTrioz Solutions: Demand Forecasting Web App')
    

data_file = st.sidebar.file_uploader("Upload CSV",type=['csv'])

def get_data():
    return pd.read_csv(data_file, sep=",", encoding='Latin-1')

df = get_data()
option = st.sidebar.selectbox( 'Choose the feature you would like to predict?', (list(df.columns.values)))
DateTime = st.sidebar.selectbox( 'Choose the date column from your dataset:', (list(df.columns.values)))
periods_input = st.sidebar.number_input('How many months forecast do you want?',
 min_value = 1, max_value = 365)
st.sidebar.write('Please input value between : 1 and 365')
if st.sidebar.button('Forecast'):
                if data_file is None:
                    st.write('Provide a dataset for forecasting')
                if data_file is not None:
                    
                    st.dataframe(df)
            
                   
                    df[f"{DateTime}"] = pd.to_datetime(df[f"{DateTime}"],errors='coerce')
                    
                    st.write(df)
                    df = df.set_index(f"{DateTime}")
                    st.write(df)
                    #df = df.drop(['level_1'], axis = 1 )
                    fig1 = plt.figure(figsize = (16,10))
                    plt.plot(df)
                    plt.legend()
                    st.pyplot(fig1)
                    
                    
                  
                    
                    

                   
                    st.write(adfuller(df))
                    df_diff = df.diff().dropna()
                    st.write(adfuller(df_diff))
                    st.write(df_diff)
                    st.write(plot_acf(df_diff))
                    df = df.dropna()
                    
                   
                    train, test = train_test_split(df, test_size =0.2)
                    
                    
                    testShape = test.shape[0]
                    st.write(testShape)
                   
                    results = pm.auto_arima(train[f'{option}'],start_p=0, start_q=0, d=1, D=1, max_p=5, max_d=5, max_q=5, start_P=0, start_Q=0, max_P=5, max_D=5, max_Q=5, random_state=20, n_fits=50,stepwise=True, suppress_warnings=True,
                           trace=True, seasonal = True, m=12)
                    st.write(results)
                    st.write(results.summary())
                    st.write(results.plot_diagnostics())
                    prediction = pd.DataFrame(results.predict(n_periods = testShape), test.index)
                    prediction.columns = ["predicted_sales"]
                    test["predicted_sales"] = prediction
                    st.write(test)
                    
                    line1 = go.Scatter(x=train.index,
                                       y=train[f'{option}'],
                                       mode='lines',
                                       name='training data')
                    line2 = go.Scatter(x=test.index,
                                       y=test[f'{option}'],
                                       mode='lines',
                                       name='testing data')
                    line3 = go.Scatter(x=test.index,
                                       y=test['predicted_sales'],
                                       mode='lines',
                                       name='prediction')
                    data = [line1, line2, line3]
                    layout = go.Layout(title="Training and Testing sales")
                    figure1 = go.Figure(data=data, layout=layout)
                    st.plotly_chart(figure1)
                    
                    
                    
                    
                    
                    
                    y = periods_input + 1;
                    future_dates = [df.index[-1] + DateOffset(months = x) for x in range(0, y)]
                    if future_dates is not None:
                        st.write(future_dates)
                        future_date_df5 = pd.DataFrame(index = future_dates[1:],columns = df.columns)
                        test2 = test.drop([f'{option}'], axis = 1 )
                        test2.columns = [f'{option}']
                        finaldf = pd.concat([train,test2])
                        st.write(finaldf)
                        
                    results1 = pm.auto_arima(finaldf,start_p=0, start_q=0, d=1, D=1, max_p=5, max_d=5, max_q=5, start_P=0, start_Q=0, max_P=5, max_D=5, max_Q=5, random_state=20, n_fits=50,stepwise=True, suppress_warnings=True,trace=True, seasonal = True, m=12)                    
                    st.write(results1)
                    st.write(results1.summary())
                    st.write(results1.plot_diagnostics())
                    future_date_df5["forecast"] = results1.predict(n_periods = periods_input,dynamic  = True )
                    future_df5 = pd.concat([df,future_date_df5])
                    st.write(future_df5)
                    line4 = go.Scatter(x=future_df5.index,
                                       y=future_df5[f'{option}'],
                                       mode='lines',
                                       name='actual sales')
                    line5 = go.Scatter(x=future_df5.index,
                                       y=future_df5['forecast'],
                                       mode='lines',
                                       name='future forecast')
                    line6 = go.Scatter(x=test.index,
                                       y=test['predicted_sales'],
                                       mode='lines',
                                       name='testData_prediction')
                    data = [line4, line5, line6]
                    layout = go.Layout(title="Future prediction")
                    figure = go.Figure(data=data, layout=layout)
                    st.plotly_chart(figure)
                    
                    



                    
                    
                    
                    

     
               
                    

            
                                   


      
                    		    
		
				
                