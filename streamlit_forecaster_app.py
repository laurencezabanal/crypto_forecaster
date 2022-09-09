# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:08:02 2022

@author: john.l.a.zabanal
"""
#import tkinter as tk
#from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#from fastquant import get_crypto_data
import streamlit as st
# seaborn example
# pyinstaller --hidden-import "babel.numbers" main.py
from string import ascii_letters
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np


import openpyxl
from tqdm import tqdm
from pycaret.regression import *
#import tkinter as tk
#from tkinter import *
#from tkinter import ttk
# tkcalendar import Calendar
import os

#import tkinter  as tk 
#from tkcalendar import DateEntry
from dateutil.relativedelta import relativedelta
from datetime import date, datetime

all_crypto = ["BTC/USDT","ETH/USDT","AXS/USDT","SLP/USDT"]
#all_crypto = ["BTC/USDT","ETH/USDT","AXS/USDT","SLP/USDT"]


# def create_plot():
#     sns.set(style="white")

#     # Generate a large random dataset
#     rs = np.random.RandomState(33)
#     d = pd.DataFrame(data=rs.normal(size=(100, 26)),
#                      columns=list(ascii_letters[26:]))

#     # Compute the correlation matrix
#     corr = d.corr()

#     # Generate a mask for the upper triangle
#     mask = np.zeros_like(corr, dtype=np.bool)
#     mask[np.triu_indices_from(mask)] = True

#     # Set up the matplotlib figure
#     f, ax = plt.subplots(figsize=(11, 9))

#     # Generate a custom diverging colormap
#     cmap = sns.diverging_palette(220, 10, as_cmap=True)

#     # Draw the heatmap with the mask and correct aspect ratio
#     sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#                 square=True, linewidths=.5, cbar_kws={"shrink": .5})

#     return f

# def create_plot2():
#     sns.set(style="white")
#     fig2, ax = plt.subplots(figsize=(10, 8))
#     #fig2 = Figure(figsize=(5, 4), dpi=100)
#     #t = np.arange(0, 3, .01)
#     #fig2.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))
    
#     canvas = FigureCanvasTkAgg(fig2, master=root)  # A tk.DrawingArea.
#     canvas.draw()
#     canvas.get_tk_widget().grid(row=4, column=0, sticky="w",pady=2)
    
#     # toolbar = NavigationToolbar2Tk(canvas, root)
#     # toolbar.update()
#     canvas.get_tk_widget().grid(row=4, column=0, sticky="w",pady=2)
    
#     return fig2


def my_forecast(*args): # triggered when value of string varaible changes
    #date_from= datetime.strptime(date_start,'%m/%d/%y')
    #date_to = datetime.strptime(date_end,'%m/%d/%y')
    #crypto_selected = ent_crypto.get()
    #item_number = 'item_' + ent_item.get()
    date_from= date_start
    date_to = date_end
    model_reference = crypto_selected.replace("/", "_")

    
    all_dates2 = pd.date_range(start=date_from, end = date_to, freq = 'D')
    score_df2 = pd.DataFrame()
    # add columns to dataset
    score_df2['date'] = all_dates2
    score_df2['month'] = [i.month for i in score_df2['date']]
    score_df2['year'] = [i.year for i in score_df2['date']]
    score_df2['quarter'] = [i.quarter for i in score_df2['date']]
    score_df2['day_of_week'] = [i.dayofweek for i in score_df2['date']]
    score_df2['day_of_year'] = [i.dayofyear for i in score_df2['date']]
    score_df2.head()

    
    #load model
    #path_init = r"C:\Users\John Laurence\Documents\Data Science Projects\Experiments\Crypto\trained_models"
    #path = path_init + '\\' + model_reference
    #dt_saved = load_model(path)
    dt_saved = load_model(model_reference)
    
    p2 = predict_model(dt_saved, data=score_df2)
    p2['crypto']=crypto_selected
    #p2['item number'] =item_number
    p2['model reference'] = model_reference
    print('Forecast successfully generated!')
    #p2.to_excel(model_reference+"Forecast.xlsx")
    
    # final_df2 = p2[['date','Label']]
    # final_df2.rename(columns={'Label': 'Forecast'},inplace=True)
    # #final_df2 = p2[['date','Label','sales']]
    
    # fig, ax = plt.subplots(1,1,figsize=(10,8))
    # df_plot2 = final_df2.set_index('date')
    # df_plot2[['Forecast']].plot(ax=ax,linewidth=2,legend=False)
    
    # df_plot2[['Label','sales']].plot(ax=ax,linewidth=2)
    # ax.fill_between(df_plot.index,
    #                 df_plot['auto_arima_season_length-12_lo-80'],
    #                 df_plot['auto_arima_season_length-12_hi-80'],
    #                 alpha=0.35,
    #                 color='green',
    #                 label = 'auto_arima_level_80'
    
    # )
    
    # ax.fill_between(df_plot.index,
    #                 df_plot['auto_arima_season_length-12_lo-95'],
    #                 df_plot['auto_arima_season_length-12_hi-95'],
    #                 alpha=0.2,
    #                 color='green',
    #                 label = 'auto_arima_level_95'
    
    # )
    
    # ax.set_title('Crypto Forecast of '+model_reference,fontsize=22)
    # ax.set_ylabel('Closing Price Forecast',fontsize=20)
    # ax.set_xlabel('Date',fontsize=20)
    # #ax.legend(prop={'size':15})
    # ax.grid()
    # for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label.set_fontsize(20)
    return p2

    
    
    # canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    # canvas.draw()
    # canvas.get_tk_widget().grid(row=4, column=0, sticky="w",pady=2)
    
    # # toolbar = NavigationToolbar2Tk(canvas, root)
    # # toolbar.update()
    # canvas.get_tk_widget().grid(row=4, column=0, sticky="w",pady=2)

    # #print(score_df2)
    # print("success")
    # #print(p2)

crypto_selected = st.selectbox('Pick one', all_crypto)

st.sidebar.info('This app is created to predict crypto currency')
st.sidebar.success('https://www.pycaret.org')

#st.sidebar.image(image_hospital)

st.title("Pycaret Forecaster App")

date_start = st.date_input('Date Start')
date_end = st.date_input('Date End')

#file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

# if file_upload is not None:
#     data = pd.read_csv(file_upload)
#     predictions = predict_model(estimator=model,data=data)
#     st.write(predictions)
    
if st.button('Predict'):
    predictions = my_forecast()
    #st.dataframe(p2)
    st.write(predictions)
    #arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots()
    #ax.hist(arr, bins=20)
    
    
    final_df2 = predictions[['date','Label']]
    final_df2.rename(columns={'Label': 'Forecast'},inplace=True)
    #final_df2 = p2[['date','Label','sales']]
    
    fig, ax = plt.subplots(1,1,figsize=(10,8))
    df_plot2 = final_df2.set_index('date')
    df_plot2[['Forecast']].plot(ax=ax,linewidth=2,legend=False)
    
    # df_plot2[['Label','sales']].plot(ax=ax,linewidth=2)
    # ax.fill_between(df_plot.index,
    #                 df_plot['auto_arima_season_length-12_lo-80'],
    #                 df_plot['auto_arima_season_length-12_hi-80'],
    #                 alpha=0.35,
    #                 color='green',
    #                 label = 'auto_arima_level_80'
    
    # )
    
    # ax.fill_between(df_plot.index,
    #                 df_plot['auto_arima_season_length-12_lo-95'],
    #                 df_plot['auto_arima_season_length-12_hi-95'],
    #                 alpha=0.2,
    #                 color='green',
    #                 label = 'auto_arima_level_95'
    
    # )
    
    ax.set_title('Crypto Forecast of '+crypto_selected,fontsize=22)
    ax.set_ylabel('Closing Price Forecast',fontsize=20)
    ax.set_xlabel('Date',fontsize=20)
    #ax.legend(prop={'size':15})
    ax.grid()    
    st.pyplot(fig)
    #prediction = predict_quality(model, features_df)
    
    #st.write(' Based on feature values, your wine quality is '+ str(prediction))
