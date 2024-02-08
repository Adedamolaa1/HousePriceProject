import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib 
import warnings 
warnings.filterwarnings('ignore')

#import data
data = pd.read_csv('USA_Housing.csv')

#import model
model = joblib.load('linear_regressionToBeSaved (1).pkl')

st.markdown("<h1 style = 'color: #1F4172; text-align: center; font-family: helvetica '>HOUSE PRICE PREDICTION PROJECT</h1>", unsafe_allow_html =True)

st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By GomyCode MTW</h4>", unsafe_allow_html = True)
st.image('pngwing.com (6).png', width = 350, use_column_width = True)
st.markdown("<br>", unsafe_allow_html = True)
st.markdown('<h4 style = "text-align: center; font-family: cursive; font-size; 40px; "> PROJECT OVERVIEW </h4>', unsafe_allow_html = True)
st.markdown('<p>Developing a house price prediction model leveraging various housing attributes such as size, location, and amenities; preprocessing data to handle missing values and encode categorical variables; selecting and training regression algorithms including Linear Regression and Gradient Boosting; evaluating model performance using metrics like Mean Absolute Error and R-squared; and deploying the model for accurate real-world price estimations. </p>', unsafe_allow_html = True)
st.sidebar.image('pngwing.com (7).png', caption = 'Welcome User')
st.dataframe(data, use_container_width = True)

input_choice = st.sidebar.radio('Choose your Input type', ['Slider Input', 'Number Input'])

if input_choice == 'Slider Input':
    area_income = st.sidebar.slider('Average Area Income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    House_age = st.sidebar.slider('Average House Age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())
    room_num = st.sidebar.slider('Average number of rooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Rooms'].max())
    bedrooms = st.sidebar.slider('Number of bedrooms', data['Avg. Area Number of Bedrooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    population = st.sidebar.slider('Area Population', data['Area Population'].min(), data['Area Population'].max())

else:
    area_income = st.sidebar.number_input('Average Area Income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    House_age = st.sidebar.number_input('Average House Age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())
    room_num = st.sidebar.number_input('Average number of rooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Rooms'].max())
    bedrooms = st.sidebar.number_input('Number of bedrooms', data['Avg. Area Number of Bedrooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    population = st.sidebar.number_input('Area Population', data['Area Population'].min(), data['Area Population'].max())

input_vars = pd.DataFrame({'Avg. Area Income': [area_income], 
                           'Avg. Area House Age':[House_age],
                           'Avg. Area Number of Rooms':[room_num], 
                           'Avg. Area Number of Bedrooms':[bedrooms],
                           'Area Population': [population] })

st.markdown("<br>", unsafe_allow_html = True)
st.markdown("<h4 style = 'text-align: center;; color: olive; font-family: helvetica'>User Input Variables</h4>", unsafe_allow_html = True)
st.dataframe(input_vars)

predicted = model.predict(input_vars)
prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push to Predict')
    if pred:
        st.success(f'The predicted price of your house is {predicted}')

with interprete:
    st.header('The Interpretation Of The Model')
    st.write(f'The intercept of the model is: {round(model.intercept_, 2)}')
    st.write(f'A unit change in the average area income causes the price to change by {model.coef_[0]} naira')
    st.write(f'A unit change in the average house age causes the price to change by {model.coef_[1]} naira')
    st.write(f'A unit change in the average number of rooms causes the price to change by {model.coef_[2]} naira')
    st.write(f'A unit change in the average number of bedrooms causes the price to change by {model.coef_[3]} naira')
    st.write(f'A unit change in the average number of populatioin causes the price to change by {model.coef_[4]} naira')
    






#Index(['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       #'Avg. Area Number of Bedrooms', 'Area Population'],
      #dtype='object')
