import pandas as pd
import numpy as np
import joblib
import streamlit as st

# load ml model
model = joblib.load(open("linear_regression_model (1).pkl", 'rb'))
st.title('Sales Prediction App')

# input
Tv = st.number_input("TV Adv budget", min_value=0.0)
Radio = st.number_input("Radio Adv budget", min_value=0.0)
Newspaper = st.number_input("Newspaper Adv budget", min_value=0.0)

# prediction
if st.button('Predict Sales'):
    input_data = np.array([[Tv, Radio, Newspaper]])
    prediction = model.predict(input_data)[0]

    st.success(f'Predicted Sales: {prediction:.2f}')
