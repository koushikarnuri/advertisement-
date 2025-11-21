import pandas as pd
import numpy as np
import joblib
import streamlit as st

#lad ml model

model=joblib.load(open("linear_regression_model (1).pkl",'rb'))
st.title('sales prediction app')

#input
TV=st.number_input("TV Adv budget",min_value=0.0)
Radio=st.number_input("Radio Adv budget",min_value=0.0)
Newspaper=st.number_input("Newspaper Adv budget",min_value=0.0)
#pred
if st.button('predict sales'):
	input_data=np.array([[TV,Radio,Newspapers]])
	prediction=model.predict(input_data)[0]
	st.success('predict sales:{prediction:.2f}')