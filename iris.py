#import library needed
import pandas as pd
import streamlit as st 
import pickle
import numpy as np

# Load the best model
with open('iris_rf.pkl', 'rb') as f:
    model = pickle.load(f)

species = ['setosa', 'versicolor', 'virginica']

st.title("Pengklasifikasian Bunga IRis")
st.write("Mengklasifikasi Bunga Iris menjadi 3 jenis secara akurat")

# Creating Sidebar for inputs
st.sidebar.title("Inputs")
sepal_length = st.sidebar.number_input("panjang kelopak (cm)", 4.3, 7.9, 5.0)
sepal_width = st.sidebar.number_input("lebar kelopak (cm)", 2.0, 4.4, 3.6)
petal_length = st.sidebar.number_input("panjang helai (cm)", 1.0, 6.9, 1.4)
petal_width = st.sidebar.number_input("lebar helai (cm)", 0.1, 2.5, 0.2)

# Button to trigger prediction
if st.button("Prediksi"):
# Getting Prediction from model
    inp = np.array([sepal_length, sepal_width, petal_length, petal_width])
    inp = np.expand_dims(inp, axis=0)
    prediction = model.predict(inp)

# Show Results when the button is clicked
    result = species[np.argmax(prediction)]
    st.write("**Bunganya masuk kelompok bunga " + result )
    st.image('result')
