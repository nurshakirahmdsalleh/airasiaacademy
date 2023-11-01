import streamlit as st
import pandas as pd
import pickle

st.title("Sales Prediction App")
st.write("This app predicts the sales based on three advertising channel features")

st.sidebar.header('User Input Parameters')

def user_input_features():
    tv = st.sidebar.slider('TV', 0, 300, 0)
    radio = st.sidebar.slider('Radio', 0, 50, 0)
    newspaper = st.sidebar.slider('Newpaper', 0, 50, 0)
    data = {'TV': tv,
            'Radio': radio,
            'Newspaper': newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

loaded_model = pickle.load(open("sales-advertising-model.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Sales Prediction')
st.write(f"{prediction[0]:.2f}")
