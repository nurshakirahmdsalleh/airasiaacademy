import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression

st.write("""
# Sales
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0, 200, 10)
    Radio = st.sidebar.slider('Radio', 0, 100, 10)
    Newspaper = st.sidebar.slider('Newspaper', 0, 100, 10)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

dfdata = pd.read_csv("Advertising.csv")
dfdata = dfdata.drop(['Unnamed: 0'],axis=1)
X = dfdata.drop(['Sales'],axis=1)
Y = dfdata.Sales

clf = LinearRegression()
clf.fit(X, Y)

prediction = clf.predict(df)

st.subheader('Prediction')
st.write(prediction[0])
#st.write(df.style.format({"Predictions": "{:.2f}"}))
#st.write(df.style.format("{:.2}"))
st.write(prediction.style.format(precision=2))
