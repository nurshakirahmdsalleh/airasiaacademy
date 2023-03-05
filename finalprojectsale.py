import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression

st.write("""
# Flight
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
prediction_proba = clf.predict_proba(df)

#st.subheader('Class labels and their corresponding index number')
#st.write(Y)

st.subheader('Prediction')
#st.write(iris.target_names[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
