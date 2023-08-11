import streamlit as st
import numpy as np
import pandas as pd

# importing package
import turtle

# set the background color
# of the turtle screen
turtle.Screen().bgcolor("orange")

st.header("My first Streamlit App")

st.write('Before you continue, please read the [terms and conditions](https://www.gnu.org/licenses/gpl-3.0.en.html)')
show = st.checkbox('I agree the terms and conditions')
if show:
    st.write(pd.DataFrame({
        'Intplan': ['yes', 'yes', 'yes', 'no'],
        'Churn Status': [0, 0, 0, 1]
    }))
