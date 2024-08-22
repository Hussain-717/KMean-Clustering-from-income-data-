import joblib
import pandas as pd 
from  sklearn.cluster import KMeans
import streamlit as st
from matplotlib import pyplot as plt

km = joblib.load("model.pkl")
st.title("KMeans Clustering")
st.write("Enter following Values")

Age = st.number_input("Enter Your Age")
Income = st.number_input("Enter your Salary")

input_data = {
    "Age" : [Age], 'Income($)' : [Income]
}
data = pd.DataFrame(input_data)
result = km.predict(data)

if st.button("Calculate"):
    st.write("Total Number of clusters: 3")
    st.write(f"You belong to {result+1} cluster")
st.title("Plotting Data") 
img = plt.imread('cluster_plot.png')  
st.image(img, use_column_width=True)
