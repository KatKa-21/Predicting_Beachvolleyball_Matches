
import streamlit as st
st.set_page_config(
    page_title="My Main Page Title",  # Hier dein benutzerdefinierter Seitentitel
    page_icon=":rocket:",             # Optional, z. B. ein Emoji oder Bild-Link
    layout="centered"                 # Oder "wide", je nach gewünschtem Layout
)


import pandas as pd
import numpy as np
import joblib  # Zum Laden gespeicherter Modelle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from PIL import Image
from io import BytesIO


#Pfad festlegen
os.chdir('C:/Users/Katharina/Desktop/Weiterbildung/Bootcamp/Bootcamp/Final_project/streamlit_parts')



#--------------------------------------------------------------------------------------------------------------------------
# Title
st.title("🏐 Welcome to the Beach Volleyball Machine Learning Experience!")
st.markdown("### Predict, Analyze, and get Informations about the Beachvolleyball World Tour 🏐")


# Introductory text
st.markdown("""
Welcome to this interactive machine learning app all about **Beach Volleyball**!  
Explore how data and predictive models can reveal new insights into player performance, team strategy, and match outcomes.

### What can you do here?
Each section of the app focuses on a different aspect of the Beach Volleyball World Tour — and includes **two machine learning models** per page.

- 🔍 **Data Overview** – Learn how the data was collected and prepared  
- 🔮 **Match Prediction** – Use two models to predict match outcomes  
- 🔮 **Error Forecasting** – Predict the total number of spike errors using different approaches  
- 📊 Compare players based on their average game statistics.
- 📊 Compare different Beachvolleyball teams based on their average game statistics.

Whether you're a fan, coach, or data scientist — dive in and explore how data and machine learning meets beach volleyball!
""")

# Funktion zum Laden von Bildern aus dem Web
def load_image(url):
    resp = requests.get(url)
    return Image.open(BytesIO(resp.content))

# Beispielbilder von Unsplash
img1_url = "https://images.unsplash.com/photo-1723138568659-d35c7680779f?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

# Zeige das einzelne Bild an
st.image(load_image(img1_url),
         #caption="Beachvolleyball 🏐",
         use_container_width=True)
# Show example images side by side
col1, col2 = st.columns(2)

# Footer
st.markdown("---")
st.caption("Created with ❤️ using Streamlit and Machine Learning.")
