

import streamlit as st
st.set_page_config(
    page_title="Data collection",       # Optional: Icon der Seite
    layout="centered"
)
import pandas as pd
import numpy as np
import joblib  # Zum Laden gespeicherter Modelle
import matplotlib.pyplot as plt
import seaborn as sns
import os




st.title("💾 Data collection")


st.markdown("""
#### **Goal**: Collect data on beach volleyball matches and players to analyze whether weather conditions significantly impact game outcomes.

### Procedure:
1. Data sources:
    - Collect beach volleyball data from the FIVB-VIS system, including tournaments, matches, players, and match statistics.
    - Merge and clean four tables to create a structured dataset.
2. Adding the Coordinates:
    - Extract tournament locations.
    - Use OpenCage (OpenStreetMap) to retrieve geographic coordinates.  
3. Adjusting Match Times:
    - Convert all time data to UTC.
    - Adjust time zones based on location coordinates.
4. Extracting Weather Data:
    - Retrieve historical weather data for each match based on location and time.
    - Use the Open-Meteo API to obtain relevant weather information.
5. Compiling the Dataset:
    - Refine multiple variables, including team names and player details.
    - Prepare a comprehensive dataset for analysis.
""")
