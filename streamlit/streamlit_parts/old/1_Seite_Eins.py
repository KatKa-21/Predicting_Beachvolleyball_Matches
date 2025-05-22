

import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Zum Laden gespeicherter Modelle
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Setze den gewünschten Seitentitel, das Icon und Layout (optional)
st.set_page_config(
    page_title="Predict Match Win",  # Hier stellst du den angezeigten Namen ein
    #page_icon=":smile:",                # Optional: Icon der Seite
    layout="centered"
)
Pfad festlegen
os.chdir('C:/Users/Katharina/Desktop/Weiterbildung/Bootcamp/Bootcamp/Final_project/streamlit_parts')


#Daten für MatchWin
df_Classif_mitW = pd.read_csv("ML_MatchWin_Weather2.csv", sep=';')
df_Classif_ohneW = pd.read_csv("ML_MatchWin_OHNEWeather_V2.csv", sep=';')


# Streamlit-Titel
st.title("Machine learning models for estimating a match win")
st.subheader('Estimating the outcome of a beachvolleyball game')
st.write("In the following, you can choose between different models to estimate the outcome of a beach volleyball match between two freely selectable teams. As an additional difference, you can choose whether the weather conditions should be taken into account or not.")



#Modelle implementieren

model_choice = st.sidebar.selectbox('Choose the model', ['Match-Win Prediction with weather impact (Random Forest)', 'Match-Win Prediction with weather impact (Gradient Boosting)', 
                                                        'Match-Win Prediction without weather impact (Random Forest)', 'Match-Win Prediction without weather impact (Gradient Boosting)'])

#Lade das Model und die relevanten Daten
if model_choice == 'Match-Win Prediction with weather impact (Random Forest)':
    model = joblib.load('random_forest_model.pkl')
    df = df_Classif_mitW
    target_variable = 'match_win'

elif model_choice == 'Match-Win Prediction with weather impact (Gradient Boosting)':
    model = joblib.load('GradientBoosting_model.pkl')
    df = df_Classif_mitW
    target_variable = 'match_win'

elif model_choice == 'Match-Win Prediction without weather impact (Random Forest)':
    model = joblib.load('random_forest_model_ohneWetter.pkl')
    df = df_Classif_ohneW
    target_variable = 'match_win'

else:
    model = joblib.load('GradientBoosting_model_ohneWetter.pkl')
    df = df_Classif_ohneW
    target_variable = 'match_win'


st.write(f'You have choosen: {model_choice}')

# Lade das ML-Modell (Beispiel: Random Forest)
#modelRF = joblib.load("random_forest_model.pkl")

###########################################################
#Filter Geschlecht
###########################################################
#Gender neu definieren
df['Gender_x'] = df['Gender_x'].astype(int)

#Mapping
gender_mapping = {
    0: 'male',
    1: 'female'
}
original_gender_values = df['Gender_x'].unique()
display_options = [gender_mapping.get(g, g) for g in original_gender_values]

selected_gender_display = st.sidebar.selectbox("Choose Gender", display_options)

# Um das inverse Mapping zu erstellen, damit wir den Originalwert erhalten:
inverse_mapping = {v: k for k, v in gender_mapping.items()}
selected_gender = inverse_mapping.get(selected_gender_display, selected_gender_display)

# ------------------
# Filter: Tunier-Typ
# ------------------
# Angenommen, der Turnier-Typ wird in der Spalte "Type" gespeichert:
selected_tournament = st.sidebar.selectbox("Choose Tournament Type", df["Type"].unique())


# ------------------
# Anwenden der Filter
# ------------------
# Kombiniere beide Filter: Geschlecht und Turnier-Typ
df_filtered = df[(df["Gender_x"] == selected_gender) & (df["Type"] == selected_tournament)]

# Optional: Zeige die gefilterten Daten in der App an, um die Auswahl zu kontrollieren
#st.write("Filtered DataFrame", df_filtered.head())


# Eingabeformular für Benutzer
st.sidebar.header('Choose Variables')

input_data = {}
if model_choice == 'Match-Win Prediction with weather impact (Random Forest)':
    input_data = {
    #             "SpikeFault": st.sidebar.slider("SpikeFault", int(df["SpikeFault"].min()), int(df["SpikeFault"].max())),
    # "SpikePoint": st.sidebar.slider("SpikePoint", int(df["SpikePoint"].min()), int(df["SpikePoint"].max())),
    # "ServeFault": st.sidebar.slider("ServeFault", int(df["ServeFault"].min()), int(df["ServeFault"].max())),
    # "ServePoint": st.sidebar.slider("ServePoint", int(df["ServePoint"].min()), int(df["ServePoint"].max())),
    # "ServeTotal": st.sidebar.slider("ServeTotal", int(df["ServeTotal"].min()), int(df["ServeTotal"].max())),
    # "BlockPoint": st.sidebar.slider("BlockPoint", int(df["BlockPoint"].min()), int(df["BlockPoint"].max())),
    # "BlockTotal": st.sidebar.slider("BlockTotal", int(df["BlockTotal"].min()), int(df["BlockTotal"].max())),
   # "DigTotal": st.sidebar.slider("DigTotal", int(df["DigTotal"].min()), int(df["DigTotal"].max())),
    #"ReceptionFault": st.sidebar.slider("ReceptionFault", int(df["ReceptionFault"].min()), int(df["ReceptionFault"].max())),
    #"SpikeTotal": st.sidebar.slider("SpikeTotal", int(df["SpikeTotal"].min()), int(df["SpikeTotal"].max())),
    "temperature": st.sidebar.slider("Temperature", float(df_filtered["temperature_2m"].min()), float(df_filtered["temperature_2m"].max())),
   # "precipitation": st.sidebar.slider("Precipitation", float(df["precipitation"].min()), float(df["precipitation"].max())),
    "wind_speed": st.sidebar.slider("Wind Speed (10m)", float(df_filtered["wind_speed_10m"].min()), float(df_filtered["wind_speed_10m"].max())),
    "rain": st.sidebar.slider("Rain", float(df_filtered["rain"].min()), float(df_filtered["rain"].max())),
    "wind_gusts": st.sidebar.slider("Wind Gusts (10m)", float(df_filtered["wind_gusts_10m"].min()), float(df_filtered["wind_gusts_10m"].max())),
    #"TeamFault_team": st.sidebar.slider("TeamFault Team", int(df["TeamFault_team"].min()), int(df["TeamFault_team"].max())),
    "Team1": st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique()),  # Auswahlbox für Teams
    "Team2": st.sidebar.selectbox("Team 2", df_filtered["Team2"].unique())   # Auswahlbox für Teams

    }
elif model_choice =='Match-Win Prediction with weather impact (Gradient Boosting)':
    input_data = {
        "temperature": st.sidebar.slider("Temperature", float(df_filtered["temperature_2m"].min()), float(df_filtered["temperature_2m"].max())),
        "wind_speed": st.sidebar.slider("Wind Speed (10m)", float(df_filtered["wind_speed_10m"].min()), float(df_filtered["wind_speed_10m"].max())),
        "rain": st.sidebar.slider("Rain", float(df_filtered["rain"].min()), float(df_filtered["rain"].max())),
        "wind_gusts": st.sidebar.slider("Wind Gusts (10m)", float(df_filtered["wind_gusts_10m"].min()), float(df_filtered["wind_gusts_10m"].max())),
        "Team1": st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique()),  # Auswahlbox für Teams
        "Team2": st.sidebar.selectbox("Team 2", df_filtered["Team2"].unique())   # Auswahlbox für Teams


    }
elif model_choice =='Match-Win Prediction without weather impact (Random Forest)':
    input_data = {
            "SpikePoint": st.sidebar.slider("SpikePoint", int(df_filtered["SpikePoint"].min()), int(df_filtered["SpikePoint"].max())),
            "ServeFault": st.sidebar.slider("ServeFault", int(df_filtered["ServeFault"].min()), int(df_filtered["ServeFault"].max())),
            "ServePoint": st.sidebar.slider("ServePoint", int(df_filtered["ServePoint"].min()), int(df_filtered["ServePoint"].max())),
            "ServeTotal": st.sidebar.slider("ServeTotal", int(df_filtered["ServeTotal"].min()), int(df_filtered["ServeTotal"].max())),
            "BlockPoint": st.sidebar.slider("BlockPoint", int(df_filtered["BlockPoint"].min()), int(df_filtered["BlockPoint"].max())),
            "BlockTotal": st.sidebar.slider("BlockTotal", int(df_filtered["BlockTotal"].min()), int(df_filtered["BlockTotal"].max())),
            "Team1": st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique()),  # Auswahlbox für Teams
            "Team2": st.sidebar.selectbox("Team 2", df_filtered["Team2"].unique())   # Auswahlbox für Teams
    }
else:
    input_data = {
            "SpikePoint": st.sidebar.slider("SpikePoint", int(df_filtered["SpikePoint"].min()), int(df_filtered["SpikePoint"].max())),
            "ServeFault": st.sidebar.slider("ServeFault", int(df_filtered["ServeFault"].min()), int(df_filtered["ServeFault"].max())),
            "ServePoint": st.sidebar.slider("ServePoint", int(df_filtered["ServePoint"].min()), int(df_filtered["ServePoint"].max())),
            "ServeTotal": st.sidebar.slider("ServeTotal", int(df_filtered["ServeTotal"].min()), int(df_filtered["ServeTotal"].max())),
            "BlockPoint": st.sidebar.slider("BlockPoint", int(df_filtered["BlockPoint"].min()), int(df_filtered["BlockPoint"].max())),
            "BlockTotal": st.sidebar.slider("BlockTotal", int(df_filtered["BlockTotal"].min()), int(df_filtered["BlockTotal"].max())),
            "Team1": st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique()),  # Auswahlbox für Teams
            "Team2": st.sidebar.selectbox("Team 2", df_filtered["Team2"].unique())   # Auswahlbox für Teams
    }



# Konvertiere Eingabe zu DataFrame
input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)

#Vorhersage kommt erst, wenn Benutzer seine Eingaben bestätigt
if st.sidebar.button("Show Prediction"):
    # Erzeuge die Vorhersage
    prediction = model.predict(input_df)
    # Anzeige der Vorhersage, angepasst an den Modelltyp
    if target_variable == 'match_win':
        result_text = '🏆 Team 1 gewinnt!' if prediction[0] == 1 else '❌ Team 1 verliert!'
        st.write(f"**Vorhersage:** {result_text}")


# if st.sidebar.button("Show Prediction"):
#     # Generate prediction
#     prediction = model.predict(input_df)
#     st.write(f"**Predicted Performance:** {prediction[0]:.2f}")

#     # Now display the selected filter values and input values.
#     st.write("### Selected Filters and Input Values:")
#     st.write(f"**Gender:** {selected_gender_display}")
#     # Loop through input_data and display each key: value pair
#     for key, value in input_data.items():
#         st.write(f"**{key}:** {value}")
