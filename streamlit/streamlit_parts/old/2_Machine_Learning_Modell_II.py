import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Zum Laden gespeicherter Modelle
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Pfad festlegen
os.chdir('C:/Users/Katharina/Desktop/Weiterbildung/Bootcamp/Bootcamp/Final_project/streamlit_parts')


# Setze den gewünschten Seitentitel, das Icon und Layout (optional)
st.set_page_config(
    page_title="Prediction of individual statistical values of one team",  # Hier stellst du den angezeigten Namen ein
    #page_icon=":smile:",                # Optional: Icon der Seite
    layout="centered"
)

st.title("Prediction of individual statistical values of one team")
st.write("This is the second Machine Learning Model.")

#Daten für SpikeFault
df_Reg_mitW = pd.read_csv('ML_SpikeFault_mitWetter_V2.csv', sep=';')
df_Reg_ohneW = pd.read_csv('ML_SpikeFault_OHNEWetter_V2.csv', sep=';')

#Modelle implementieren

model_choice = st.sidebar.selectbox('Choose the model', ['Spike-Fault with weather impact (Random Forest)', 'Spike-Fault with weather impact (Gradient Boosting)',
                                                        'Spike-Fault without weather impact (Random Forest)', 'Spike-Fault without weather impact (Gradient Boosting)'])

#Lade das Model und die relevanten Daten
if model_choice ==  'Spike-Fault with weather impact (Random Forest)':
    model = joblib.load('RandomForest_SpikeFault_mitWetter.pkl')
    df = df_Reg_mitW
    target_variable = 'SpikeFault'
elif model_choice == 'Spike-Fault with weather impact (Gradient Boosting)': 
    model = joblib.load('GradientBoosting_SpikeFault_mitWetter.pkl')
    df = df_Reg_mitW
    target_variable = 'SpikeFault'

elif model_choice == 'Spike-Fault without weather impact (Random Forest)':
    model = joblib.load('RandomForest_SpikeFault_OHNEWetter.pkl')
    df = df_Reg_ohneW
    target_variable = 'SpikeFault'

else: 
    model = joblib.load('GradientBoosting_SpikeFault_OHNEWetter.pkl')
    df = df_Reg_ohneW
    target_variable = 'SpikeFault'

st.write(f'You have choosen:** {model_choice}')


#---------------------------------------------------------
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
# Anwenden der Filter
# ------------------
# Kombiniere beide Filter: Geschlecht und Turnier-Typ
df_filtered = df[(df["Gender_x"] == selected_gender)]

# Eingabeformular für Benutzer
st.sidebar.header('Choose Variables')

input_data = {}
if model_choice == 'Spike-Fault with weather impact (Random Forest)':
    input_data = {
            "temperature": st.sidebar.slider("Temperature", float(df_filtered["temperature_2m"].min()), float(df_filtered["temperature_2m"].max())),
            "wind_speed": st.sidebar.slider("Wind Speed (10m)", float(df_filtered["wind_speed_10m"].min()), float(df_filtered["wind_speed_10m"].max())),
            "rain": st.sidebar.slider("Rain", float(df_filtered["rain"].min()), float(df_filtered["rain"].max())),
            "wind_gusts": st.sidebar.slider("Wind Gusts (10m)", float(df_filtered["wind_gusts_10m"].min()), float(df_filtered["wind_gusts_10m"].max())),
            "Team": st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique())#,  # Auswahlbox für Teams
            #"Team2": st.sidebar.selectbox("Team 2", df["Team2"].unique())   # Auswahlbox für Teams

    }
elif model_choice =='Spike-Fault with weather impact (Gradient Boosting)':
    input_data = {
            "temperature": st.sidebar.slider("Temperature", float(df_filtered["temperature_2m"].min()), float(df_filtered["temperature_2m"].max())),
            "wind_speed": st.sidebar.slider("Wind Speed (10m)", float(df_filtered["wind_speed_10m"].min()), float(df_filtered["wind_speed_10m"].max())),
            "rain": st.sidebar.slider("Rain", float(df_filtered["rain"].min()), float(df_filtered["rain"].max())),
            "wind_gusts": st.sidebar.slider("Wind Gusts (10m)", float(df_filtered["wind_gusts_10m"].min()), float(df_filtered["wind_gusts_10m"].max())),
            "Team1": st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique())
    }
elif model_choice =='Spike-Fault without weather impact (Random Forest)':
    input_data = {
            #"SpikePoint": st.sidebar.slider("SpikePoint", int(df["SpikePoint"].min()), int(df["SpikePoint"].max())),
            "ServeFault": st.sidebar.slider("ServeFault", int(df_filtered["ServeFault"].min()), int(df_filtered["ServeFault"].max())),
            "ServePoint": st.sidebar.slider("ServePoint", int(df_filtered["ServePoint"].min()), int(df_filtered["ServePoint"].max())),
            "ServeTotal": st.sidebar.slider("ServeTotal", int(df_filtered["ServeTotal"].min()), int(df_filtered["ServeTotal"].max())),
            "BlockPoint": st.sidebar.slider("BlockPoint", int(df_filtered["BlockPoint"].min()), int(df_filtered["BlockPoint"].max())),
            "BlockTotal": st.sidebar.slider("BlockTotal", int(df_filtered["BlockTotal"].min()), int(df_filtered["BlockTotal"].max())),
            "Team1": st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique())
    }
elif model_choice =='Spike-Fault without weather impact (Gradient Boosting)':
    input_data = {
            #"SpikePoint": st.sidebar.slider("SpikePoint", int(df["SpikePoint"].min()), int(df["SpikePoint"].max())),
            "ServeFault": st.sidebar.slider("ServeFault", int(df_filtered["ServeFault"].min()), int(df_filtered["ServeFault"].max())),
            "ServePoint": st.sidebar.slider("ServePoint", int(df_filtered["ServePoint"].min()), int(df_filtered["ServePoint"].max())),
            "ServeTotal": st.sidebar.slider("ServeTotal", int(df_filtered["ServeTotal"].min()), int(df_filtered["ServeTotal"].max())),
            "BlockPoint": st.sidebar.slider("BlockPoint", int(df_filtered["BlockPoint"].min()), int(df_filtered["BlockPoint"].max())),
            "BlockTotal": st.sidebar.slider("BlockTotal", int(df_filtered["BlockTotal"].min()), int(df_filtered["BlockTotal"].max())),
            "Team1": st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique())
    }


# Konvertiere Eingabe zu DataFrame
input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)

#Vorhersage kommt erst, wenn Benutzer seine Eingaben bestätigt
if st.sidebar.button("Show Prediction"):#anpassen
    # Generate prediction
    prediction = model.predict(input_df)
    st.write(f"**Predicted Performance:** {prediction[0]:.2f}")

    # Now display the selected filter values and input values.
    st.write("### Selected Filters and Input Values:")
    st.write(f"**Gender:** {selected_gender_display}")
    # Loop through input_data and display each key: value pair
    for key, value in input_data.items():
        st.write(f"**{key}:** {value}")
