#3_Machine_Learning_Modell_II.py


import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Zum Laden gespeicherter Modelle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
from plotly.subplots import make_subplots
from plotly import graph_objects as go
#Pfad festlegen
os.chdir('C:/Users/Katharina/Desktop/Weiterbildung/Bootcamp/Bootcamp/Final_project/streamlit_parts')


# Setze den gewünschten Seitentitel, das Icon und Layout (optional)
st.set_page_config(
    page_title="Prediction of individual statistical values of one team",  # Hier stellst du den angezeigten Namen ein
    #page_icon=":smile:",                # Optional: Icon der Seite
    layout="centered"
)

st.title("🔍 Prediction of Individual Statistical Values for One Team")

st.markdown("""
## 📊 Model Options

Choose between two different prediction models:

- 🌀 **With Weather Impact**  
  &nbsp;&nbsp;&nbsp;&nbsp;→ Predicts the total number of *spike faults* in a game, considering weather conditions  
  &nbsp;&nbsp;&nbsp;&nbsp;*(Models: Random Forest & Gradient Boosting)*

- ☀️ **Without Weather Impact**  
  &nbsp;&nbsp;&nbsp;&nbsp;→ Predicts the total number of *spike faults* in a game, without considering weather conditions  
  &nbsp;&nbsp;&nbsp;&nbsp;*(Models: Random Forest & Gradient Boosting)*

---

## 🛠️ Adjust Input Values

Use the filters in the **sidebar** to customize input values and generate more accurate predictions.
""")

#Daten für SpikeFault
df_Reg_mitW = pd.read_csv('ML_SpikeFault_mitWetter.csv', sep=';')
df_Reg_ohneW = pd.read_csv('ML_SpikeFault_OHNEWetter.csv', sep=';')


#Modelle implementieren

model_choice = st.sidebar.selectbox('Choose the model', ['Spike-Fault with weather impact (Random Forest)', 'Spike-Fault with weather impact (Gradient Boosting)',
                                                        'Spike-Fault without weather impact (Random Forest)', 'Spike-Fault without weather impact (Gradient Boosting)',
                                                        #'Total-Spikes with weather impact (Random Forest)','Total-Spikes with weather impact (Gradient Boosting)',
                                                        #'Total-Spikes without weather impact (Random Forest)','Total-Spikes without weather impact (Gradient Boosting)'
                                                        ])

#Lade das Model und die relevanten Daten
if model_choice ==  'Spike-Fault with weather impact (Random Forest)':
    model = joblib.load('RandomForest_SpikeFault_mitWetter_NEUEDATEN.pkl')
    df = df_Reg_mitW
    target_variable = 'SpikeFault'
elif model_choice == 'Spike-Fault with weather impact (Gradient Boosting)': 
    model = joblib.load('GradientBoosting_SpikeFault_mitWetter_NEUEDATEN.pkl')
    df = df_Reg_mitW
    target_variable = 'SpikeFault'

elif model_choice == 'Spike-Fault without weather impact (Random Forest)':
    model = joblib.load('RandomForest_SpikeFault_OHNEWetter_NEUEDATEN.pkl')
    df = df_Reg_ohneW
    target_variable = 'SpikeFault'

elif model_choice == 'Spike-Fault without weather impact (Gradient Boosting)': 
    model = joblib.load('GradientBoosting_SpikeFault_OHNEWetter_NEUEDATEN.pkl')
    df = df_Reg_ohneW
    target_variable = 'SpikeFault'

# Add spacing before showing selected model
st.markdown("<br>", unsafe_allow_html=True)

# Display chosen model with styled text
st.markdown(f"""
### ✅ Selected Model  
You have chosen: <span style='font-weight:bold; color:#4CAF50;'>{model_choice}</span>
""", unsafe_allow_html=True)

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
    temperature_val = st.sidebar.slider("Temperature (°C)",  float(df_filtered["temperature_2m"].min()),  float(df_filtered["temperature_2m"].max()))
    wind_speed_val = st.sidebar.slider( "Wind Speed (km/h)", float(df_filtered["wind_speed_10m"].min()), float(df_filtered["wind_speed_10m"].max()))
    wind_gusts_val = st.sidebar.slider( "Wind Gusts (km/h)", float(df_filtered["wind_gusts_10m"].min()), float(df_filtered["wind_gusts_10m"].max()))
    rain_val = st.sidebar.slider("Rain (mm)", float(df_filtered["rain"].min()), float(df_filtered["rain"].max()))

    # Abfrage, ob ein dritter Satz berücksichtigt werden soll
    third_set_selection = st.sidebar.selectbox("Third Set?", options=["No", "Yes"])
    third_set_value = 1 if third_set_selection == "Yes" else 0

    # Falls Third Set = Yes (also third_set_value==1), dann kann der Slider für DurationSet3 angezeigt werden.
    if third_set_value == 1:
        # Filtere den DataFrame basierend auf DurationSet3_indicator
        df_third_filtered = df_filtered[df_filtered["@DurationSet3_indicator"] == third_set_value]
        if not df_third_filtered.empty:
            duration_min = float(df_third_filtered['@DurationSet3'].min())/ 60  # Sekunden in Minuten umwandeln
            duration_max = round(float(df_third_filtered['@DurationSet3'].max()) / 60, 2)        

        else:
            duration_min, duration_max = 0.0, 0.0

        duration_val = st.sidebar.slider('Duration Set 3 (in min)', duration_min, duration_max)
    else:
        # Wenn kein Third Set ausgewählt wurde, setzen wir DurationSet3 automatisch auf 0
        duration_val = 0.0



    # Team-Auswahl (unabhängig vom Third Set Filter – oder auch hier ggf. anpassen)
    team1_selection = st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique())

    # Zusammenstellung der Eingabedaten – beachte, dass die Schlüssel exakt den im Modell verwendeten Spaltennamen entsprechen müssen.
    input_data = {
    "temperature_2m": temperature_val,
    "wind_speed_10m": wind_speed_val,
    "wind_gusts_10m": wind_gusts_val,
    "rain": rain_val,
    "@DurationSet3": duration_val,          # Vom Slider (oder automatisch 0)
    "@DurationSet3_indicator": third_set_value,             # 1 bei Third Set = Yes, sonst 0
    "Team1": team1_selection
    }

elif model_choice =='Spike-Fault with weather impact (Gradient Boosting)':

    temperature_val = st.sidebar.slider("Temperature (°C)",  float(df_filtered["temperature_2m"].min()),  float(df_filtered["temperature_2m"].max()))
    wind_speed_val = st.sidebar.slider( "Wind Speed (km/h)", float(df_filtered["wind_speed_10m"].min()), float(df_filtered["wind_speed_10m"].max()))
    wind_gusts_val = st.sidebar.slider( "Wind Gusts (km/h)", float(df_filtered["wind_gusts_10m"].min()), float(df_filtered["wind_gusts_10m"].max()))
    rain_val = st.sidebar.slider("Rain (mm)", float(df_filtered["rain"].min()), float(df_filtered["rain"].max()))

    # Abfrage, ob ein dritter Satz berücksichtigt werden soll
    third_set_selection = st.sidebar.selectbox("Third Set?", options=["No", "Yes"])
    third_set_value = 1 if third_set_selection == "Yes" else 0

    # Falls Third Set = Yes (also third_set_value==1), dann kann der Slider für DurationSet3 angezeigt werden.
    if third_set_value == 1:
        # Filtere den DataFrame basierend auf DurationSet3_indicator
        df_third_filtered = df_filtered[df_filtered["@DurationSet3_indicator"] == third_set_value]
        if not df_third_filtered.empty:
            duration_min = float(df_third_filtered['@DurationSet3'].min())/ 60  # Sekunden in Minuten umwandeln
            duration_max = round(float(df_third_filtered['@DurationSet3'].max()) / 60, 2)        
        else:
            duration_min, duration_max = 0.0, 0.0

        duration_val = st.sidebar.slider('Duration Set 3 (in min)', duration_min, duration_max)
    else:
        # Wenn kein Third Set ausgewählt wurde, setzen wir DurationSet3 automatisch auf 0
        duration_val = 0.0

    # Team-Auswahl (unabhängig vom Third Set Filter – oder auch hier ggf. anpassen)
    team1_selection = st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique())

    # Zusammenstellung der Eingabedaten – beachte, dass die Schlüssel exakt den im Modell verwendeten Spaltennamen entsprechen müssen.
    input_data = {
    "temperature_2m": temperature_val,
    "wind_speed_10m": wind_speed_val,
    "wind_gusts_10m": wind_gusts_val,
    "rain": rain_val,
    "@DurationSet3": duration_val,          # Vom Slider (oder automatisch 0)
    "@DurationSet3_indicator": third_set_value,             # 1 bei Third Set = Yes, sonst 0
    "Team1": team1_selection
    }




elif model_choice =='Spike-Fault without weather impact (Random Forest)':
    ServeFault_val =  st.sidebar.slider("ServeFault", int(df_filtered["ServeFault"].min()), int(df_filtered["ServeFault"].max()))
    ServeTotal_val =  st.sidebar.slider("ServeTotal", int(df_filtered["ServeTotal"].min()), int(df_filtered["ServeTotal"].max()))
    BlockTotal_val = st.sidebar.slider("BlockTotal", int(df_filtered["BlockTotal"].min()), int(df_filtered["BlockTotal"].max()))

    # Abfrage, ob ein dritter Satz berücksichtigt werden soll
    third_set_selection = st.sidebar.selectbox("Third Set?", options=["No", "Yes"])
    third_set_value = 1 if third_set_selection == "Yes" else 0

    # Falls Third Set = Yes (also third_set_value==1), dann kann der Slider für DurationSet3 angezeigt werden.
    if third_set_value == 1:
        # Filtere den DataFrame basierend auf DurationSet3_indicator
        df_third_filtered = df_filtered[df_filtered["@DurationSet3_indicator"] == third_set_value]
        if not df_third_filtered.empty:
            duration_min = float(df_third_filtered['@DurationSet3'].min())/ 60  # Sekunden in Minuten umwandeln
            duration_max = round(float(df_third_filtered['@DurationSet3'].max()) / 60, 2)        
        else:
            duration_min, duration_max = 0.0, 0.0

        duration_val = st.sidebar.slider('Duration Set 3 (in min)', duration_min, duration_max)
    else:
        # Wenn kein Third Set ausgewählt wurde, setzen wir DurationSet3 automatisch auf 0
        duration_val = 0.0

    # Team-Auswahl (unabhängig vom Third Set Filter – oder auch hier ggf. anpassen)
    team1_selection = st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique())

    # Zusammenstellung der Eingabedaten – beachte, dass die Schlüssel exakt den im Modell verwendeten Spaltennamen entsprechen müssen.
    input_data = {
    "ServeFault": ServeFault_val,
    "ServeTotal": ServeTotal_val,
    "BlockTotal": BlockTotal_val,
    "@DurationSet3": duration_val,          # Vom Slider (oder automatisch 0)
    "@DurationSet3_indicator": third_set_value,             # 1 bei Third Set = Yes, sonst 0
    "Team1": team1_selection
    }


elif model_choice =='Spike-Fault without weather impact (Gradient Boosting)':

    ServeFault_val = st.sidebar.slider("ServeFault", int(df_filtered["ServeFault"].min()), int(df_filtered["ServeFault"].max())),
    ServeTotal_val = st.sidebar.slider("ServeTotal", int(df_filtered["ServeTotal"].min()), int(df_filtered["ServeTotal"].max())),
    BlockTotal_val = st.sidebar.slider("BlockTotal", int(df_filtered["BlockTotal"].min()), int(df_filtered["BlockTotal"].max()))

    # Abfrage, ob ein dritter Satz berücksichtigt werden soll
    third_set_selection = st.sidebar.selectbox("Third Set?", options=["No", "Yes"])
    third_set_value = 1 if third_set_selection == "Yes" else 0

    # Falls Third Set = Yes (also third_set_value==1), dann kann der Slider für DurationSet3 angezeigt werden.
    if third_set_value == 1:
        # Filtere den DataFrame basierend auf DurationSet3_indicator
        df_third_filtered = df_filtered[df_filtered["@DurationSet3_indicator"] == third_set_value]
        if not df_third_filtered.empty:
            duration_min = float(df_third_filtered['@DurationSet3'].min())/ 60  # Sekunden in Minuten umwandeln
            duration_max = round(float(df_third_filtered['@DurationSet3'].max()) / 60, 2)
        else:
            duration_min, duration_max = 0.0, 0.0

        duration_val = st.sidebar.slider('Duration Set3 (in min)', duration_min, duration_max)
    else:
        # Wenn kein Third Set ausgewählt wurde, setzen wir DurationSet3 automatisch auf 0
        duration_val = 0.0

    # Team-Auswahl (unabhängig vom Third Set Filter – oder auch hier ggf. anpassen)
    team1_selection = st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique())

    # Zusammenstellung der Eingabedaten – beachte, dass die Schlüssel exakt den im Modell verwendeten Spaltennamen entsprechen müssen.
    input_data = {
    "ServeFault": ServeFault_val,
    "ServeTotal": ServeTotal_val,
    "BlockTotal": BlockTotal_val,
    "@DurationSet3": duration_val,          # Vom Slider (oder automatisch 0)
    "@DurationSet3_indicator": third_set_value,             # 1 bei Third Set = Yes, sonst 0
    "Team1": team1_selection
    }

# Konvertiere Eingabe zu DataFrame
input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)

# Prediction section – only shown after user confirms input
if st.sidebar.button("🔮 Show Prediction"):
    # Generate prediction
    prediction = model.predict(input_df)

    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Display result with styled formatting
    st.markdown("""
    ### 📈 Prediction Result
    """)
    st.success(f"**Predicted Performance:** {prediction[0]:.2f} spike faults")

    # Optional: Add more spacing below
    st.markdown("<br><br>", unsafe_allow_html=True)



#---------------------------------------------------------------------
if model_choice in ['Spike-Fault with weather impact (Random Forest)', 'Spike-Fault with weather impact (Gradient Boosting)']:
    if st.sidebar.button("Show influence of weather variables"):
        var_labels = {
        "temperature_2m": "Temperature (°C)",
        "wind_speed_10m": "Wind Speed (m/s)",
        "wind_gusts_10m": "Wind Gusts (m/s)",
        "rain": "Rain (mm)"
        }


        base_input = {
        "temperature_2m": st.session_state.get("temperature_val", float(df_filtered["temperature_2m"].mean())),
        "wind_speed_10m": st.session_state.get("wind_speed_val", float(df_filtered["wind_speed_10m"].mean())),
        "wind_gusts_10m": st.session_state.get("wind_gusts_val", float(df_filtered["wind_gusts_10m"].mean())),
        "rain": st.session_state.get("rain_val", float(df_filtered["rain"].mean())),
        "@DurationSet3": st.session_state.get("duration_val", 0.0),
        "@DurationSet3_indicator": st.session_state.get("third_set_value", 0),
        "Team1": st.session_state.get("team1_selection", df_filtered["Team1"].unique()[0])
        }

        # Definiere, welche Wettervariablen du untersuchen möchtest
        weather_vars = ["temperature_2m", "wind_speed_10m", "wind_gusts_10m", "rain"]

        #   Erstelle ein 2x2 Gitter für die Subplots
        fig = make_subplots(rows=2, cols=2)
        #, subplot_titles=[None] * len(weather_vars)
        # Für jede Wettervariable den Wertebereich bestimmen und darin Vorhersagen berechnen
        for i, var in enumerate(weather_vars):
            # Bestimme den Minimal- und Maximalwert aus deinen Daten
            vmin = float(df_filtered[var].min())
            vmax = float(df_filtered[var].max())

            # Erstelle ein Array mit Werten zwischen vmin und vmax (z.B. 100 Punkte)
            values = np.linspace(vmin, vmax, num=100)
            predictions = []

            # Für jeden Wert der aktuellen Wettervariable:
            for val in values:
                # Kopiere den Basis-Datensatz und setze den aktuellen Wetterwert
                current_input = base_input.copy()
                current_input[var] = val

                # Erstelle ein DataFrame, das die Reihenfolge der Features einhält, wie es dein Modell erwartet
                input_df = pd.DataFrame([current_input], columns=model.feature_names_in_)

                # Vorhersage des Modells (Anzahl der Spikes)
                prediction = model.predict(input_df)[0]
                predictions.append(prediction)

            # Option 1: Glätten mittels Moving Average
            window_size = 5  # anpassbar
            smoothed_predictions = np.convolve(predictions, np.ones(window_size) / window_size, mode='same')

            # Option 2: Trendlinie mittels linearer Regression
            coeffs = np.polyfit(values, predictions, 1)
            trend_line = np.polyval(coeffs, values)

            # Bestimme die Position im Subplot-Gitter (2 Spalten)
            row = i // 2 + 1
            col = i % 2 + 1

            # ursprüngliche Glättungskurve plotten:
            fig.add_trace(
                go.Scatter(x=values, y=smoothed_predictions, mode='lines', name=f'{var} (smoothed)'),
                row=row, col=col
            )

            # Trendlinie hinzufügen
            fig.add_trace(
                go.Scatter(x=values, y=trend_line, mode='lines', name=f'{var} (trend)', line=dict(dash='dash')),
                row=row, col=col
            )

            fig.update_xaxes(title_text=var_labels.get(var, var), row=row, col=col)
            fig.update_yaxes(title_text="Predicted Spikes", row=row, col=col)

            #     # Berechne den minimalen und maximalen y-Wert und füge einen zusätzlichen Margin hinzu
            # y_min = np.min(predictions)
            # y_max = np.max(predictions)
            # margin = (y_max - y_min) * 0.2  # 20% extra Platz an der oberen Seite

            # # Füge eine Linie zur entsprechenden Subplot hinzu
            # fig.add_trace(
            #     px.line(x=values, y=predictions).data[0],
            #     row=row, col=col
            #     )
            # # Achsentitel setzen
            # fig.update_xaxes(title_text=var_labels.get(var, var), row=row, col=col)
            # fig.update_yaxes(title_text="Predicted Spikes", row=row, col=col,range=[y_min, y_max + margin])

        # Passe das Layout an (Titel, Abstände, etc.)
        fig.update_layout(
            title_text="Influence of the weather for sum of SpikesFaults",
            height=700, width=900,
            showlegend=False
        )

        st.plotly_chart(fig)






