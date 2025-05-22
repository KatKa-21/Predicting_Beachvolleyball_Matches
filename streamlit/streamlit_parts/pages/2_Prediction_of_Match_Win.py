
import streamlit as st
# Setze den gewünschten Seitentitel, das Icon und Layout (optional)
st.set_page_config(
    page_title="2: Machine Learning - Predict Match Win",  # Hier stellst du den angezeigten Namen ein
    page_icon="🔮"
)

import pandas as pd
import numpy as np
import joblib  # Zum Laden gespeicherter Modelle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#Pfad festlegen

os.chdir('C:/Users/Katharina/Desktop/Weiterbildung/Bootcamp/Bootcamp/Final_project/streamlit_parts')


#Daten für MatchWin
df_Classif_mitW = pd.read_csv("ML_MatchWin_Weather2.csv", sep=';')
df_Classif_ohneW = pd.read_csv("ML_MatchWin_OHNEWeather_V2.csv", sep=';')


# Streamlit-Titel
# Streamlit Title and Subtitle
st.title("🏐 Machine Learning Models for Estimating a Match Win")
st.subheader("Predicting the Outcome of a Beach Volleyball Match")

# Add spacing
st.markdown("<br>", unsafe_allow_html=True)

# Descriptive introduction
st.markdown("""
## 📊 Model Options

Choose between different machine learning models to estimate the total number of *spike faults* in a game:

- 🌀 **With Weather Impact**  
  &nbsp;&nbsp;&nbsp;&nbsp;→ Predicts the *match outcome* of a game, considering weather conditions  
  &nbsp;&nbsp;&nbsp;&nbsp;*(Models: Random Forest & Gradient Boosting)*

- ☀️ **Without Weather Impact**  
  &nbsp;&nbsp;&nbsp;&nbsp;→ Predicts the *match outcome* of a game, without considering weather conditions  
  &nbsp;&nbsp;&nbsp;&nbsp;*(Models: Random Forest & Gradient Boosting)*

---

## 🛠️ Adjust Input Values

Use the filters in the **sidebar** to customize input values and generate more accurate predictions.
""")


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

# Add spacing before showing model choice
st.markdown("<br>", unsafe_allow_html=True)

# Display chosen model with styled text
st.markdown(f"""

---

### ✅ Selected Model  
You have chosen: <span style='font-weight:bold; color:#4CAF50;'>{model_choice}</span>

---

""", unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------------------------------------
###########################################################
#Filter Geschlecht
###########################################################
#Gender neu definieren
df['Gender_x'] = df['Gender_x'].astype(int)

#Mapping
gender_mapping = {
    0: 'Male',
    1: 'Female'
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
#array([ 4, 33,  5, 42, 51, 52]) -> als integer definiert
# type_mapping = {
#         4: 'World Championship',
#         5:'Olympic Games',
#         33: 'World Tour Finals',
#         51:'BeachProTour-Challenger',
#         52:'BeachProTour-Elite',
#         42: 'else'
# }
# original_type_values= df['Type'].unique()
# display_options2 = [type_mapping.get(g, g) for g in original_type_values]

# #selected_tournament = st.sidebar.selectbox("Choose Tournament Type", display_options2)

# inverse_mapping2 = {v: k for k, v in type_mapping.items()}
# selected_type = inverse_mapping2.get(selected_tournament, selected_tournament)

# ------------------
# Anwenden der Filter
# ------------------
# Kombiniere beide Filter: Geschlecht und Turnier-Typ
df_filtered = df[(df["Gender_x"] == selected_gender)]# & (df["Type"] == selected_type)]


# Eingabeformular für Benutzer
st.sidebar.header('Choose Input Values')

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
    "temperature": st.sidebar.slider("Temperature (°C)", float(df_filtered["temperature_2m"].min()), float(df_filtered["temperature_2m"].max())),
    "wind_speed": st.sidebar.slider("Wind Speed (km/h)", float(df_filtered["wind_speed_10m"].min()), float(df_filtered["wind_speed_10m"].max())),
    "rain": st.sidebar.slider("Rain (mm)", float(df_filtered["rain"].min()), float(df_filtered["rain"].max())),
    "wind_gusts": st.sidebar.slider("Wind Gusts (km/h)", float(df_filtered["wind_gusts_10m"].min()), float(df_filtered["wind_gusts_10m"].max())),
    "Team1": st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique()),  # Auswahlbox für Teams
    "Team2": st.sidebar.selectbox("Team 2", df_filtered["Team2"].unique())   # Auswahlbox für Teams

    }
elif model_choice =='Match-Win Prediction with weather impact (Gradient Boosting)':
    input_data = {
        "temperature": st.sidebar.slider("Temperature (°C)", float(df_filtered["temperature_2m"].min()), float(df_filtered["temperature_2m"].max())),
        "wind_speed": st.sidebar.slider("Wind Speed (km/h)", float(df_filtered["wind_speed_10m"].min()), float(df_filtered["wind_speed_10m"].max())),
        "rain": st.sidebar.slider("Rain (mm)", float(df_filtered["rain"].min()), float(df_filtered["rain"].max())),
        "wind_gusts": st.sidebar.slider("Wind Gusts (km/h)", float(df_filtered["wind_gusts_10m"].min()), float(df_filtered["wind_gusts_10m"].max())),
        "Team1": st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique()),  # Auswahlbox für Teams
        "Team2": st.sidebar.selectbox("Team 2", df_filtered["Team2"].unique())   # Auswahlbox für Teams


    }
elif model_choice =='Match-Win Prediction without weather impact (Random Forest)':
    input_data = {
            "SpikePoint": st.sidebar.slider("SpikePoint", int(df_filtered["SpikePoint"].min()), int(df_filtered["SpikePoint"].max())),
            "ServeFault": st.sidebar.slider("ServeFault", int(df_filtered["ServeFault"].min()), int(df_filtered["ServeFault"].max())),
            "ServePoint": st.sidebar.slider("ServePoint", int(df_filtered["ServePoint"].min()), int(df_filtered["ServePoint"].max())),
            "ServeTotal": st.sidebar.slider("ServeTotal", int(df_filtered["ServeTotal"].min()), int(df_filtered["ServeTotal"].max())),
            #"BlockPoint": st.sidebar.slider("BlockPoint", int(df_filtered["BlockPoint"].min()), int(df_filtered["BlockPoint"].max())),
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
            #"BlockPoint": st.sidebar.slider("BlockPoint", int(df_filtered["BlockPoint"].min()), int(df_filtered["BlockPoint"].max())),
            "BlockTotal": st.sidebar.slider("BlockTotal", int(df_filtered["BlockTotal"].min()), int(df_filtered["BlockTotal"].max())),
            "Team1": st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique()),  # Auswahlbox für Teams
            "Team2": st.sidebar.selectbox("Team 2", df_filtered["Team2"].unique())   # Auswahlbox für Teams
    }



# Konvertiere Eingabe zu DataFrame
input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)


# Vorhersage und Wahrscheinlichkeitsberechnung erst starten, wenn der Button geklickt wurde
if st.sidebar.button("🔮 Show Prediction"):

    # Berechne die Vorhersage (z.B. 1: Gewinn, 0: Verlust)
    prediction = model.predict(input_df)
    # Wenn prediction[0] == 1, gewinnt Team 1, andernfalls gewinnt Team 2.
    if target_variable == 'match_win':
        winning_team = input_data["Team1"] if prediction[0] == 1 else input_data["Team2"]
        result_text = f'🏆 {winning_team} win this match!'

        st.markdown(
            f'<p style="font-size:24px; font-weight:bold;">Prediction: {result_text}</p>',
            unsafe_allow_html=True
        )
        #result_text = '🏆 Team 1 wins!' if prediction[0] == 1 else '❌ Team 1 loose!'
        #st.write(f"**Prediction:** {result_text}")

    st.markdown("<br>" , unsafe_allow_html=True)
    # Berechne die Wahrscheinlichkeit für die Gewinnklasse
    winning_probabilities = model.predict_proba(input_df)
    win_probability = winning_probabilities[0][1].round(2)
    #st.write("Predicted Winning Probability for Team 1:", win_probability)

    st.markdown(
    f'<p style="font-size:22px; font-weight:bold;">Predicted Winning Probability for {winning_team}: {win_probability}%</p>',
    unsafe_allow_html=True)

#-------------------------------------
#prediction = model.predict(input_df)
#winning_team = input_data["Team1"] if prediction[0] == 1 else input_data["Team2"]
from plotly import graph_objects as go  
if model_choice in ['Match-Win Prediction with weather impact (Random Forest)', 'Match-Win Prediction with weather impact (Gradient Boosting)']:
    if st.sidebar.button("Show influence of weather variables"):

        st.markdown(f"#### 📈 Weather Impact for {input_data['Team1']}:")

        # Definiere die Wettervariablen und zugehörige Label (auf Englisch, europäische Einheiten)
        # Erstelle konstanten Input aus den Sidebar-Werten
        constant_input = input_data.copy()

        # Verwende nur das in der Sidebar ausgewählte Team, z.B. Team 1
        selected_team = constant_input["Team1"]

        # Definiere die Wettervariablen und zugehörigen Labels (auf Englisch, europäische Einheiten)
        weather_vars = ["temperature_2m", "wind_speed_10m", "rain", "wind_gusts_10m"]
        var_labels = {
            "temperature_2m": "Temperature (°C)",
            "wind_speed_10m": "Wind Speed (km/h)",
            "wind_gusts_10m": "Wind Gusts (km/h)",
            "rain": "Rain (mm)"
        }

        # Für jede Wettervariable den Wertebereich bestimmen aus df_filtered
        weather_ranges = {}
        for var in weather_vars:
            weather_ranges[var] = np.linspace(df_filtered[var].min(), df_filtered[var].max(), num=50)

        from plotly.subplots import make_subplots
        from plotly import graph_objects as go

        # Erstelle eine 2x2 Subplotfigur, eine Achse pro Wettervariable
        fig = make_subplots(
            rows=2, cols=2, 
            subplot_titles=[""]*4, #[var_labels[var] for var in weather_vars],
            vertical_spacing=0.15,    # Mehr Abstand zwischen den Zeilen; Standard ist meist 0.1
            horizontal_spacing=0.2
        )

        window_size = 5  # Fenstergröße für das Glätten mittels Moving-Average

        # Für jede Wettervariable berechnen wir die Vorhersagen basierend auf dem konstanten Input und dem ausgewählten Team
        for i, var in enumerate(weather_vars):
            predictions = []  # Gewinnwahrscheinlichkeiten sammeln
            xvals = weather_ranges[var]

            for val in xvals:
                new_input = constant_input.copy()
                new_input[var] = val

                # Hier wird der Input beibehalten, inklusive des bereits ausgewählten Teams
                new_input_df = pd.DataFrame([new_input], columns=model.feature_names_in_)

                # Berechne die Gewinnwahrscheinlichkeit für den ausgewählten Fall
                win_prob = model.predict_proba(new_input_df)[0][1]
                predictions.append(win_prob)

            predictions = np.array(predictions)

            # Glätten der Vorhersagekurve mittels Moving-Average
            smoothed_predictions = np.convolve(predictions, np.ones(window_size) / window_size, mode='same')

            # Berechne eine Trendlinie via linearer Regression
            coeffs = np.polyfit(xvals, predictions, 1)
            trend_line = np.polyval(coeffs, xvals)

            # Bestimme Position im 2x2 Subplot-Gitter
            row = i // 2 + 1
            col = i % 2 + 1

            # Füge die geglättete Kurve hinzu, mit Hinweis auf das ausgewählte Team
            fig.add_trace(
                go.Scatter(
                    x=xvals, 
                    y=smoothed_predictions, 
                    mode='lines',
                    name=f'{var_labels.get(var)} (smoothed) - {selected_team}', 
                    line=dict(width=2)
                ),
                row=row, col=col
            )

            # Füge die Trendlinie als gestrichelte Linie hinzu
            fig.add_trace(
                go.Scatter(
                    x=xvals, 
                    y=trend_line, 
                    mode='lines',
                    name=f'{var_labels.get(var)} (trend) - {selected_team}', 
                    line=dict(dash='dash', width=2)
                ),
                row=row, col=col
            )

            # Optionale Achsentitel pro Subplot
            fig.update_xaxes(title_text=var_labels.get(var, var), row=row, col=col)
            fig.update_yaxes(title_text="Win Probability", row=row, col=col)

        # Layout des Plots anpassen
        fig.update_layout(
            #title_text="Influence of Weather Variables on Win Probability for selected Team",
            height=700,
            showlegend=False
        )

        st.plotly_chart(fig)



