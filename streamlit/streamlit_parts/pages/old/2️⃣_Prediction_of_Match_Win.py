
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
    "wind_speed": st.sidebar.slider("Wind Speed", float(df_filtered["wind_speed_10m"].min()), float(df_filtered["wind_speed_10m"].max())),
    "rain": st.sidebar.slider("Rain", float(df_filtered["rain"].min()), float(df_filtered["rain"].max())),
    "wind_gusts": st.sidebar.slider("Wind Gusts", float(df_filtered["wind_gusts_10m"].min()), float(df_filtered["wind_gusts_10m"].max())),
    #"TeamFault_team": st.sidebar.slider("TeamFault Team", int(df["TeamFault_team"].min()), int(df["TeamFault_team"].max())),
    "Team1": st.sidebar.selectbox("Team 1", df_filtered["Team1"].unique()),  # Auswahlbox für Teams
    "Team2": st.sidebar.selectbox("Team 2", df_filtered["Team2"].unique())   # Auswahlbox für Teams

    }
elif model_choice =='Match-Win Prediction with weather impact (Gradient Boosting)':
    input_data = {
        "temperature": st.sidebar.slider("Temperature", float(df_filtered["temperature_2m"].min()), float(df_filtered["temperature_2m"].max())),
        "wind_speed": st.sidebar.slider("Wind Speed", float(df_filtered["wind_speed_10m"].min()), float(df_filtered["wind_speed_10m"].max())),
        "rain": st.sidebar.slider("Rain", float(df_filtered["rain"].min()), float(df_filtered["rain"].max())),
        "wind_gusts": st.sidebar.slider("Wind Gusts", float(df_filtered["wind_gusts_10m"].min()), float(df_filtered["wind_gusts_10m"].max())),
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


# Vorhersage und Wahrscheinlichkeitsberechnung erst starten, wenn der Button geklickt wurde
if st.sidebar.button("Show Prediction"):

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
# # Definiere die Wettervariablen, die du untersuchen möchtest
#if st.sidebar.button("Show influence of weather variables"):
#     weather_vars = ["temperature_2m", "wind_speed_10m", "rain", "wind_gusts_10m"]

#     # Für jeden Wetterparameter legen wir einen Wertebereich fest. Hier:    
#     weather_ranges = {}
#     for var in weather_vars:
#         # Erzeuge 50 gleich verteilte Werte von min bis max (aus df_filtered, damit es zu deinen Daten passt)
#         weather_ranges[var] = np.linspace(df_filtered[var].min(), df_filtered[var].max(), num=50)

# # Erstelle konstanten Input auf Basis der aktuellen Benutzereingaben
# # Wir nehmen an, dass input_data bereits aus den Sidebar-Widgets erzeugt wurde.
# # Um den Einfluss der Wettervariablen isoliert zu betrachten, nutzen wir dieselbe Eingabe,
# # und überschreiben dann jeweils nur den aktuellen Wetterwert.
#     constant_input = input_data.copy()

# # Erstelle eine Plotly-Subplot-Figur – hier in einem 2x2 Layout (anpassbar, wenn du mehr/weniger Wettervariablen hast)
#     fig = make_subplots(rows=2, cols=2, subplot_titles=weather_vars)

# # Jetzt iterieren wir über die Wettervariablen und berechnen für jeden Wertebereich die Vorhersage
#     for i, var in enumerate(weather_vars):
#         predictions = []  # Hier sammeln wir die Gewinnwahrscheinlichkeiten
#         xvals = weather_ranges[var]  # Die Werte, die wir für die aktuelle Variable durchlaufen
#         for val in xvals:
#             # Kopiere den konstanten Input und setze den aktuellen Wetterwert
#             new_input = constant_input.copy()
#             new_input[var] = val

#         # Erstelle ein DataFrame für die Vorhersage
#             new_input_df = pd.DataFrame([new_input], columns=model.feature_names_in_)

#             # Berechne die Gewinnwahrscheinlichkeit; wir nehmen an, dass Team 1 gewinnt, wenn Prediction 1 liefert
#             win_prob = model.predict_proba(new_input_df)[0][1]
#             predictions.append(win_prob)

#         # Bestimme die Position des Plots im Layout (2 Spalten)
#         row = i // 2 + 1
#         col = i % 2 + 1
#         # Füge die Linie dem Plot hinzu
#         fig.add_trace(
#             go.Scatter(x=xvals, y=predictions, mode='lines+markers', name=var),
#             row=row, col=col
#         )
#         fig.update_xaxes(title_text=var, row=row, col=col)
#         fig.update_yaxes(title_text="Win Probability", row=row, col=col)

#     fig.update_layout(
#         title_text="Einfluss der Wettervariablen auf die Gewinnwahrscheinlichkeit",
#         height=700
#     )

#     st.plotly_chart(fig)  
from plotly import graph_objects as go  
from plotly import graph_objects as go  
if st.sidebar.button("Show influence of weather variables"):

    var_labels = {
        "temperature_2m": "Temperature (°C)",
        "wind_speed_10m": "Wind Speed (m/s)",
        "wind_gusts_10m": "Wind Gusts (m/s)",
        "rain": "Rain (mm)"
    }

    weather_vars = ["temperature_2m", "wind_speed_10m", "rain", "wind_gusts_10m"]

    # Für jede Wettervariable den Wertebereich bestimmen
    weather_ranges = {}
    for var in weather_vars:
        weather_ranges[var] = np.linspace(df_filtered[var].min(), df_filtered[var].max(), num=50)

    # Erstelle konstanten Input auf Basis der aktuellen Benutzereingaben.
    # input_data wird zuvor über Sidebar-Widgets erzeugt.
    constant_input = input_data.copy()

    # Erstelle eine Plotly-Subplot-Figur – hier in einem 2x2 Layout.
    # Die Subplot-Titel werden hier aus var_labels genommen.
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=[var_labels[var] for var in weather_vars]
    )

    # Jetzt iterieren wir über die Wettervariablen und berechnen für jeden Wertebereich die Vorhersage
    for i, var in enumerate(weather_vars):
        predictions = []  # Hier werden die Gewinnwahrscheinlichkeiten gesammelt
        xvals = weather_ranges[var]  # Array der Werte für die aktuelle Variable

        for val in xvals:
            new_input = constant_input.copy()
            new_input[var] = val

            # Erstelle ein DataFrame, das die Reihenfolge der Features einhält, wie vom Modell erwartet
            new_input_df = pd.DataFrame([new_input], columns=model.feature_names_in_)

            # Berechne die Gewinnwahrscheinlichkeit (z. B. für Team 1, wenn predict_proba()[0][1])
            win_prob = model.predict_proba(new_input_df)[0][1]
            predictions.append(win_prob)

        # Konvertiere die Liste in ein NumPy-Array, um Glättung und Trend leichter berechnen zu können
        predictions = np.array(predictions)

        # Option 1: Glätten mithilfe eines Moving-Average-Filters
        window_size = 5  # Fenstergröße; je nach Bedarf anpassbar
        smoothed_predictions = np.convolve(predictions, np.ones(window_size) / window_size, mode='same')

        # Option 2: Trendlinie mittels linearer Regression
        coeffs = np.polyfit(xvals, predictions, 1)  # Liniearen Trend (Grad 1) berechnen
        trend_line = np.polyval(coeffs, xvals)

        # Bestimme Position im 2x2 Subplot-Gitter
        row = i // 2 + 1
        col = i % 2 + 1

        # Füge die geglättete Kurve hinzu
        fig.add_trace(
            go.Scatter(x=xvals, y=smoothed_predictions, mode='lines',
                       name=f'{var} (smoothed)', line=dict(width=2)),
            row=row, col=col
        )

        # Füge die Trendlinie hinzu (dargestellt als gestrichelte Linie)
        fig.add_trace(
            go.Scatter(x=xvals, y=trend_line, mode='lines',
                       name=f'{var} (trend)', line=dict(dash='dash', width=2)),
            row=row, col=col
        )

        # Setze die Achsentitel
        fig.update_xaxes(title_text=var_labels.get(var, var), row=row, col=col)
        fig.update_yaxes(title_text="Win Probability", row=row, col=col)

    fig.update_layout(
        title_text="Influence of weather variables on the probability of winning",
        height=700,
        showlegend=False   # Legende deaktivieren
    )

    st.plotly_chart(fig)


