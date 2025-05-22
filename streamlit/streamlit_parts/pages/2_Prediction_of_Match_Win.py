
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
# Anwenden der Filter
# ------------------
df_filtered = df[(df["Gender_x"] == selected_gender)]# & (df["Type"] == selected_type)]


st.sidebar.header('Choose Input Values')

input_data = {}
if model_choice == 'Match-Win Prediction with weather impact (Random Forest)':
    input_data = {
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
if model_choice in ['Match-Win Prediction with weather impact (Random Forest)', 'Match-Win Prediction with weather impact (Gradient Boosting)']:
    if st.sidebar.button("Show influence of weather variables"):

        st.markdown(f"#### 📈 Weather Impact for {input_data['Team1']}:")

        constant_input = input_data.copy()

        selected_team = constant_input["Team1"]

        weather_vars = ["temperature_2m", "wind_speed_10m", "rain", "wind_gusts_10m"]
        var_labels = {
            "temperature_2m": "Temperature (°C)",
            "wind_speed_10m": "Wind Speed (km/h)",
            "wind_gusts_10m": "Wind Gusts (km/h)",
            "rain": "Rain (mm)"
        }

        weather_ranges = {}
        for var in weather_vars:
            weather_ranges[var] = np.linspace(df_filtered[var].min(), df_filtered[var].max(), num=50)

        

        fig = make_subplots(
            rows=2, cols=2, 
            subplot_titles=[""]*4, 
            vertical_spacing=0.15,    
            horizontal_spacing=0.2
        )

        window_size = 5   

        
        for i, var in enumerate(weather_vars):
            predictions = []  # Gewinnwahrscheinlichkeiten sammeln
            xvals = weather_ranges[var]

            for val in xvals:
                new_input = constant_input.copy()
                new_input[var] = val

                new_input_df = pd.DataFrame([new_input], columns=model.feature_names_in_)

                win_prob = model.predict_proba(new_input_df)[0][1]
                predictions.append(win_prob)

            predictions = np.array(predictions)

            smoothed_predictions = np.convolve(predictions, np.ones(window_size) / window_size, mode='same')

            coeffs = np.polyfit(xvals, predictions, 1)
            trend_line = np.polyval(coeffs, xvals)

            row = i // 2 + 1
            col = i % 2 + 1

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

            fig.update_xaxes(title_text=var_labels.get(var, var), row=row, col=col)
            fig.update_yaxes(title_text="Win Probability", row=row, col=col)

        fig.update_layout(
            #title_text="Influence of Weather Variables on Win Probability for selected Team",
            height=700,
            showlegend=False
        )

        st.plotly_chart(fig)



