import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import joblib  # Zum Laden gespeicherter Modelle
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
from utils.data import load_data, get_avg_servepoints_by_tournament

df = load_data()


# Setze den gewünschten Seitentitel, das Icon und Layout (optional)
st.set_page_config(
    page_title="Comparison of Beachvolleyball Players",  # Hier stellst du den angezeigten Namen ein
    #page_icon=":smile:",                # Optional: Icon der Seite
    layout="centered"
)
st.title("Comparison of Beachvolleyball Players")

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

#selected_gender = st.sidebar.selectbox("Choose Gender:", df["Gender_x"].unique())

# Filter: Saison. Annahme: Die Spalte "Saison" enthält die Saisoninformationen.
#Season umcodieren
# df['Season'] = df['Season'].astype(int)
# selected_season = st.sidebar.selectbox("Choose Season:", df["Season"].unique())

# **Für die Team-Auswahl:** 
# Filtere nur nach Gender (nicht nach Saison), damit die Teamliste konstant bleibt.
df_team_selection = df[df["Gender_x"] == selected_gender]
spieler_liste = df_team_selection["Full_Name"].unique()

# Zwei einzelne Teams zur Auswahl anbieten
spieler1 = st.selectbox("Choose Player 1:", spieler_liste, key="spieler1")
spieler2 = st.selectbox("Choose Player 2:", spieler_liste, key="spieler2")

st.markdown("<br>" * 2, unsafe_allow_html=True)
st.write('The values are the average of all played games by one player')


# Sidebar: Season Filter (wird später nur für die Vergleichsdaten genutzt)
# Es ist sinnvoll, die Seasons zu sortieren und ggf. als Integer darzustellen.
season_options = sorted({int(s) for s in df["Season"].unique()})
selected_season = st.sidebar.selectbox("Choose Season", season_options)
#df_filtered_all = df[(df["Gender_x"] == selected_gender) & (df["Season"] == selected_season)]

# Annahme: Die CSV enthält eine Spalte "Spieler" zur Identifikation der Spieler.
# spieler_liste = df_filtered_all["Full_Name"].unique()

# #Zwei einzelne Spieler zur Auswahl anbieten:
# spieler1 = st.selectbox("Choose Player 1:", spieler_liste, key="spieler1")
# spieler2 = st.selectbox("Choose Player 2:", spieler_liste, key="spieler2")

# Wenn beide Spieler ausgewählt wurden:
if spieler1 and spieler2:
    df_spieler1 = df[(df["Gender_x"] == selected_gender) &
                      (df["Full_Name"] == spieler1) &
                      (df["Season"] == selected_season)]
    df_spieler2 = df[(df["Gender_x"] == selected_gender) &
                      (df["Full_Name"] == spieler2) &
                      (df["Season"] == selected_season)]
    # Liste der Kennzahlen, die verglichen werden sollen. 
    # Passe diese Liste an deine tatsächlichen Spaltennamen im Datensatz an.
    kennzahlen = ["ServePoint", "DigTotal", "SpikeFault", 'ServeFault', 'BlockPoint', 'ReceptionFault']

    # Falls es eventuell mehrere Beobachtungen pro Spieler gibt, berechnen wir den Durchschnitt.
    # Ist pro Spieler aber nur eine Zeile hinterlegt, kann dieser Schritt auch entfallen.
    stats_spieler1 = df_spieler1[kennzahlen].mean()
    stats_spieler2 = df_spieler2[kennzahlen].mean()

    # Erstelle ein DataFrame, das sich gut für Plotly Express eignet (Long-Format):
    daten = {
        "keyfigure": kennzahlen * 2,  # Wiederhole die Liste der Kennzahlen je Spieler
        "value": list(stats_spieler1.values) + list(stats_spieler2.values),
        "player": [spieler1] * len(kennzahlen) + [spieler2] * len(kennzahlen)
    }
    df_plot = pd.DataFrame(daten)

    # Erstelle ein gruppiertes Balkendiagramm (horizontal, damit die Kennzahlen in den Zeilen stehen)
    fig = px.bar(
        df_plot,
        x="value",
        y="keyfigure",
        color="player",
        barmode="group",
        orientation="h",
        title="Comparison of players"
    )

    st.plotly_chart(fig)
