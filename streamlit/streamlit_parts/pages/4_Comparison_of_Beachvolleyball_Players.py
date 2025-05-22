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
st.title("🏐 Comparison of Beach Volleyball Players")

# Add a brief introduction
st.markdown("""
In this section, you can compare the performance of different beach volleyball players based on their past game data.

Use the filters below to select players and examine their statistics, allowing you to gain deeper insights into their strengths and weaknesses.

🔍 **How to Compare:**
- Select two players from the dropdown lists.
- Adjust additional filters to refine the comparison.
- Explore various performance metrics, such as *spike faults*, *sets won*, and more!

Start comparing and find out which player stands out!
""")

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

# Sidebar
selected_gender_display = st.sidebar.selectbox("Choose Gender", display_options)

# Um das inverse Mapping zu erstellen, damit wir den Originalwert erhalten:
inverse_mapping = {v: k for k, v in gender_mapping.items()}
selected_gender = inverse_mapping.get(selected_gender_display, selected_gender_display)

# 🔹 Season selection: Add "All" option
season_options = ["All"] + sorted({int(s) for s in df["Season"].unique()})
selected_season = st.sidebar.selectbox("Choose Season", season_options)

# 🔹 **Spieler-Liste bleibt stabil** – Nur nach Gender gefiltert, nicht nach Season!
df_team_selection = df[df["Gender_x"] == selected_gender]
spieler_liste = df_team_selection["Full_Name"].unique()

# Zwei einzelne Spieler zur Auswahl anbieten
spieler1 = st.selectbox("Choose Player 1:", spieler_liste, key="spieler1")
spieler2 = st.selectbox("Choose Player 2:", spieler_liste, key="spieler2")

st.markdown("<br>" * 2, unsafe_allow_html=True)
#st.write('The values are the average of all played games by the chosen players')

# 🔹 Apply filters based on selection für den Vergleich der Kennzahlen:
if selected_season == "All":
    # Season-Filter nicht anwenden
    df_filtered_all = df[df["Gender_x"] == selected_gender]
else:
    df_filtered_all = df[(df["Gender_x"] == selected_gender) & (df["Season"] == selected_season)]

# Wenn beide Spieler ausgewählt wurden:
if spieler1 and spieler2:
    if selected_season == "All":
        df_spieler1 = df_filtered_all[df_filtered_all["Full_Name"] == spieler1]
        df_spieler2 = df_filtered_all[df_filtered_all["Full_Name"] == spieler2]
    else:
        df_spieler1 = df_filtered_all[(df_filtered_all["Full_Name"] == spieler1) &
                                      (df_filtered_all["Season"] == selected_season)]
        df_spieler2 = df_filtered_all[(df_filtered_all["Full_Name"] == spieler2) &
                                      (df_filtered_all["Season"] == selected_season)]

    # Liste der Kennzahlen, die verglichen werden sollen
    kennzahlen = ["ServePoint", "DigTotal", "SpikeFault", "ServeFault", "BlockPoint", "ReceptionFault"]

    # Durchschnittswerte der Kennzahlen berechnen, falls es mehrere Beobachtungen pro Spieler gibt
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
        title="Comparison of players",
        text="value"
    )


    # Rundet die Zahlen, die als Text angezeigt werden, auf zwei Nachkommastellen
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', hovertemplate='%{y}: %{x:.2f}', cliponaxis=False)
    fig.update_layout(width = 1800, margin=dict(l=50, r=150, t=50, b=50), xaxis_title="", yaxis_title="",xaxis=dict(showticklabels=False))
    st.plotly_chart(fig)


# Add spacing before the description
st.markdown("<br>" * 2, unsafe_allow_html=True)

# Add explanation about the average values
st.write("""
The values represent the **average** of all played games by the selected players.

This gives you a clearer overview of their overall performance and consistency across various matches.
""")


