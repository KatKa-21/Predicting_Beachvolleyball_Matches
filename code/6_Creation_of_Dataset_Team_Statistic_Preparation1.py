# Creation of Dataset for Team Statistics
import pandas as pd
import os
from pathlib import Path
import glob
import numpy as np
import unicodedata


BeachTeams = pd.read_csv('Beach_Team_Infos.csv', sep=';')


#es ist der Fall das Teams doppelt vorkommen, weil die Namen vertauscht sind
def normalize_string(s):
    """
    Entfernt diakritische Zeichen aus einem String, sodass z.B. "Å" zu "A" wird.
    """
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')

def standardize_names(name_str):
    """
    Teilt den Eingabestring, der zwei Namen enthält (getrennt durch '/'),
    entfernt überflüssige Leerzeichen und sortiert die beiden Namen anhand des
    normalisierten Strings, sodass immer der Name zuerst steht, der alphabetisch
    (unter Ignorieren von Sonderzeichen) früher kommt.
    """
    # Teile den String anhand von '/' und entferne eventuelle Leerzeichen
    parts = [part.strip() for part in name_str.split('/')]
    
    # Sortiere die beiden Namen. Dabei wird der normalisierte, kleingeschriebene Wert 
    # als Schlüssel verwendet, sodass z.B. "Åhman" (-> "ahman") richtiger sortiert wird.
    parts_sorted = sorted(parts, key=lambda x: normalize_string(x).lower())
    
    # Füge die sortierten Namen wieder mit '/' zusammen
    return "/".join(parts_sorted)



def reorder_player_numbers(row):
    """
    Vergleicht den ursprünglichen Namen (gesplittet in zwei Teile) mit der 
    alphabetisch sortierten Version. Stimmen sie nicht überein, wird angenommen,
    dass auch die Player-Daten vertauscht wurden – und es werden die zugehörigen
    numerischen Werte getauscht.
    """
    # Die ursprüngliche Reihenfolge aus der Spalte '@Namen'
    unsorted_names = [name.strip() for name in row["@Name"].split('/')]
    # Berechne die standardisierte (sortierte) Reihenfolge:
    sorted_names = sorted(unsorted_names, key=lambda x: normalize_string(x).lower())
    
    # Falls die Reihenfolge nicht gleich ist, tausche die zugehörigen numerischen Spalten:
    if unsorted_names != sorted_names:
        row["@NoPlayer1"], row["@NoPlayer2"] = row["@NoPlayer2"], row["@NoPlayer1"]
        # Höhen tauschen (vorausgesetzt, es gibt die Spalten "@Player1height" und "@Player2height")
        row["@Player1Height"], row["@Player2Height"] = row["@Player2Height"], row["@Player1Height"]
        # Teamnamen tauschen (vorausgesetzt, es gibt die Spalten "@Player1TeamName" und "@Player2TeamName")
        row["@Player1TeamName"], row["@Player2TeamName"] = row["@Player2TeamName"], row["@Player1TeamName"]

        # Falls weitere Spalten (z.B. playerbezogene Positionen) in diesem Zusammenhang,
        # können diese ebenfalls getauscht werden.
    return row


# 1. Erstelle die Spalte Standard_Namen (für die Gruppierung)
BeachTeams["Standard_Namen"] = BeachTeams["@Name"].apply(standardize_names)

# 2. Wende die Funktion auf jede Zeile an, um invertierte Zuordnungen zu korrigieren.
df_corrected = BeachTeams.apply(reorder_player_numbers, axis=1)



#daten anpassen
BeachTeams_grouped = df_corrected.groupby(['Standard_Namen', '@NoPlayer1', '@NoPlayer2'], as_index=False).agg({
    '@PositionInEntry':  lambda x: x.mean(skipna=True),
    '@Rank':  lambda x: x.mean(skipna=True),
    '@EarnedPointsTeam':  lambda x: x.mean(skipna=True),
    '@EarningsTotalTeam':  lambda x: x.mean(skipna=True),
})


#BeachTeams_grouped1 = BeachTeams_grouped[~BeachTeams_grouped["@Name"].str.contains(r"\?", na=False)]
BeachTeams_grouped1 = BeachTeams_grouped[~BeachTeams_grouped["Standard_Namen"].str.contains(r'[?#"\']', na=False)]



df = BeachTeams_grouped1
# Erstellen der neuen Spalte "Team1":
# Wenn TeamDesignation = A, dann soll TeamAName in Team1 stehen,
# andernfalls (also bei B) TeamBName.
df["Team1"] = np.where(
    df["TeamDesignation"].str.upper() == "A",
    df["@TeamAName"],
    df["@TeamBName"]
)

# Erstellen der Spalte "Team2" als das jeweils andere Team:
df["Team2"] = np.where(
    df["TeamDesignation"].str.upper() == "A",
    df["@TeamBName"],
    df["@TeamAName"]
)



#Namen sortieren
df_BeachTeams = df.copy()
df_BeachTeams['Standard_Namen'] = df['Team1'].apply(standardize_names)
df_BeachTeams["Standard_Namen_team2"] = df_BeachTeams["Team2"].apply(standardize_names)
DatG_test = pd.merge(BeachTeams_grouped1, df_BeachTeams, left_on=['Standard_Namen'], right_on=['Standard_Namen'], how='inner')




