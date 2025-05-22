# # Vorbereitung Machine Learning


import pandas as pd
import os
from pathlib import Path
import glob
import numpy as np
import re

#Daten einlesen
Database = pd.read_csv('Daten_zusammen_final.csv', sep=';')

#Variablen, die keine Rolle spielen entfernen
DatM = Database.drop(['Actions', 'BuyTicketsUrl','Deadline','DefaultLocalTimeOffset','DefaultMatchFormat','DefaultTimeZone',
'DispatchMethod', 'DispatchStatus', 'EarningsCurrencyCode', 'EndDateFederationQuota',
'EntryPointsBaseDate','EntryPointsDayOffset', 'EntryPointsTemplateNo','EventAuxiliaryPersons',
'EventLogos','IsFreeEntrance', 'IsVisManaged','Logos','MatchPointsMethod',
'MaxRegisterFederation', 'MaxRegisterHost', 'MaxReserveTeams', 'MaxTeamsDispatchFederation',
'MaxTeamsDispatchHost','MaxTeamsFederation','MaxTeamsHost','MaxTeamsMainDrawFederation', 'MaxTeamsMainDrawHost',
'MinConfederationTeams', 'MinTeamsDispatchHost', 'NbTeamsFromQualification',
'NbTeamsMainDraw', 'NbTeamsQualification','NbUploads', 'NbWildCards', 'NoTemplateEarnedPoints',
'NoTemplatePrizeMoney','OrganizerCode', 'OrganizerType','Parameters', 'PreliminaryInquiryMainDraw',
'PreliminaryInquiryQualification','SeedPointsBaseDate', 'SeedPointsDayOffset', 'SeedPointsTemplateNo',
'StartDateFederationQuota', 'TechnicalEntryPointsTemplateNo','TechnicalSeedPointsTemplateNo',
'Draws', 'check', 'check2', 'NoSet', 'point', 'latitude', 'longitude', 'timezone', 'UTC_offset_hours', 
'Zeitpunkt', 'index_col','source_row','time','Code','CountryCode','CountryName','DefaultCity','DefaultVenue',
'EndDateMainDraw','EndDateQualification','FederationCode_x','Name_x','No','NoEvent','Season',
'StartDate', 'StartDateMainDraw','StartDateQualification','Status', 'Title',
'Version_x','WebSite','NoItem','No_x',
'FederationCode_x',  'Gender_y', 'PlaysBeach', 'PlaysVolley', 'No_y','Version_y',
'Rank','EarnedPointsTeam', '@MatchPointsA','@MatchPointsB','No','@Version', '@No'],axis=1)


def assign_team_idx(group):
    """
    Für jedes Match (Gruppe) werden:
      - Bei den Spielerzeilen (ItemType = 30) der Team-Index anhand der existierenden FederationCode zugewiesen.
        Die eindeutigen FederationCodes werden alphabetisch sortiert; 
        z. B. 'ABC' → 1, 'DEF' → 2.
      - Bei den Teamzeilen (ItemType = 11) wird der Team-Index entsprechend der Reihenfolge (aufsteigend nach dem Index)
        zugewiesen.
    """
    # Für Spieler: Filtern, sortieren und Mapping erstellen
    players = group[group['ItemType'] == 30].copy()
    unique_codes = sorted(players['FederationCode_y'].dropna().unique())
    mapping = {code: i+1 for i, code in enumerate(unique_codes)}
    
    # Zuweisung des Team-Index zu Spielerzeilen
    group.loc[group['ItemType'] == 30, 'team_idx'] = group.loc[group['ItemType'] == 30, 'FederationCode_y'].map(mapping)
    
    # Für Team-Zeilen (ItemType = 11): Zuweisung nach Reihenfolge (aufsteigend nach Index)
    team_rows = group[group['ItemType'] == 11].copy().sort_index()
    team_idxs = list(range(1, len(team_rows) + 1))
    group.loc[group['ItemType'] == 11, 'team_idx'] = team_idxs
    
    return group


df = DatM
# Wende die Funktion pro Match an
df_assigned = df.groupby('NoMatch', group_keys=False).apply(assign_team_idx)

# Extrahiere die Teamzeilen (ItemType =11) mit MatchNo und team_idx; bei diesen Zeilen ist TeamFault gesetzt
df_team = df_assigned[df_assigned['ItemType'] == 11][['NoMatch', 'team_idx', 'TeamFault','NoPlayer1', 'NoPlayer2']].drop_duplicates()

# Extrahiere die Spielerzeilen (ItemType = 30) – hier ist FederationCode vorhanden
df_players = df_assigned[df_assigned['ItemType'] == 30].copy()

# Merge: Verbinde mittels MatchNo und team_idx die Spielerzeilen mit den Teamzeilen, um TeamFault an die Spieler zu hängen
df_merged = pd.merge(df_players, df_team, on=['NoMatch', 'team_idx'], how='left', suffixes=('', '_team'))

# Optional: Sortiere nach Index, um die ursprüngliche Reihenfolge beizubehalten
df_merged.sort_index(inplace=True)




#Datensatz nach den Zeilen filtern, wo Statistikdaten vorhanden sind

DatS = df_merged[df_merged['ServeTotal'] !=0]


DatS.to_csv('DatenML_V1.csv', index=False, sep=';')


def assign_team_label(row):
    # Lese den Teamnamen der aktuellen Zeile aus
    team_name = row['TeamName']
    # Zerlege den String in @TeamAName in einzelne Namen – Trimme Leerzeichen
    team_a_names = [name.strip() for name in row['@TeamAName'].split('/')]
    # Falls team_name in der Liste der TeamA-Namen enthalten ist, handelt es sich um Team A, ansonsten um Team B
    return 'A' if team_name in team_a_names else 'B'




DatSS = DatS.copy()
DatSS['TeamDesignation'] = DatSS.apply(assign_team_label, axis=1)

def group_and_sum_with_second_names(df):

    # Spalten, die aufsummiert werden sollen
    sum_cols = [
        'SpikeFault', 'SpikePoint', 'ServeFault', 'ServePoint',
        'ServeTotal', 'BlockPoint', 'BlockTotal', 'DigTotal',
        'NoMatch', 'PointTotal', 'ReceptionFault', 'SpikeTotal'
    ]
    # Gruppenschlüssel
    group_keys = ['NoMatch', 'TeamDesignation']
    
    # Erstelle das Aggregations-Dictionary
    agg_dict = {}
    for col in df.columns:
        if col in group_keys:
            agg_dict[col] = 'first'
        elif col in sum_cols:
            agg_dict[col] = 'sum'
        # Für FirstName und LastName legen wir zuerst 'first' fest
        # und ermitteln später den "zweiten" Namen (der sonst verloren ginge)
        elif col in ['FirstName', 'LastName']:
            agg_dict[col] = 'first'
        else:
            agg_dict[col] = 'first'
    
    # Gruppierung des DataFrames
    df_grouped = df.groupby(group_keys, as_index=False).agg(agg_dict)
    
    # Hilfsfunktion, um den zweiten Wert einer Gruppe zu erhalten
    def get_second(x):
        """Gibt den zweiten Eintrag von x zurück, falls vorhanden, ansonsten None."""
        if len(x) > 1:
            return x.iloc[1]
        return None
    
    # Ermittle für jede Gruppe den zweiten FirstName und LastName
    second_names = df.groupby(group_keys).agg({
        'FirstName': get_second,
        'LastName': get_second
    }).reset_index()
    
    # Umbenennen der Spalten auf FirstName2 und LastName2
    second_names = second_names.rename(columns={'FirstName': 'FirstName2', 'LastName': 'LastName2'})
    
    # Zusammenführen der aggregierten Daten mit den "zweiten" Namen
    df_final = pd.merge(df_grouped, second_names, on=group_keys, how='left')
    
    return df_final

# Beispielaufruf:
df_aggregated = group_and_sum_with_second_names(DatSS)


#variante 3


df = df_aggregated.copy()

# 1. Dynamisch alle Set-Nummern finden
# Wir suchen in den Spalten nach Mustern wie "@PointsTeamASetX" und extrahieren X
set_nums = sorted(
    {int(re.search(r'@PointsTeamASet(\d+)', col).group(1))
     for col in df.columns
     if re.match(r'@PointsTeamASet\d+', col)}
)

# 2. Gewinn-Indikatoren für jedes Team und jedes Set erstellen
for i in set_nums:
    df[f'win_set{i}_A'] = (df[f'@PointsTeamASet{i}'] > df[f'@PointsTeamBSet{i}']).astype(int)
    df[f'win_set{i}_B'] = (df[f'@PointsTeamBSet{i}'] > df[f'@PointsTeamASet{i}']).astype(int)

# 3. Summiere die gewonnenen Sätze je Match und bestimme den Gewinner
setA_cols = [f'win_set{i}_A' for i in set_nums]
setB_cols = [f'win_set{i}_B' for i in set_nums]

match_totals = (
    df
    .groupby('NoMatch')[setA_cols + setB_cols]
    .sum()
    .reset_index()
)

match_totals['match_winner'] = match_totals.apply(
    lambda row: 'A' if row[setA_cols].sum() > row[setB_cols].sum()
                else ('B' if row[setB_cols].sum() > row[setA_cols].sum()
                      else pd.NA),
    axis=1
)

# 4. Merge und match_win erzeugen
df = df.merge(
    match_totals[['NoMatch', 'match_winner']],
    on='NoMatch',
    how='left'
)
df['match_win'] = (df['TeamDesignation'] == df['match_winner']).astype(int)

df_final3 = df.drop(columns=setA_cols + setB_cols + ['match_winner'])


#Variablen entfernen
DatS1 = df_final3[['Gender_x','Type', 'TournamentNo', 'SpikeFault', 'SpikePoint', 'ServeFault',
       'ServePoint', 'ServeTotal', 'BlockPoint', 'BlockTotal', 'DigTotal',
        'ReceptionFault', 'SpikeTotal','@LocalDate',
       '@LocalTime', 'FederationCode_y', 'FirstName', 'LastName','FirstName2','LastName2', '@TeamAName', '@TeamBName', 'NoPlayer1_team', 'NoPlayer2_team', '@PointsTeamASet1',
       '@PointsTeamBSet1', '@PointsTeamASet2','@PointsTeamBSet2','@PointsTeamASet3','@PointsTeamBSet3',
        '@DurationSet1',
       '@DurationSet2', '@DurationSet3', 'temperature_2m',
       'precipitation', 'wind_speed_10m', 'rain', 'wind_gusts_10m','TeamFault_team',  'match_win', 'TeamDesignation']]

DatS2 = DatS1.dropna(subset=['temperature_2m', 'precipitation', 'wind_speed_10m', 'rain', 'wind_gusts_10m'])


DatS3 = DatS2.copy()

# Neue Spalte basierend auf TeamDesignation erstellen:
DatS3["TeamNameFull"] = np.where(DatS3["TeamDesignation"] == "A", DatS3["@TeamAName"], DatS3["@TeamBName"])


import unicodedata
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

######################################################################################################################
# Wende die Funktion auf die Spalte "@Namen" an und speichere das Ergebnis in der neuen Spalte "Standard_Namen"
BeachTeams1 = test.copy()
BeachTeams1["Standard_Namen"] = BeachTeams1["TeamNameFull"].apply(standardize_names)

def reorder_player_numbers(row):
    """
    Vergleicht den ursprünglichen Namen (gesplittet in zwei Teile) mit der 
    alphabetisch sortierten Version. Stimmen sie nicht überein, wird angenommen,
    dass auch die Player-Daten vertauscht wurden – und es werden die zugehörigen
    numerischen Werte getauscht.
    """
    # Die ursprüngliche Reihenfolge aus der Spalte '@Namen'
    unsorted_names = [name.strip() for name in row["TeamNameFull"].split('/')]
    # Berechne die standardisierte (sortierte) Reihenfolge:
    sorted_names = sorted(unsorted_names, key=lambda x: normalize_string(x).lower())
    
    # Falls die Reihenfolge nicht gleich ist, tausche die zugehörigen numerischen Spalten:
    if unsorted_names != sorted_names:
        row["NoPlayer1_team"], row["NoPlayer2_team"] = row["@NoPlayer2"], row["@NoPlayer1"]

        # Falls weitere Spalten (z.B. playerbezogene Positionen) in diesem Zusammenhang,
        # können diese ebenfalls getauscht werden.
    return row


