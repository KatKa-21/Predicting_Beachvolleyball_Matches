# Preparation Dataset for plots - single player
import pandas as pd
import glob
import os
import numpy as np

DatS = pd.read_csv('DatenML_V1.csv', sep=';')


DatS1 = pd.merge(DatS[["FirstName", "LastName"]], df2[['Player.@FederationCode', 'Player.@FirstName', 'Player.@Gender',
       'Player.@LastName','Player.@Height','Player.@BeachHighBlock', 'Player.@BeachHighJump',
       'Player.@BeachPosition', 'Player.@BeachYearBegin', 'Player.@Birthdate',
       'Player.@Handedness']], left_on=["FirstName", "LastName"], right_on=["Player.@FirstName", "Player.@LastName"], how="left")


#welche spieler fehlen?

playerMissing = DatS1[DatS1['Player.@FirstName'].isna()].drop_duplicates()


#csv beach_teams --> da sind die nummern drin --> damit scrapen
os.chdir('C:/Users/Katharina/Desktop/Weiterbildung/Bootcamp/Bootcamp/Final_project/data')

BeachTeams = pd.read_csv('beach_teams.csv', sep=',')
BeachTeams = BeachTeams.dropna(subset=['NoPlayer1'])
BeachTeams[["Player1", "Player2"]] = BeachTeams["Name"].str.split("/", expand=True, n=1)
BeachTeams2 = BeachTeams.drop_duplicates(subset='Name')
df2 = pd.merge(playerMissing[["FirstName", "LastName"]], BeachTeams2[['Player1', 'NoPlayer1']],how='left', left_on='LastName', right_on='Player1')


# Funktion zum Ermitteln des modalen Werts; 
# wird der Mode nicht ermittelt (leere Serie), gibt sie np.nan zurück.
def get_mode(series):
    modes = series.mode()
    return modes[0] if not modes.empty else np.nan

# Gruppiere nach FirstName und LastName und bestimme den häufigsten Wert in NoPlayer
mode_df = df2.groupby(["FirstName", "LastName"])["NoPlayer1"].agg(get_mode).reset_index()
mode_df = mode_df.rename(columns={"NoPlayer1": "ModeNoPlayer1"})

# Merge den modalen Wert zurück in das Original-DataFrame
test2 = df2.merge(mode_df, on=["FirstName", "LastName"], how="left")


test3 = test2.copy()
test3 = test3.drop_duplicates(subset=['ModeNoPlayer1'])
test3.dropna()


test3.to_csv('MissingPlayer_listID.csv', index=False, sep=';')


#Variante zwei
test2_A = pd.merge(playerMissing[["FirstName", "LastName"]], BeachTeams2[['Player2', 'NoPlayer2']],how='left', left_on='LastName', right_on='Player2')

# Gruppiere nach FirstName und LastName und bestimme den häufigsten Wert in NoPlayer
mode_df = test2_A.groupby(["FirstName", "LastName"])["NoPlayer2"].agg(get_mode).reset_index()
mode_df = mode_df.rename(columns={"NoPlayer2": "ModeNoPlayer2"})

# Merge den modalen Wert zurück in das Original-DataFrame
test2_B = test2_A.merge(mode_df, on=["FirstName", "LastName"], how="left")

#Duplikate entfernen
test3_A = test2_B.copy()
test3_A = test3_A.drop_duplicates(subset=['ModeNoPlayer2'])
test3_A = test3_A.dropna()


test3_A.to_csv('MissingPlayer_listID_V2.csv', index=False, sep=';')


#restliche spieler gescrapt
MissPl = pd.read_csv('player_list_Missing0705.csv', sep=';')


#Teil2 der Missing gescrapt
MissPl2 = pd.read_csv('player_list_Missing0805.csv', sep=';')


DatMiss = pd.concat([MissPl, MissPl2],axis=0)
DatMiss1 = DatMiss.drop_duplicates(subset=['Player.@No'])

#-> Vergleich mit PlayerMissing: FirstName, LastName


#Prüfen, welche spieler jetzt gefunden worden sind
MISS_DF = pd.merge(DatMiss1, playerMissing[["FirstName", 'LastName']], left_on=["Player.@FirstName", 'Player.@LastName'], 
                   right_on=["FirstName", 'LastName'], how="inner")


#Zusammenfügen mit anderer Spielerliste -> df2 (Drop 'NoData.@id') MissDF (drop 'FirstName', 'LastName')
PLayerList = df2.drop(columns='NoData.@id')
PlayerMiss = MISS_DF.drop(columns=['FirstName', 'LastName'])


#Listen zusammenfügen

Player_fin = pd.concat([PLayerList, PlayerMiss], axis=0)


Player_fin.to_csv('Player_properties_final.csv', index=False, sep=';',)


