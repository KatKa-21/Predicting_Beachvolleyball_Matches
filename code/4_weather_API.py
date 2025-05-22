####################################################################################
## Wetterdaten scapen und zusammenfügen

import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime, timedelta
import pytz
import urllib.parse
import os

#1. Komplette Daten einlesen mit ID
Database = pd.read_csv('Data_final_mitTimezone.csv', sep=';')

#Zeit anpassen

Database['Zeitpunkt'] = Database['@LocalDate'] + ' ' + Database['@LocalTime']

Data_Prep =Database[['ID_DefaultCity', '@LocalDate', '@LocalTime', 'longitude', 'latitude', 'timezone', 'Zeitpunkt']].drop_duplicates()

### Anwendung Wetter API
# Datensatz muss LocalDate, LocalTime, Latitude, Longitude und Timezone enthalten 

def get_weather_data_for_df(df, wait_per_call=0.72):

    results = []
    
    for idx, row in df.iterrows():
        date = row['@LocalDate']
        start_time = row['@LocalTime']
        latitude = row['latitude']
        longitude = row['longitude']
        timezone_str = row['timezone']  # Dynamische Zeitzone aus dem DataFrame
        
        try:
            # Lokaler Zeitpunkt: Datum + Uhrzeit in der angegebenen Zeitzone
            local_tz = pytz.timezone(timezone_str)
            local_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M:%S")
            local_dt = local_tz.localize(local_dt)
            
            # Eine Stunde als Zeitraum definieren
            end_local_dt = local_dt + timedelta(hours=1)
            
            # Formatierung als ISO-String (ohne Offset, da die API über den Parameter 'timezone'
            # die Zeitzone vorgibt)
            start_datetime_str = local_dt.strftime("%Y-%m-%dT%H:%M:%S")
            end_datetime_str = end_local_dt.strftime("%Y-%m-%dT%H:%M:%S")
            local_date_str = local_dt.strftime("%Y-%m-%d")
            
            # URL-kodierte Zeitzone (z. B. "Europe/Berlin" → "Europe%2FBerlin")
            encoded_timezone = urllib.parse.quote(timezone_str)
            
            # Aufbau der API-URL; hier wird die Zeitzone dynamisch übergeben
            url = (
                f"https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={latitude}&longitude={longitude}"
                f"&start_date={local_date_str}&end_date={local_date_str}"
                f"&hourly=temperature_2m,precipitation,wind_speed_10m,rain,wind_gusts_10m"
                f"&timezone={encoded_timezone}"
            )
            
            response = requests.get(url)
            response.raise_for_status()
            dataW = response.json()
            
            df_weather = pd.DataFrame(dataW['hourly'])
            # Filter: nur die Zeilen, die innerhalb des einstündigen Zeitraums liegen
            df_filtered = df_weather[
                (df_weather['time'] >= start_datetime_str) & 
                (df_weather['time'] < end_datetime_str)
            ].copy()
            
            # Zur späteren Zuordnung merken wir uns den Zeilenindex
            if not df_filtered.empty:
                df_filtered['source_row'] = idx
            
            results.append(df_filtered)
            
        except Exception as e:
            print(f"Fehler bei Zeile {idx}: {e}")
            continue
        
        # Throttling: Warten, um sowohl 600 Aufrufe pro Minute als auch maximal 5000 Aufrufe pro Stunde einzuhalten.
        time.sleep(wait_per_call)
    
    if results:
        weather_all = pd.concat(results, ignore_index=True)
        return weather_all
    else:
        return pd.DataFrame()


# Aufteilen in 3 ungefähr gleich große Teile
dfs = np.array_split(Data_Prep, 3)

all_weather_results = []
for i, sub_df in enumerate(dfs, start=1):
    print(f"Verarbeite Batch {i} von {len(dfs)} (Zeilen: {len(sub_df)})...")
    weather_batch = get_weather_data_for_df(sub_df, wait_per_call=0.72)
    all_weather_results.append(weather_batch)
    
    # Optionale zusätzliche Pause zwischen den Batches, falls notwendig:
    time.sleep(60)

# Zusammenführen aller Ergebnisse
final_weather_df = pd.concat(all_weather_results, ignore_index=True)
print("Verarbeitung abgeschlossen. Gesamtanzahl abgerufener Wetter-Datensätze:", len(final_weather_df))

# Ausgangsdatensatz
Data_Prep['source_row'] = Data_Prep.index

DatZ1 = Data_Prep.merge(final_weather_df, left_on='source_row', right_on='source_row', how='left')


#zeilen mit fehlender Uhrzeit entfernen
DatZ2 = DatZ1.dropna(subset=('@LocalTime'))


DatFZ = Database.merge(DatZ1[['ID_DefaultCity',  'Zeitpunkt', 'index_col', 'source_row', 'time',
       'temperature_2m', 'precipitation', 'wind_speed_10m', 'rain',
       'wind_gusts_10m']], left_on=['ID_DefaultCity', 'Zeitpunkt'],
                       right_on=['ID_DefaultCity', 'Zeitpunkt'],
                       how='left')

diff=DatFZ[DatFZ.duplicated(subset='ID_row', keep=False)]
#die zeilen haben keine wetter infos
DatFZ2 = DatFZ[~DatFZ.duplicated(subset='ID_row', keep=False)]

#Variablen, die keine Rolle im Modell spielen, entfernen

DatFZ3 = DatFZ2.drop(['Actions', 'BuyTicketsUrl','Deadline','DefaultLocalTimeOffset','DefaultMatchFormat','DefaultTimeZone',
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
'Zeitpunkt', 'index_col','source_row','time'],axis=1)


