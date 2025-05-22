# %% [markdown]
# ## Verwendung der OpenCage - API
# 
# Ziel ist es mit Hilfe von Ortsnamen, die jeweiligen Koordinaten zu erhalten (forward geocoding)

# %%
#API_opencage = '5eebd8503f3341e59f5de1a26fae7457'

# %%
from opencage.geocoder import OpenCageGeocode
from pprint import pprint
import pandas as pd
import os
import geopy
from geopy.geocoders import Nominatim

# %% [markdown]
# ## Bisherigen Datenbestand einlesen

# %%
os.chdir('C:/Users/Katharina/Desktop/Weiterbildung/Bootcamp/Bootcamp/Final_project/data')

# %%
Database = pd.read_csv('Datenbestand_Final1.csv', sep=';')

# %%
#Id anfügen
Database['ID_row'] = range(1, len(Database) +1)
Database['ID_DefaultCity'] = Database['DefaultCity'].astype('category').cat.codes


# %%
Database.to_csv('Daten_mit_IDs.csv', index=False, sep=';')

# %%
#use list with unique locations
Orte = Database[['DefaultCity', 'ID_DefaultCity', 'CountryName']]
Orte_unique = Orte.drop_duplicates(subset=['DefaultCity']).copy()

Orte_unique.loc[:, 'Koord_Name'] = Orte_unique['DefaultCity'] + ', ' + Orte_unique['CountryName']

# %%
geolocator = Nominatim(user_agent="my_app")

from geopy.extra.rate_limiter import RateLimiter
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
Orte_unique['location'] = Orte_unique['Koord_Name'].apply(geocode)

Orte_unique['point'] = Orte_unique['location'].apply(lambda loc: tuple(loc.point) if loc else None)

# %%
#Mergen mit ID_DefaultCity

Data_Kom1 = Database.merge( Orte_unique.iloc[:, [1, 4, 5]], left_on='ID_DefaultCity', right_on='ID_DefaultCity', how='left')


# %%
#orte mit Koordinaten erstmal abspeichern

Orte_unique.to_csv('Orte_mit_Koordinaten.csv', index=False, sep=';')

# %% [markdown]
# #### Data cleaning bevor Koordinaten berechnen

# %% [markdown]
# #####  als erstes mit Koordinaten Unterschied zu utc berechnen

# %%
DatU = Data_Kom1[['CountryName', 'DefaultTimeZone', 'DefaultCity', 'point', 'ID_DefaultCity']].drop_duplicates()
DatU2 = DatU.copy()
DatU2['point'] = DatU2['point'].astype(str)

# %%
DatU2[['latitude', 'longitude']] = (
    DatU2['point']
    .str.strip('()')  # Klammern entfernen
    .str.split(',', expand=True)  # in 3 Teile splitten
    .iloc[:, :2]  # nur die ersten 2 Spalten behalten
    #.apply(lambda x: x.astype(float))  # in float umwandeln
).copy()

# %%
DatU2 = DatU2.dropna()
DatU2['latitude'] = DatU2['latitude'].astype(float)
DatU2['longitude'] = DatU2['longitude'].astype(float)

# %%
Koordinates = DatU2.iloc[:,[5,6]]

# %%
import pandas as pd
from timezonefinder import TimezoneFinder
import pytz
from datetime import datetime


#Diese Funktion ergänzt zwei Spalten: timezone und UTC_offset_hours in Stunden



def add_timezone_and_offset(df, lat_col='latitude', lon_col='longitude'):

    tf = TimezoneFinder()
    
    timezones = []
    utc_offsets = []
    
    for index, row in df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        
        # Ermittle den Zeitzonennamen anhand von Breiten- und Längengrad
        tz_str = tf.timezone_at(lat=lat, lng=lon)
        
        if tz_str is None:
            timezones.append(None)
            utc_offsets.append(None)
        else:
            timezones.append(tz_str)
            # Erzeuge ein pytz-Zeitzonenobjekt
            tz = pytz.timezone(tz_str)
            # Erhalte das aktuelle Datum und die Zeit in der ermittelten Zeitzone
            now = datetime.now(tz)
            offset = now.utcoffset()
            offset_hours = offset.total_seconds() / 3600 if offset is not None else None
            utc_offsets.append(offset_hours)

    # Füge die ermittelten Daten als neue Spalten hinzu
    df['timezone'] = timezones
    df['UTC_offset_hours'] = utc_offsets
    
    return df


# %%
DatAdd = add_timezone_and_offset(Koordinates)

# %%
dat1 = DatU2.merge(DatAdd, left_on=['latitude', 'longitude'], right_on=['latitude', 'longitude'], how='inner')


# %%
DatKom2 = Data_Kom1.copy()
DatKom2['point'] = DatKom2['point'].astype(str)

# %%
#anfügen an großen datensatz

DatF = DatKom2.merge(dat1.iloc[:,[4,5,6,7,8]], left_on='ID_DefaultCity', right_on='ID_DefaultCity', how='left')


# %%
DatF = DatF.drop('location',axis=1)

# %%
DatF2 = DatF.drop_duplicates()

# %%
DatF2.to_csv('Data_final_mitTimezone.csv', index=False, sep=';')

# %% [markdown]
# Doku
# https://open-meteo.com/en/docs/historical-weather-api?hourly=temperature_2m,rain,wind_speed_100m,wind_direction_100m,soil_temperature_7_to_28cm,is_day&daily=temperature_2m_mean,sunrise,sunset,temperature_2m_max,temperature_2m_min,wind_gusts_10m_max,wind_direction_10m_dominant&latitude=53.550341&longitude=10.000654&timezone=Europe%2FBerlin&start_date=2024-08-22&end_date=2024-08-22


