# Using the OpenCage API
# The goal is to obtain the corresponding coordinates (forward geocoding) using location names.

from opencage.geocoder import OpenCageGeocode
from pprint import pprint
import pandas as pd
import os
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from timezonefinder import TimezoneFinder
import pytz
from datetime import datetime

# Read existing data
Database = pd.read_csv('Datenbestand_Final1.csv', sep=';')

# Add IDs
Database['ID_row'] = range(1, len(Database) + 1)
Database['ID_DefaultCity'] = Database['DefaultCity'].astype('category').cat.codes

Database.to_csv('Daten_mit_IDs.csv', index=False, sep=';')

# Process unique locations
Orte = Database[['DefaultCity', 'ID_DefaultCity', 'CountryName']]
Orte_unique = Orte.drop_duplicates(subset=['DefaultCity']).copy()
Orte_unique.loc[:, 'Koord_Name'] = Orte_unique['DefaultCity'] + ', ' + Orte_unique['CountryName']

# Geolocation setup
geolocator = Nominatim(user_agent="my_app")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

Orte_unique['location'] = Orte_unique['Koord_Name'].apply(geocode)
Orte_unique['point'] = Orte_unique['location'].apply(lambda loc: tuple(loc.point) if loc else None)

# Merge with ID_DefaultCity
Data_Kom1 = Database.merge(Orte_unique.iloc[:, [1, 4, 5]], left_on='ID_DefaultCity', right_on='ID_DefaultCity', how='left')

# Save locations with coordinates
Orte_unique.to_csv('Orte_mit_Koordinaten.csv', index=False, sep=';')

# Data cleaning before calculating coordinates
# First, calculate the difference to UTC using coordinates

DatU = Data_Kom1[['CountryName', 'DefaultTimeZone', 'DefaultCity', 'point', 'ID_DefaultCity']].drop_duplicates()
DatU2 = DatU.copy()
DatU2['point'] = DatU2['point'].astype(str)

DatU2[['latitude', 'longitude']] = (
    DatU2['point']
    .str.strip('()')  # Remove parentheses
    .str.split(',', expand=True)  # Split into 3 parts
    .iloc[:, :2]  # Keep only the first 2 columns
).copy()

DatU2 = DatU2.dropna()
DatU2['latitude'] = DatU2['latitude'].astype(float)
DatU2['longitude'] = DatU2['longitude'].astype(float)

Koordinates = DatU2.iloc[:, [5, 6]]

# Function to add time zone and UTC offset
def add_timezone_and_offset(df, lat_col='latitude', lon_col='longitude'):
    tf = TimezoneFinder()
    
    timezones = []
    utc_offsets = []
    
    for _, row in df.iterrows():
        lat, lon = row[lat_col], row[lon_col]
        tz_str = tf.timezone_at(lat=lat, lng=lon)
        
        if tz_str is None:
            timezones.append(None)
            utc_offsets.append(None)
        else:
            timezones.append(tz_str)
            tz = pytz.timezone(tz_str)
            now = datetime.now(tz)
            offset = now.utcoffset()
            offset_hours = offset.total_seconds() / 3600 if offset else None
            utc_offsets.append(offset_hours)

    df['timezone'] = timezones
    df['UTC_offset_hours'] = utc_offsets
    
    return df

DatAdd = add_timezone_and_offset(Koordinates)

dat1 = DatU2.merge(DatAdd, left_on=['latitude', 'longitude'], right_on=['latitude', 'longitude'], how='inner')

DatKom2 = Data_Kom1.copy()
DatKom2['point'] = DatKom2['point'].astype(str)

# Append to main dataset
DatF = DatKom2.merge(dat1.iloc[:, [4, 5, 6, 7, 8]], left_on='ID_DefaultCity', right_on='ID_DefaultCity', how='left')

DatF = DatF.drop('location', axis=1)
DatF2 = DatF.drop_duplicates()

# Save final dataset
DatF2.to_csv('Data_final_mitTimezone.csv', index=False, sep=';')

# Documentation for historical weather API
# https://open-meteo.com/en/docs/historical-weather-api
