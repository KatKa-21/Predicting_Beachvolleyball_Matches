
# data.py
import os
import pandas as pd

def load_data(path="Playerdata.csv"):
    df = pd.read_csv(path, sep=";")
    return df

def get_avg_servepoints_by_tournament(df):
    grouped = (
        df.groupby(["Full_Name", "Season"], as_index=False)["ServePoint"]
        .mean()
        .rename(columns={"ServePoint": "AverageServePoints"})
    )
    return grouped
