import pandas as pd

# Carregar os dados
df = pd.read_csv("Airplane_Crashes_and_Fatalities_Since_1908.csv")

def get_unique_locations(df):
    return sorted(df['Location'].unique())

def get_unique_operators(df):
    return sorted(df['Operator'].unique())

