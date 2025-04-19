import pandas as pd

# Carregar os dados
df = pd.read_csv("Airplane_Crashes_and_Fatalities_Since_1908.csv")

# Obter todos os valores únicos da coluna 'Location'
locations = df['Location'].unique()
print("Localizações únicas:", locations)

# Obter todos os valores únicos da coluna 'Operator'
operators = df['Operator'].unique()
print("Operadores únicos:", operators)
