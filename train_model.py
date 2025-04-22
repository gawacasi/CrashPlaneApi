import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv("Airplane_Crashes_and_Fatalities_Since_1908.csv")

# Remover registros com informações essenciais ausentes
df = df.dropna(subset=["Date", "Operator", "Type", "Location", "Aboard", "Fatalities", "Ground"])

# Criar a variável alvo (1 = teve fatalidade, 0 = não teve)
df["fatal"] = (df["Fatalities"] / df["Aboard"] > 0.7).astype(int)

# Verificar balanceamento da variável alvo
print("📊 Distribuição da variável alvo (fatal):")
print(df["fatal"].value_counts())
print(df["fatal"].value_counts(normalize=True))

# Converter a coluna de data para datetime e extrair partes úteis
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Weekday"] = df["Date"].dt.dayofweek

# Codificar variáveis categóricas
le_operator = LabelEncoder()
le_type = LabelEncoder()
le_location = LabelEncoder()

df["Operator_encoded"] = le_operator.fit_transform(df["Operator"])
df["Type_encoded"] = le_type.fit_transform(df["Type"])
df["Location_encoded"] = le_location.fit_transform(df["Location"])

# Selecionar as features para o modelo
features = df[[
    "Aboard", "Ground", "Year", "Month", "Weekday",
    "Operator_encoded", "Type_encoded", "Location_encoded"
]]
target = df["fatal"]

# Dividir os dados com estratificação para manter proporção
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

# Aplicar SMOTE para balancear as classes
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Criar e treinar o modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# Previsões
y_pred = model.predict(X_test)

# Avaliação
print("\n📊 Relatório de Classificação:")
print(classification_report(y_test, y_pred))


# Salvar o modelo e encoders
with open("modelo.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "features": features.columns.tolist(),
        "le_operator": le_operator,
        "le_type": le_type,
        "le_location": le_location
    }, f)

print("✅ Modelo salvo com sucesso!")
