# Recriar as etapas do treinamento com as features necessárias
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle

# Carregar os dados
df = pd.read_csv("Airplane_Crashes_and_Fatalities_Since_1908.csv")

# Excluir valores faltantes essenciais
df = df.dropna(subset=["Date", "Operator", "Type", "Aboard", "Fatalities", "Ground"])

# Criar a variável alvo
df["fatal"] = (df["Fatalities"] > 0).astype(int)

# Converte a coluna "Date" para datetime e extrai o ano, mês e dia da semana
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Weekday"] = df["Date"].dt.dayofweek

# Codificar a coluna "Operator" com Label Encoding
le_operator = LabelEncoder()
df["Operator_encoded"] = le_operator.fit_transform(df["Operator"])

# Codificar a coluna "Type" com Label Encoding
le_type = LabelEncoder()
df["Type_encoded"] = le_type.fit_transform(df["Type"])

# Codificar a coluna "Location" com Label Encoding
le_location = LabelEncoder()
df["Location_encoded"] = le_location.fit_transform(df["Location"])

# Selecionar as features que serão usadas no treinamento
features = df[["Aboard", "Ground", "Year", "Month", "Weekday", "Operator_encoded", "Type_encoded", "Location_encoded"]]
target = df["fatal"]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Aplicar SMOTE para balancear as classes
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Criar e treinar o modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# Fazer previsões
y_pred = model.predict(X_test)

# Exibir o relatório de classificação
print("📊 Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Salvar o modelo treinado
with open('modelo.pkl', 'wb') as f:
    pickle.dump({
        "model": model,
        "features": features.columns.tolist(),
        "le_operator": le_operator,
        "le_location": le_location,
        "le_type": le_type
    }, f)

print("✅ Modelo salvo com sucesso!")
