from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Conexão com PostgreSQL
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="senha",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Carregar os dados CSV
df = pd.read_csv('Airplane_Crashes_and_Fatalities_Since_1908.csv')

# Carregar o modelo treinado
with open('modelo.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
le_operator = model_data['le_operator']
le_location = model_data['le_location']
le_type = model_data['le_type']

# Função para converter gráfico para Base64
def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    location = data['location']
    operator = data['operator']
    aboard = data['aboard']
    ground = data['ground']
    year = data['year']
    month = data['month']
    weekday = data['weekday']
    aircraft_type = data['type']

    # Codificar as variáveis categóricas
    location_encoded = le_location.transform([location])[0]
    operator_encoded = le_operator.transform([operator])[0]
    type_encoded = le_type.transform([aircraft_type])[0]

    # Criar DataFrame com os nomes corretos das features
    input_df = pd.DataFrame([{
        "Aboard": aboard,
        "Ground": ground,
        "Year": year,
        "Month": month,
        "Weekday": weekday,
        "Operator_encoded": operator_encoded,
        "Type_encoded": type_encoded,
        "Location_encoded": location_encoded
    }])

    # Fazer a predição
    prediction = model.predict(input_df)[0]

    # Criar gráfico de barras da entrada (features)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(input_df.columns), y=input_df.values[0])
    plt.xticks(rotation=45)
    plt.title("📊 Entrada da Predição")
    plt.tight_layout()

    # Converter gráfico para Base64
    input_chart_base64 = plot_to_base64(plt)
    plt.close()

    # Criar gráfico de barras do resultado da predição
    plt.figure(figsize=(4, 3))
    sns.barplot(x=["Sobreviveu", "Fatal"], y=[int(prediction == 0), int(prediction == 1)])
    plt.title("Resultado da Predição")
    plt.tight_layout()

    # Converter gráfico de resultado para Base64
    result_chart_base64 = plot_to_base64(plt)
    plt.close()

    # Inserir dados no PostgreSQL com Base64 dos gráficos
    cursor.execute("""
        INSERT INTO predicoes (location, operator, aboard, ground, year, month, weekday, aircraft_type, prediction, input_chart_base64, result_chart_base64)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        location, operator, aboard, ground, year, month, weekday,
        aircraft_type, int(prediction), input_chart_base64, result_chart_base64
    ))
    conn.commit()

    # Retornar os resultados com gráficos em Base64
    return jsonify({
        'survivors': int(prediction == 0),
        'fatalities': int(prediction == 1),
        'input_chart': input_chart_base64,
        'result_chart': result_chart_base64
    })

@app.route('/options', methods=['GET'])
def get_options():
    locations = df['Location'].dropna().unique().tolist()
    operators = df['Operator'].dropna().unique().tolist()
    types = df['Type'].dropna().unique().tolist()
    return jsonify({
        'locations': locations,
        'operators': operators,
        'types': types
    })


@app.route('/predictions', methods=['GET'])
def get_predictions():
    # Consultar todos os registros na tabela 'predicoes'
    cursor.execute("""
        SELECT id, location, operator, aboard, ground, year, month, weekday, aircraft_type, prediction, input_chart_base64, result_chart_base64
        FROM predicoes
    """)
    # Recuperar os resultados
    rows = cursor.fetchall()

    # Formatando os dados em JSON
    predictions = []
    for row in rows:
        predictions.append({
            'id': row[0],
            'location': row[1],
            'operator': row[2],
            'aboard': row[3],
            'ground': row[4],
            'year': row[5],
            'month': row[6],
            'weekday': row[7],
            'aircraft_type': row[8],
            'prediction': row[9],
            'input_chart_base64': row[10],
            'result_chart_base64': row[11]
        })

    # Retornar as predições como resposta JSON
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
