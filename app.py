import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Carregar o modelo treinado e os encoders
with open('modelo.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    le_operator = model_data["le_operator"]
    le_location = model_data["le_location"]
    le_type = model_data["le_type"]

# Função para gerar gráfico de barras com fatalidades por localização
def generate_location_graph(df):
    plt.figure(figsize=(10, 6))
    location_counts = df.groupby('Location')['Fatalities'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=location_counts.values, y=location_counts.index, palette='viridis')
    plt.title('Top 10 Localizações com Maior Número de Fatalidades')
    plt.xlabel('Fatalidades')
    plt.ylabel('Localização')

    # Salvar gráfico em buffer de memória
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return img_b64


# Função para gerar gráfico de barras com fatalidades por operador
def generate_operator_graph(df):
    plt.figure(figsize=(10, 6))
    operator_counts = df.groupby('Operator')['Fatalities'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=operator_counts.values, y=operator_counts.index, palette='magma')
    plt.title('Top 10 Operadores com Maior Número de Fatalidades')
    plt.xlabel('Fatalidades')
    plt.ylabel('Operador')

    # Salvar gráfico em buffer de memória
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return img_b64


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obter dados do JSON
        data = request.get_json()

        # Extrair valores do JSON
        location = data['location']
        operator = data['operator']
        aboard = data['aboard']
        ground = data['ground']
        year = data['year']

        # Codificar os valores de "location" e "operator" usando os encoders
        try:
            location_encoded = le_location.transform([location])[0]
        except KeyError:
            return jsonify({'error': f"Localização '{location}' não encontrada no encoder.", 'message': 'Erro na codificação da localização.'}), 400

        try:
            operator_encoded = le_operator.transform([operator])[0]
        except KeyError:
            return jsonify({'error': f"Operador '{operator}' não encontrado no encoder.", 'message': 'Erro na codificação do operador.'}), 400

        # Extrair as variáveis adicionais, como 'Month' e 'Weekday'
        try:
            month = pd.to_datetime(f"{year}-01-01").month
            weekday = pd.to_datetime(f"{year}-01-01").weekday()
        except Exception as e:
            return jsonify({'error': str(e), 'message': 'Erro ao processar as variáveis de data.'}), 400

        # Codificar a coluna 'Type' com um valor arbitrário, já que não está sendo usada aqui.
        type_encoded = 0  # ou algum valor válido

        # Criar o DataFrame de entrada para a predição
        input_data = pd.DataFrame([{
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
        try:
            prediction = model.predict(input_data)
        except Exception as e:
            return jsonify({'error': str(e), 'message': 'Erro na predição.'}), 500

        # Carregar os dados para gerar os gráficos
        try:
            df = pd.read_csv("Airplane_Crashes_and_Fatalities_Since_1908.csv")
            location_graph = generate_location_graph(df)
            operator_graph = generate_operator_graph(df)
        except Exception as e:
            return jsonify({'error': str(e), 'message': 'Erro ao gerar gráficos.'}), 500

        # Retornar a resposta
        result = {
            'survivors': bool(prediction[0]),  # 1 = sobreviventes, 0 = não sobreviventes
            'input_data': {
                'location': location,
                'operator': operator,
                'aboard': aboard,
                'ground': ground,
                'year': year,
                'month': month,
                'weekday': weekday,
                'location_encoded': int(location_encoded),  # Convertido para int
                'operator_encoded': int(operator_encoded),  # Convertido para int
                'type_encoded': type_encoded
            },
            'graphs': {
                'location_graph': location_graph,
                'operator_graph': operator_graph
            }
        }

        return jsonify(result)

    except Exception as e:
        # Exibir erro detalhado no console
        print("Erro detalhado:", str(e))
        return jsonify({'error': str(e), 'message': 'Erro ao processar a predição.'}), 500


if __name__ == '__main__':
    app.run(debug=True)
