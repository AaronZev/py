from flask import Flask, request, jsonify
import os
import pyodbc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import base64
import json

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Configuración de la conexión a la base de datos
        server = 'adminpy.database.windows.net' 
        database = 'Nirpy' 
        username = 'adminpy' 
        password = 'CCoNNa2205**' 
        driver = 'ODBC Driver 17 for SQL Server'

        # Cadena de conexión
        conn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

        # Crear la conexión
        connection = pyodbc.connect(conn_str)
        cursor = connection.cursor()

        # Consulta SQL para obtener el último archivo cargado
        query = "SELECT TOP 1 FileContent FROM FileUpload ORDER BY DateUploaded DESC"

        # Ejecutar la consulta
        cursor.execute(query)
        row = cursor.fetchone()

        # Decodificar la cadena Base64 y almacenar en la variable CafeAbsorbancia
        CafeAbsorbancia = ""
        if row and row[0]:
            base64_data = row[0]
            decoded_data = base64.b64decode(base64_data)

            # Convertir los datos decodificados a una cadena y asignar a la variable CafeAbsorbancia
            CafeAbsorbancia = decoded_data.decode('utf-8')

            # Validar si es un archivo CSV
            try:
                # Crear un objeto StringIO y omitir las primeras 28 filas
                cafe_absorbancia_io = StringIO(CafeAbsorbancia)
                for _ in range(28):
                    cafe_absorbancia_io.readline()

                # Cargar los datos en un DataFrame de pandas
                df = pd.read_csv(cafe_absorbancia_io)

                try:
                    # Dividir los datos en conjuntos de entrenamiento y prueba
                    X = df['Wavelength (nm)'].values.reshape(-1, 1)
                    y = df['Absorbance (AU)'].values
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Entrenar y evaluar el modelo
                    results = train_and_evaluate_model(X_train, y_train, X_test, y_test)

                    # Convertir los resultados a formato JSON
                    result_dict = {
                        "rmse_train": results[3],
                        "rmse_test": results[7],
                        "img_train_base64": plot_to_base64(X_train, y_train, results[2], 'Entrenamiento - Predicciones vs. Real'),
                        "img_test_base64": plot_to_base64(X_test, y_test, results[6], 'Prueba - Predicciones vs. Real')
                    }

                    return jsonify(result_dict)

                except pd.errors.ParserError:
                    return jsonify({"error": "El Archivo no tiene la estructura admitida"})

            except pd.errors.ParserError:
                return jsonify({"error": "Los datos no son un archivo CSV válido."})

        else:
            return jsonify({"error": "No se encontraron resultados."})

    except pyodbc.Error as e:
        return jsonify({"error": f"Error al conectar a la base de datos: {e}"})


def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    # Modelo de regresión PLS
    model = PLSRegression(n_components=1)
    model.fit(X_train, y_train)

    # Predicciones en conjunto de entrenamiento
    y_pred_train = model.predict(X_train)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)

    # Predicciones en conjunto de prueba
    y_pred_test = model.predict(X_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

    # Devolver todos los resultados necesarios
    return X_train, y_train, y_pred_train, rmse_train, X_test, y_test, y_pred_test, rmse_test

def plot_to_base64(X, y, y_pred, title):
    # Crear una figura de matplotlib
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, label='Real', alpha=0.7)
    plt.scatter(X, y_pred, label='Predicción', marker='x', alpha=0.7)
    plt.xlabel('Longitud de onda')
    plt.ylabel('Absorbancia')
    plt.title(title)
    plt.legend()

    # Guardar la figura en BytesIO
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)

    # Convertir la imagen a base64
    img_base64 = base64.b64encode(img_stream.read()).decode('utf-8')

    return img_base64


if __name__ == '__main__':
    app.run(debug=True)
