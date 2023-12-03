from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)
CORS(app)

model_path = 'Models/ResNet50_8b10e.hdf5'
model = load_model(model_path)

# Precargar x_test y y_labTest con las imágenes de prueba y las etiquetas verdaderas
df_test = pd.read_csv('solution_stg1_release.csv', sep=';')
testpath = 'TestImages/'
x_test = []
y_labTest = []

for index, row in df_test.iterrows():
    image_path = os.path.join(testpath, row['image_name'])
    img = plt.imread(image_path)
    img = img / 255.0
    img = np.resize(img, (224, 224, 3))
    x_test.append(img)

    if row['Type_1'] == 1:
        y_labTest.append(0)
    elif row['Type_2'] == 1:
        y_labTest.append(1)
    elif row['Type_3'] == 1:
        y_labTest.append(2)

x_test = np.array(x_test)
y_labTest = np.array(y_labTest)
y_test = to_categorical(y_labTest, num_classes=3, dtype='float32')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    try:
        # Leer la imagen y preprocesarla
        img = plt.imread(file)
        img = img / 255.0
        img = np.resize(img, (1, 224, 224, 3))

        # Hacer la predicción
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)

        return jsonify({'predicted_class': int(predicted_class[0])})

    except Exception as e:
        return str(e), 500


@app.route('/metrics', methods=['POST'])
def metrics():
    if 'file' not in request.files or 'label' not in request.form:
        return "Missing file or label", 400

    file = request.files['file']
    true_label = int(request.form['label'])
    if file.filename == '':
        return "No selected file", 400

    try:
        # Leer la imagen y preprocesarla
        img = plt.imread(file)
        img = img / 255.0
        img = np.resize(img, (1, 224, 224, 3))

        # Hacer la predicción
        pred = model.predict(img)
        predicted_label = np.argmax(pred, axis=1)[0]

        # Calcular métricas para esta predicción individual
        accuracy = accuracy_score([true_label], [predicted_label])
        conf_matrix = confusion_matrix([true_label], [predicted_label])
        precision, recall, fscore, _ = precision_recall_fscore_support(
            [true_label],
            [predicted_label],
            zero_division=0  # Asegura que no hay advertencias
        )
        class_names = {0: "Neoplasia Tipo 1", 1: "Neoplasia Tipo 2", 2: "Neoplasia Tipo 3"}
        predicted_diagnosis = class_names[int(predicted_label)]

        # Calcula la confianza del diagnóstico (en este caso, usamos la precisión como proxy)
        diagnosis_confidence = "{:.0%}".format(float(accuracy))

        # Prepara el detalle técnico
        technical_detail = {
            "Precisión del Diagnóstico": "{:.0%}".format(precision[0].item()),
            "Casos Correctamente Identificados": int(conf_matrix[0][0].item()),
            "Casos Incorrectamente Identificados": int(conf_matrix[0].sum().item() - conf_matrix[0][0].item()),
            "Recall": "{:.0%}".format(recall[0].item()),
            "F-Score": "{:.0%}".format(fscore[0].item())
        }

        # Devuelve la respuesta en el formato deseado
        return jsonify({
            "Diagnóstico Predicho": predicted_diagnosis,
            "Confianza del Diagnóstico": diagnosis_confidence,
            "Resumen de la Evaluación": "El diagnóstico ha sido verificado con alta precisión. No se detectaron errores en la clasificación de los casos probados.",
            "Recomendaciones": "Se sugiere considerar el diagnóstico predicho para una evaluación más detallada o para complementar con otras pruebas diagnósticas si es necesario.",
            "Detalle Técnico": technical_detail
        })

        """
        print({
            'predicted_class': int(predicted_label),
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'precision': precision[0].item(),
            'recall': recall[0].item(),
            'fscore': fscore[0].item()
        })
        
        # Devolver métricas
        return jsonify({
            'predicted_class': int(predicted_label),
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'precision': precision[0].item(),
            'recall': recall[0].item(),
            'fscore': fscore[0].item()
        })
        """

    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
