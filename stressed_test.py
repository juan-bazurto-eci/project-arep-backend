import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.utils import to_categorical

# Carga el modelo entrenado
model = load_model('./Models/ResNet50_8b10e.hdf5')

# Carga las imágenes de prueba
testpath = '../TestImages/'
df_test = pd.read_csv('../solution_stg1_release.csv', sep=';')

x_test = []
y_labTest = []
for i in range(len(df_test)):
    x_test.append(plt.imread(testpath + df_test.iloc[i]['image_name']))
    # Aquí asumo que tienes una columna para cada tipo en tu CSV
    if df_test.iloc[i]['Type_1'] == 1:
        y_labTest.append(0)
    if df_test.iloc[i]['Type_2'] == 1:
        y_labTest.append(1)
    if df_test.iloc[i]['Type_3'] == 1:
        y_labTest.append(2)

x_test = np.array(x_test) / 255.
y_labTest = np.array(y_labTest)
y_test = to_categorical(y_labTest, num_classes=3, dtype='float32')

# Definir diferentes generadores para cada prueba de estrés
pruebas = {
    'rotacion_extrema': ImageDataGenerator(rotation_range=90),
    'desplazamiento_extremo': ImageDataGenerator(width_shift_range=0.5, height_shift_range=0.5),
    'zoom_extremo': ImageDataGenerator(zoom_range=0.5)
    # Puedes agregar más pruebas aquí
}

resultados = {}

for prueba, datagen in pruebas.items():
    # Aplicar la transformación y proporcionar etiquetas
    datagen.fit(x_test)
    x_test_transformed = datagen.flow(x_test, y_test, batch_size=1, shuffle=False)

    # Evaluar el modelo (no es necesario proporcionar y_test aquí)
    scores = model.evaluate(x_test_transformed, verbose=1)

    # Reiniciar el generador para la predicción
    x_test_transformed.reset()
    y_pred = model.predict(x_test_transformed, verbose=1)

    # Calcular métricas
    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    precision, recall, fscore, support = precision_recall_fscore_support(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    # Guardar resultados
    resultados[prueba] = {
        'Test Score': scores[0],
        'Test Accuracy': scores[1],
        'Confusion Matrix': conf_matrix.tolist(),
        'Precision': precision.tolist(),
        'Recall': recall.tolist(),
        'F-Score': fscore.tolist(),
        'Support': support.tolist()
    }

# Crear un DataFrame de pandas para cada prueba y guardar en Excel
with pd.ExcelWriter('resultados_pruebas_estres.xlsx') as writer:
    for nombre_prueba, datos in resultados.items():
        df = pd.DataFrame(datos)
        df.to_excel(writer, sheet_name=nombre_prueba)
