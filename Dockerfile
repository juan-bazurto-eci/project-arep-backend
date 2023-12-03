# Utiliza una imagen base oficial de Python
FROM python:3.11.5

# Establece un directorio de trabajo
WORKDIR /app

# Instala las librerías necesarias
# Nota: La lista de paquetes puede incluir otras librerías requeridas por tu aplicación.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copia los archivos necesarios al contenedor
COPY . /app

# Expone el puerto que tu aplicación usará
EXPOSE 5000

# Define el comando para ejecutar la aplicación
CMD ["python", "app.py"]