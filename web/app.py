from flask import Flask, render_template, request, jsonify
from PIL import Image
import os
import json
import numpy as np
from threading import Lock
import matplotlib.pyplot as plt

# Se importan las bibliotecas necesarias de TensorFlow y PIL
import tensorflow as tf

# Se define la clase TFModel


class TFModel:
    def __init__(self, dir_path='./model') -> None:
        # Directorio donde se encuentra el modelo
        self.model_dir = os.path.dirname('./model/')
        # Se asegura de que la carpeta de SavedModel exportado exista
        with open(os.path.join(self.model_dir, "signature.json"), "r") as f:
            self.signature = json.load(f)
        self.model_file = os.path.join(
            self.model_dir, self.signature.get("filename"))
        if not os.path.isfile(self.model_file):
            raise FileNotFoundError(f"Model file does not exist")
        self.inputs = self.signature.get("inputs")
        self.outputs = self.signature.get("outputs")
        self.lock = Lock()

        # Carga del modelo guardado
        self.model = tf.saved_model.load(
            tags=self.signature.get("tags"), export_dir=self.model_dir)
        self.predict_fn = self.model.signatures["serving_default"]

        # Busca la versión en el archivo de firma.
        # Si no se encuentra o no coincide con la esperada, imprime un mensaje
        version = self.signature.get("export_model_version")
        if version is None or version != EXPORT_MODEL_VERSION:
            print(
                f"There has been a change to the model format. Please use a model with a signature 'export_model_version' that matches {EXPORT_MODEL_VERSION}."
            )

    def predict(self, image: Image.Image) -> dict:
        # Preprocesamiento de la imagen antes de pasarla al modelo
        image = self.process_image(
            image, self.inputs.get("Image").get("shape"))

        with self.lock:
            # Crear el diccionario de alimentación que es la entrada al modelo
            feed_dict = {}
            # Primero, agregar nuestra imagen al diccionario (viene de nuestro archivo signature.json)
            feed_dict[list(self.inputs.keys())[0]
                      ] = tf.convert_to_tensor(image)
            # ¡Ejecutar el modelo!
            outputs = self.predict_fn(**feed_dict)
            # Devolver la salida procesada
            return self.process_output(outputs)

    def process_image(self, image, input_shape) -> np.ndarray:
        """
        Dada una imagen PIL, recorta el centro cuadrado y la redimensiona para que se ajuste a la entrada del modelo esperada,
        y convierte los valores de [0,255] a [0,1].
        """
        width, height = image.size
        # Asegurarse de que el tipo de imagen sea compatible con el modelo y convertir si no lo es
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Recortar la imagen al centro (puedes sustituir cualquier otro método para hacer una imagen cuadrada, como simplemente redimensionar o rellenar los bordes con 0)
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            # Recortar el centro de la imagen
            image = image.crop((left, top, right, bottom))
        # Ahora la imagen es cuadrada, redimensionarla para que tenga la forma correcta para la entrada del modelo
        input_width, input_height = input_shape[1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))

        # Convertir de 0-255 entero a flotante de 0-1 (que PIL Image carga de forma predeterminada)
        image = np.asarray(image) / 255.0
        # Añadir una dimensión de lote adicional según lo esperado por el modelo
        return np.expand_dims(image, axis=0).astype(np.float32)

    def process_output(self, outputs) -> dict:
        # Realizar un poco de postprocesamiento
        out_keys = ["label", "confidence"]
        results = {}
        # Dado que realmente se ejecutó en un lote de tamaño 1, extraer los elementos de los arrays numpy devueltos
        for key, tf_val in outputs.items():
            val = tf_val.numpy().tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        confs = results["Confidences"]
        labels = self.signature.get("classes").get("Label")
        output = [dict(zip(out_keys, group)) for group in zip(labels, confs)]
        sorted_output = {"predictions": sorted(
            output, key=lambda k: k["confidence"], reverse=True)}
        return sorted_output


# Directorio donde se encuentran los modelos exportados y las imágenes de entrada
MODEL_DIRECTORY = "models"
EXPORT_MODEL_VERSION = 1

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image provided'})

    # Guardar la imagen temporalmente
    temp_image_path = "temp_image.jpg"
    image_file.save(temp_image_path)

    try:
        image = Image.open(temp_image_path)
        model = TFModel(dir_path=MODEL_DIRECTORY)
        outputs = model.predict(image)

        # Procesar los resultados para obtener etiquetas y confianzas
        labels = [item['label'] for item in outputs['predictions']]
        confidences = [item['confidence'] for item in outputs['predictions']]

        # Crear un gráfico de barras
        plt.figure(figsize=(10, 6))
        plt.bar(labels, confidences)
        plt.xlabel('Label')
        plt.ylabel('Confidence')
        plt.title('Prediction Results')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Guardar el gráfico en un archivo temporal
        plt_path = "temp_plot.png"
        plt.savefig(plt_path)

        # Devolver la imagen del gráfico
        with open(plt_path, 'rb') as f:
            plot_data = f.read()

        # Eliminar el archivo temporal del gráfico
        os.remove(plt_path)

        return plot_data, 200, {'Content-Type': 'image/png'}

    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        # Cerrar la imagen después de haberla utilizado
        image.close()
        # Eliminar la imagen temporal después de usarla
        os.remove(temp_image_path)


if __name__ == '__main__':
    app.run(debug=True)
