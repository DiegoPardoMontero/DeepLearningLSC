# Clasificación de LSC usando ResNet50 👋🤖👾

## Descripción del Proyecto

Este proyecto se enfoca en la clasificación de imágenes utilizando una arquitectura de red neuronal convolucional (CNN) basada en ResNet50 preentrenada. El modelo está diseñado para clasificar imágenes en diferentes categorías, tales como "anciano", "joven", "bebe", entre otras. El script de entrenamiento (`train.py`) carga un conjunto de datos de imágenes organizadas en directorios por clase, entrena el modelo y guarda tanto el modelo entrenado como el binarizador de etiquetas para su posterior uso.

<details>
  <summary><strong>Estructura del Proyecto</strong></summary>
  <ul>
    <li><strong>train.py</strong>: Script principal para entrenar el modelo.</li>
    <li><strong>model/</strong>: Directorio donde se guardará el modelo entrenado y el binarizador de etiquetas.</li>
    <li><strong>dataset/</strong>: Directorio que contiene las imágenes organizadas por clase.</li>
    <li><strong>plot.png</strong>: Gráfico de la pérdida y precisión del entrenamiento.</li>
  </ul>
</details>

<details>
  <summary><strong>Requisitos</strong></summary>
  <ul>
    <li>Python 3.x</li>
    <li>TensorFlow y Keras</li>
    <li>NumPy</li>
    <li>Scikit-learn</li>
    <li>OpenCV</li>
    <li>Imutils</li>
    <li>Matplotlib</li>
    <li>Argparse</li>
  </ul>
  <p>Puedes instalar las dependencias con el siguiente comando:</p>
  <pre>
  <code>bash
  pip install tensorflow keras numpy scikit-learn opencv-python imutils matplotlib argparse
  </code>
  </pre>
</details>

<details>
  <summary><strong>Estructura del Dataset</strong></summary>
  <pre>
  /dataset
  ├── anciano
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  ├── joven
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  └── ...
  </pre>
</details>

<details>
  <summary><strong>Uso</strong></summary>
  <p>Para entrenar el modelo, ejecuta el siguiente comando:</p>
  <pre>
  <code>bash
  python train.py --dataset /ruta/al/dataset --model model/activity.model --label-bin model/lb.pickle --epochs 100
  </code>
  </pre>
  <ul>
    <li><code>--dataset</code>: Ruta al directorio del conjunto de datos.</li>
    <li><code>--model</code>: Ruta para guardar el modelo entrenado.</li>
    <li><code>--label-bin</code>: Ruta para guardar el binarizador de etiquetas.</li>
    <li><code>--epochs</code>: Número de épocas para entrenar el modelo.</li>
  </ul>
</details>

<details>
  <summary><strong>Explicación del Script `train.py`</strong></summary>
  <ol>
    <li><strong>Importación de Paquetes y Configuración Inicial</strong>: Se importan las bibliotecas necesarias y se configura Matplotlib para no requerir una interfaz gráfica.</li>
    <li><strong>Argumentos del Script</strong>: Define los argumentos necesarios para la ejecución del script: ruta del dataset, ruta para guardar el modelo y el binarizador de etiquetas, y el número de épocas de entrenamiento.</li>
    <li><strong>Cargar Imágenes del Dataset</strong>: Se cargan las imágenes del directorio del dataset y se redimensionan a 224x224 píxeles. Las imágenes se normalizan y se almacenan junto con sus etiquetas correspondientes.</li>
    <li><strong>Binarizar las Etiquetas</strong>: Convierte las etiquetas de clase en una representación binaria utilizando <code>LabelBinarizer</code>.</li>
    <li><strong>Dividir Datos en Conjuntos de Entrenamiento y Prueba</strong>: Divide los datos en conjuntos de entrenamiento (75%) y prueba (25%).</li>
    <li><strong>Inicializar el Modelo ResNet50</strong>: Carga la arquitectura ResNet50 preentrenada con los pesos de ImageNet, excluyendo la capa superior.</li>
    <li><strong>Construir la Cabeza del Modelo</strong>: Añade capas adicionales a la salida de ResNet50 para adaptar el modelo a la tarea específica de clasificación.</li>
    <li><strong>Congelar las Capas del Modelo Base</strong>: Evita que las capas preentrenadas de ResNet50 se actualicen durante el entrenamiento inicial.</li>
    <li><strong>Compilar el Modelo</strong>: Configura el modelo con el optimizador SGD, la función de pérdida <code>categorical_crossentropy</code> y la métrica de precisión.</li>
    <li><strong>Entrenar el Modelo</strong>: Entrena el modelo utilizando los datos de entrenamiento y valida el rendimiento con los datos de prueba.</li>
    <li><strong>Evaluar el Modelo</strong>: Genera predicciones en el conjunto de prueba y evalúa el rendimiento del modelo.</li>
    <li><strong>Guardar el Modelo y el Binarizador de Etiquetas</strong>: Serializa y guarda el modelo entrenado y el binarizador de etiquetas.</li>
    <li><strong>Graficar la Pérdida y Precisión del Entrenamiento</strong>: Crea y guarda un gráfico de la pérdida y precisión durante el entrenamiento en <code>plot.png</code>.</li>
  </ol>
</details>

<details>
  <summary><strong>Resultados</strong></summary>
  <p>El modelo entrenado se evalúa y genera un informe de clasificación detallado. Además, se guarda un gráfico (<code>plot.png</code>) que muestra la pérdida y precisión del entrenamiento a lo largo de las épocas.</p>
</details>


### Fork de Proyecto [kerasDeepLSC](https://github.com/ypmunoz36/kerasDeepLSC)
### Archivo README de autoria propia!
