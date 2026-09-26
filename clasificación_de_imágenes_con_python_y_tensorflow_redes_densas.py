# -*- coding: utf-8 -*-
"""Clasificación de imágenes con Python y TensorFlow - Redes Densas

Adapted from the TensorFlow "Basic classification of Fashion-MNIST" tutorial
(https://www.tensorflow.org/tutorials/keras/basic_classification) and
originally run in Google Colab, now running locally.

Configuracion: valores en el archivo .env (ver .env.example)
"""

import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Leer .env ANTES de importar tensorflow para que TF_CPP_MIN_LOG_LEVEL aplique.
# Sin override: si la variable ya esta en el entorno del sistema, esa gana.
load_dotenv()

EPOCHS = int(os.getenv("EPOCHS", "5"))
TAMANO_LOTE = int(os.getenv("TAMANO_LOTE", "32"))
SEMILLA = int(os.getenv("SEMILLA", "42"))
MOSTRAR_GRAFICAS = os.getenv("MOSTRAR_GRAFICAS", "1") == "1"
GRAFICAS_DIR = Path(os.getenv("GRAFICAS_DIR", "graficas"))
MODELO_H5 = os.getenv("MODELO_H5", "modelo_exportado.h5")
RUTA_SALIDA_TFJS = Path(os.getenv("RUTA_SALIDA_TFJS", "modelo"))

# Semilla para que el entrenamiento sea reproducible
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
random.seed(SEMILLA)

import numpy as np
import tensorflow as tf

tf.random.set_seed(SEMILLA)
np.random.seed(SEMILLA)

# En modo headless no hay backend interactivo: usamos Agg y guardamos PNG
if MOSTRAR_GRAFICAS:
    import matplotlib.pyplot as plt
else:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


def mostrar_o_guardar(figura, nombre):
    """Muestra la figura si hay escritorio; si no, la guarda como PNG."""
    if MOSTRAR_GRAFICAS:
        plt.show()
    else:
        GRAFICAS_DIR.mkdir(parents=True, exist_ok=True)
        ruta = GRAFICAS_DIR / f"{nombre}.png"
        figura.savefig(ruta, dpi=120, bbox_inches="tight")
        print(f"[grafico] guardado en {ruta}")


# Descargar set de datos de Fashion MNIST de Zalando
(datos_entrenamiento, etiquetas_entreno), (datos_pruebas, etiquetas_prueba) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

print(f"Entrenamiento: {datos_entrenamiento.shape} -> {len(datos_entrenamiento)} imagenes")
print(f"Pruebas:       {datos_pruebas.shape} -> {len(datos_pruebas)} imagenes")

# Etiquetas de las 10 categorias posibles
nombres_clases = [
    "Camiseta",
    "Pantalón",
    "Suéter",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatilla",
    "Bolso",
    "Botín",
]

# Funcion de preprocesamiento para los datos: normalizar y añadir dimension de canal
def preprocesar(imagenes, etiquetas):
    # Asegurarse de que las imagenes esten en float32 y normalizadas a [0, 1]
    imagenes = tf.cast(imagenes, tf.float32)  # Cast to float32
    imagenes = imagenes / 255.0  # Normalize to 0-1 (this is the ONLY normalization step)
    imagenes = tf.expand_dims(imagenes, axis=-1)  # (28, 28) -> (28, 28, 1)
    etiquetas = tf.cast(etiquetas, tf.int64)  # SparseCategoricalCrossentropy
    return imagenes, etiquetas


# Convertir los arreglos de NumPy a objetos Dataset de TensorFlow
dataset_entreno = tf.data.Dataset.from_tensor_slices((datos_entrenamiento, etiquetas_entreno))
dataset_prueba = tf.data.Dataset.from_tensor_slices((datos_pruebas, etiquetas_prueba))

# Aplicar la funcion de preprocesamiento a los datasets
dataset_entreno = dataset_entreno.map(preprocesar, num_parallel_calls=tf.data.AUTOTUNE)
dataset_prueba = dataset_prueba.map(preprocesar, num_parallel_calls=tf.data.AUTOTUNE)

# Aplicar Cache, Shuffle (mezclar) y Batch (lotes) para optimizar el entrenamiento
# El set de pruebas no se mezcla ni se repite: se evalua tal cual
dataset_prueba = dataset_prueba.cache().batch(TAMANO_LOTE).prefetch(tf.data.AUTOTUNE)

# Mostrar una imagen de los datos de pruebas, de momento mostremos la primera
# El dataset de entrenamiento todavia no esta en lotes, asi que take(1) da 1 imagen
for imagenes, etiqueta in dataset_entreno.take(1):
    break
imagen = imagenes[0].numpy()  # (28, 28, 1)

# Dibujar
fig, ax = plt.subplots()
imagen_pintada = ax.imshow(imagen, cmap=plt.cm.binary)
fig.colorbar(imagen_pintada, ax=ax, fraction=0.046, pad=0.04)
ax.grid(False)
fig.tight_layout()
mostrar_o_guardar(fig, "01_imagen_entrada")

# Crear el modelo
modelo = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(28, 28, 1)),  # Capa de entrada explicita
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation=tf.nn.relu),
        tf.keras.layers.Dense(50, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),  # Para redes de clasificacion
    ]
)

modelo.summary()

# Compilar el modelo
modelo.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

# Los numeros de datos en entrenamiento y pruebas (60k y 10k)
num_ej_entrenamiento = datos_entrenamiento.shape[0]
num_ej_pruebas = datos_pruebas.shape[0]

# Cache + Shuffle + Repeat + Batch para el entrenamiento.
# repeat() se aplica ANTES de batch() a proposito: si se hiciera despues, el
# batch se aplicaria dos veces (lotes de 32 x 32 = 1024) y el entrenamiento
# seria 32 veces mas lento en CPU.
dataset_entreno = (
    dataset_entreno.cache()
    .shuffle(num_ej_entrenamiento, reshuffle_each_iteration=True)
    .repeat()
    .batch(TAMANO_LOTE)
    .prefetch(tf.data.AUTOTUNE)
)

historial = modelo.fit(
    dataset_entreno,
    epochs=EPOCHS,
    steps_per_epoch=math.ceil(num_ej_entrenamiento / TAMANO_LOTE),
    # El dataset ya viene mezclado con .shuffle(), no hace falta que Keras lo haga
    shuffle=False,
)

# Ver la funcion de perdida
fig, ax = plt.subplots()
ax.plot(historial.history["loss"])
ax.set_xlabel("# Epoca")
ax.set_ylabel("Magnitud de perdida")
ax.set_title("Perdida: entrenamiento vs pruebas")
ax.plot(historial.history.get("val_loss", []), label="pruebas")
ax.legend()
fig.tight_layout()
mostrar_o_guardar(fig, "02_perdida")

# Evaluar el modelo sobre el set de pruebas
resultado = modelo.evaluate(dataset_prueba, steps=math.ceil(num_ej_pruebas / TAMANO_LOTE))
print(f"Perdida en pruebas: {resultado[0]:.4f} | Exactitud en pruebas: {resultado[1]:.4f}")

# Pintar una cuadricula con varias predicciones, y marcar si fue correcta (azul) o incorrecta (roja)
for imagenes_prueba, etiquetas_prueba in dataset_prueba.take(1):
    imagenes_prueba = imagenes_prueba.numpy()
    etiquetas_prueba = etiquetas_prueba.numpy()
    predicciones = modelo.predict(imagenes_prueba, verbose=0)


def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
    arr_predicciones, etiqueta_real, img = (
        arr_predicciones[i],
        etiquetas_reales[i],
        imagenes[i],
    )
    ax = plt.gca()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.imshow(img[..., 0], cmap=plt.cm.binary)

    etiqueta_prediccion = int(np.argmax(arr_predicciones))
    color = "blue" if etiqueta_prediccion == etiqueta_real else "red"

    ax.set_xlabel(
        "{} {:2.0f}% ({})".format(
            nombres_clases[etiqueta_prediccion],
            100 * np.max(arr_predicciones),
            nombres_clases[etiqueta_real],
        ),
        color=color,
    )


def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
    arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
    ax = plt.gca()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    grafica = ax.bar(range(10), arr_predicciones, color="#777777")
    ax.set_ylim([0, 1])
    etiqueta_prediccion = int(np.argmax(arr_predicciones))

    grafica[etiqueta_prediccion].set_color("red")
    grafica[etiqueta_real].set_color("blue")


filas = 5
columnas = 5
num_imagenes = filas * columnas
fig, axs = plt.subplots(filas, 2 * columnas, figsize=(2 * columnas, 2 * filas))
for i in range(num_imagenes):
    graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
    axs[i // columnas, 2 * (i % columnas)].axis("off")

    graficar_valor_arreglo(i, predicciones, etiquetas_prueba)
    axs[i // columnas, 2 * (i % columnas) + 1].axis("off")

fig.suptitle("Predicciones: azul = correcta, rojo = incorrecta")
fig.tight_layout()
mostrar_o_guardar(fig, "03_predicciones")

# Probar una imagen suelta
imagen = imagenes_prueba[8]
imagen = np.array([imagen])
prediccion = modelo.predict(imagen, verbose=0)

print("Prediccion: " + nombres_clases[np.argmax(prediccion[0])])

# Exportacion del modelo a h5
modelo.save(MODELO_H5)
print(f"[modelo] guardado en {MODELO_H5}")

# Conversion del h5 a TensorFlow.js con tensorflowjs_converter.
# tensorflowjs no se puede instalar en Windows nativo: su dependencia
# tensorflow-decision-forests no publica wheels para win_amd64 en ninguna
# version. En Windows usa convertir_a_tfjs.ipynb en Google Colab; en WSL2,
# Linux o macOS usa convertir_a_tfjs.py.
CONVERSOR = "tensorflowjs_converter"
dir_temporal = Path("tfjs_target_dir")

if shutil.which(CONVERSOR) is None:
    print("=" * 70)
    print("[tfjs] tensorflowjs_converter no esta instalado en este interprete.")
    print("[tfjs] Motivo: tensorflowjs depende de tensorflow-decision-forests,")
    print("[tfjs] que no tiene wheels para Windows en ninguna version.")
    print("[tfjs]")
    print("[tfjs] WINDOWS  -> sube modelo_exportado.h5 a Google Colab y ejecuta")
    print("[tfjs]             convertir_a_tfjs.ipynb, luego descarga la carpeta modelo/.")
    print("[tfjs] WSL2/LINUX/MAC -> python -m pip install -r requirements-tfjs.txt")
    print("[tfjs]             python convertir_a_tfjs.py")
    print("[tfjs] (la carpeta modelo/ del repo ya tiene los archivos convertidos)")
    print("=" * 70)
else:
    if dir_temporal.exists():
        shutil.rmtree(dir_temporal)
    subprocess.run(
        [
            CONVERSOR,
            "--input_format=keras",
            MODELO_H5,
            str(dir_temporal),
        ],
        check=True,
    )

    # Copiar los archivos convertidos a la carpeta que usa el HTML (./modelo/model.json)
    RUTA_SALIDA_TFJS.mkdir(parents=True, exist_ok=True)
    for archivo in dir_temporal.iterdir():
        destino = RUTA_SALIDA_TFJS / archivo.name
        shutil.copy2(archivo, destino)
        print(f"[tfjs] {destino}")
    print(f"[tfjs] archivos copiados a: {RUTA_SALIDA_TFJS.resolve()}")

    shutil.rmtree(dir_temporal)

print("Listo.")
sys.exit(0)
