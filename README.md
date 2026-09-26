# Clasificación de imágenes con Python y TensorFlow — Redes Densas

Clasificador de imágenes de **Fashion-MNIST** (10 categorías) con una red densa
de Keras, más una interfaz web en HTML/JS que permite dibujar una prenda con el
mouse y clasificarla en el navegador con TensorFlow.js.

Adaptado del tutorial oficial de TensorFlow
(*Basic classification of Fashion-MNIST*) y del trabajo original en Google Colab,
que fue convertido a un script ejecutable en local.

---

## 1. Estructura

| Archivo | Descripción |
|---|---|
| `clasificación_de_imágenes_con_python_y_tensorflow_redes_densas.py` | Script principal: descarga los datos, entrena la red, evalúa, grafica y exporta el modelo. |
| `clasificador_fashion_mnist_dibujo.html` | Interfaz web. Carga el modelo de `./modelo/model.json` y clasifica lo que se dibuje. |
| `modelo/` | Modelo convertido a TensorFlow.js (`model.json` + `group1-shard1of1.bin`). Es lo que lee el HTML. |
| `requirements.txt` | Dependencias del entrenamiento. |
| `requirements-tfjs.txt` | Dependencias del conversor a TF.js (**solo Linux/macOS/WSL2**). |
| `.env` | Configuración del entrenamiento. |
| `.env.example` | Plantilla de `.env` para reposar. |
| `.gitignore` | Excluye venv, caché de Python, `.h5` y carpetas temporales del conversor. |
| `convertir_a_tfjs.py` | Convierte el `.h5` a TensorFlow.js (Linux/macOS/WSL2). |
| `convertir_a_tfjs.ipynb` | Notebook de Colab para hacer la conversión en Windows. |
| `Proyecto_1_Analisis_Clasificador_Fashion_MNIST.docx` | Informe de la actividad. |
| `README.md` | Instrucciones generales del proyecto. |
| `CAMBIOS.md` | Detalle de **qué cambió respecto al script original del repo y por qué**. |

Regenerado en cada ejecución: `modelo_exportado.h5` (540 KB, no versionado).
Los PNG de `graficas/` **sí** están versionados: son la evidencia de la corrida
verificada (87.02 % en pruebas, 5 épocas, `SEMILLA=42`).

---

## 2. Requisitos

* **Python 3.13.15** (probado también con el `python` global del sistema).
* No se necesita tarjeta gráfica. TensorFlow 2.21 no usa GPU en Windows nativo.

### Instalar el entorno

> **Importante:** el venv **no** debe crearse dentro de esta carpeta.
> La ruta del proyecto supera los 260 caracteres y Windows no tiene activados
> los "long paths", por lo que la instalación de TensorFlow falla con
> `WinError 32` al copiar archivos de `include/`.

```powershell
python -m venv C:\venvs\deep
C:\venvs\deep\Scripts\python.exe -m pip install --upgrade pip
C:\venvs\deep\Scripts\python.exe -m pip install -r requirements.txt
```

Versiones instaladas y verificadas:

| Paquete | Versión |
|---|---|
| Python | 3.13.15 |
| tensorflow | 2.21.0 |
| keras (tf) | 3.15.1 |
| numpy | 2.5.3 |
| matplotlib | 3.11.2 |
| python-dotenv | 1.2.3 |

---

## 3. Cómo ejecutar

### 3.1 Entrenar el modelo

```powershell
C:\venvs\deep\Scripts\python.exe ".\clasificación_de_imágenes_con_python_y_tensorflow_redes_densas.py"
```

Salida esperada (5 épocas, valores reales medidos):

```
Entrenamiento: (60000, 28, 28) -> 60000 imagenes
Pruebas:       (10000, 28, 28) -> 10000 imagenes
 Total params: 42,310 (165.27 KB)
Epoch 1/5 ... Epoch 5/5
Perdida en pruebas: 0.3627 | Exactitud en pruebas: 0.8702
Prediccion: Sandalia
[grafico] guardado en graficas\01_imagen_entrada.png
[grafico] guardado en graficas\02_perdida.png
[grafico] guardado en graficas\03_predicciones.png
[modelo] guardado en modelo_exportado.h5
Listo.
```

Progresión de exactitud en pruebas medida en esta máquina:

| Épocas | Exactitud en pruebas |
|---|---|
| 1 | 83.53 % |
| 2 | 86.42 % |
| 5 | **87.02 %** |

### 3.2 Abrir la interfaz web

El HTML **no** se puede abrir con doble clic. Con `file://` el navegador
bloquea la carga de `model.json` por CORS y el modelo nunca carga. Hay que
servirlo por HTTP:

```powershell
C:\venvs\deep\Scripts\python.exe -m http.server 8000
```

Luego abrir <http://127.0.0.1:8000/clasificador_fashion_mnist_dibujo.html>,
dibujar sobre el lienzo y presionar **«Clasificar dibujo»**.

Verificado en Chrome: el modelo carga con entrada `[null,28,28,1]` y salida
`[null,10]`, y una predicción real devolvió `Sandalia` con 80.63 % de
confianza.

---

## 4. Configuración (`.env`)

Se lee con `python-dotenv` **antes** de importar TensorFlow, para que
`TF_CPP_MIN_LOG_LEVEL` surta efecto.

| Variable | Por defecto | Descripción |
|---|---|---|
| `EPOCHS` | `5` | Número de épocas. |
| `TAMANO_LOTE` | `32` | Tamaño de lote. |
| `SEMILLA` | `42` | Semilla para resultados reproducibles. |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | Silencio del log interno de TF. |
| `MOSTRAR_GRAFICAS` | `1` | `1` abre las ventanas de matplotlib; `0` guarda PNG y no abre nada. |
| `GRAFICAS_DIR` | `graficas` | Carpeta de los PNG. |
| `MODELO_H5` | `modelo_exportado.h5` | Ruta del modelo exportado. |
| `RUTA_SALIDA_TFJS` | `modelo` | Carpeta del modelo TF.js. Debe coincidir con `MODEL_URL` del HTML. |

`load_dotenv()` se llama **sin** `override`, así que una variable ya presente en
el entorno del sistema tiene prioridad sobre el `.env`. Eso permite pruebas
rápidas sin editar el archivo:

```powershell
$env:EPOCHS="1"; $env:MOSTRAR_GRAFICAS="0"
C:\venvs\deep\Scripts\python.exe ".\clasificación_de_imágenes_con_python_y_tensorflow_redes_densas.py"
```

`MOSTRAR_GRAFICAS=0` es obligatorio en consola sin escritorio: con `1`, el
`plt.show()` se queda esperando a que se cierre la ventana.

---

## 5. Cambios aplicados al script original

El archivo venía exportado desde Colab y no era ejecutable en local. Resumen de
lo corregido:

1. **Código exclusivo de Colab eliminado.** `!pip install`, `!mkdir`, `!ls`,
   `from google.colab import drive` y `drive.mount()` no existen en Python
   normal. Se reemplazaron por `os`, `shutil` y `subprocess`.
2. **`TF_USE_LEGACY_KERAS=1` eliminado.** Con TensorFlow 2.21 (Keras 3) esa
   variable rompe el import, porque obliga a usar el paquete `tf_keras`, que no
   está instalado. El modelo no lo necesita: `Sequential`, `Flatten`, `Dense`,
   `Input`, `compile` y `save` funcionan igual en Keras 3.
3. **Bug de doble `batch()` corregido.** El script aplicaba `.batch(32)` dos
   veces sobre el mismo dataset (el patrón venía del tutorial oficial), lo que
   generaba lotes reales de 32 × 32 = 1024 muestras y hacía el entrenamiento
   **32 veces más lento en CPU** sin ganar nada. Ahora `repeat()` va antes de
   `batch()`.
4. **`tensorflow-datasets` eliminado.** Se importaba pero nunca se usaba; los
   datos se cargan con `tf.keras.datasets.fashion_mnist`.
5. **`import tensorflow` duplicado eliminado** (aparecía dos veces).
6. **`plt.show()` reemplazado** por `mostrar_o_guardar()`, que guarda PNG en
   modo headless y añade `tight_layout()` para que las etiquetas no se corten.
7. **Se añadió `modelo.evaluate()`** sobre el set de pruebas y el trazado de
   `val_loss` en la gráfica de pérdida.
8. **Semillas fijadas** (`random`, `np.random`, `tf.random`) para que el
   entrenamiento sea reproducible.
9. **`shuffle=False` en `model.fit()`**, porque el dataset ya viene mezclado con
   `.shuffle()` y Keras emitía un warning al respecto.
10. **Gráficas migradas a la API orientada a objetos** (`fig, ax = plt.subplots()`).
    En el original la curva de pérdida se dibujaba sobre el axes de la imagen
    con su colorbar, porque no se creaba una figura nueva antes del `plt.plot()`.

El desglose completo, con líneas del original, motivo de cada cambio y lo que
**no** se tocó, está en **[`CAMBIOS.md`](CAMBIOS.md)**.

---

## 6. Limitación conocida: `tensorflowjs` en Windows

**`tensorflowjs` no se puede instalar en Windows nativo, en ninguna versión de
Python.** No es un problema de versión: su dependencia
`tensorflow-decision-forests` **no publica wheels para `win_amd64`** (solo
`macosx` y `manylinux`), y `tensorflowjs` la importa a nivel de módulo.

Se probó `pip install --no-deps` con `tensorflowjs` 4.18, 4.19, 4.20, 4.21 y
4.22: las cinco fallan con
`ModuleNotFoundError: No module named 'tensorflow_decision_forests'`.

Por eso `tensorflowjs` **no está** en `requirements.txt` y el script principal
solo invoca el conversor si lo encuentra en el `PATH`; si no, muestra las
alternativas.

### Opciones para convertir el modelo

| Entorno | Cómo |
|---|---|
| **Windows (recomendado)** | Subir `modelo_exportado.h5` a [colab.research.google.com](https://colab.research.google.com) y ejecutar `convertir_a_tfjs.ipynb`. Descargar el `modelo.zip` y reemplazar la carpeta `modelo/`. |
| **WSL2 / Linux** | `python -m venv .venv-tfjs` → `pip install -r requirements-tfjs.txt` → `python convertir_a_tfjs.py`. |
| **macOS** | Igual que WSL2. |

> **Nota sobre la extensión de Colab para VS Code:** no sirve para esto. Ejecuta
> el notebook con un kernel Python **local**, así que `!pip install tensorflowjs`
> se ejecutaría en Windows y fallaría por la misma razón.

**No hace falta convertir nada para entregar:** la carpeta `modelo/` del
repositorio ya contiene `modelo.json` y `group1-shard1of1.bin`, y el HTML
funciona con ellos.

---

## 7. Solución de problemas

| Síntoma | Causa y solución |
|---|---|
| `WinError 32` al instalar TensorFlow | La ruta del proyecto supera 260 caracteres. Crear el venv fuera: `python -m venv C:\venvs\deep`. |
| `No module named 'tf_keras'` | Quedó el `TF_USE_LEGACY_KERAS=1`. Ya se eliminó; si vuelve a aparecer, no activates esa variable. |
| El script se queda colgado sin imprimir nada | `MOSTRAR_GRAFICAS=1` con `plt.show()` esperando una ventana. Usar `MOSTRAR_GRAFICAS=0`. |
| El HTML muestra «No fue posible cargar el modelo» | Se abrió con doble clic (`file://`). Servir por `python -m http.server 8000`. |
| Faltan pesos al predecir en el navegador | `model.json` y los `.bin` deben estar **juntos** dentro de `modelo/`. |
| El `.env` no cambia nada | Editar `.env`; las variables ya definidas en la consola tienen prioridad. |
