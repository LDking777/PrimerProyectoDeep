# Cambios respecto al script original del repositorio

Documento de referencia para quien quiera correr el proyecto. Explica **qué
cambió**, **por qué** y **cómo ejecutarlo**.

- **Original:** `clasificación_de_imágenes_con_python_y_tensorflow_redes_densas.py`
  tal como está en el commit `0d4bb8c` (exportado desde Google Colab).
- **Actual:** el mismo archivo, adaptado para ejecutarse en local/CLI.
- Diff: **1 archivo, +241 / -132 lineas.**

> Si solo quieres instrucciones de ejecución, ve a la [sección 4](#4-cómo-correrlo).
> Si quieres el detalle técnico de cada cambio, sigue leyendo desde la
> [sección 3](#3-cambio-por-cambio).

---

## 1. Resumen

El archivo original **no era ejecutable fuera de Colab**:

| Tipo | Cantidad |
|---|---|
| Bloqueantes (el script ni siquiera arranca) | 4 |
| Bugs de lógica /orrectness | 5 |
| Mejoras de reproducibilidad y observabilidad | 6 |
| Archivos nuevos de soporte | 7 |

Además se agregaron los archivos de infraestructura que faltaban
(`requirements.txt`, `.env`, `.gitignore`) y dos vías para resolver el
`tensorflowjs` en Windows, que es el problema de portability más delicado del
proyecto.

---

## 2. Bugs y bloqueantes

### 2.1 Código exclusivo de Colab (4 bloqueantes)

El original termina con 20 líneas que **solo funcionan dentro de Colab**:

```python
!pip install tensorflowjs          # línea 165
!mkdir tfjs_target_dir             # línea 168
!tensorflowjs_converter ...         # línea 169
!ls                                # línea 172
from google.colab import drive      # línea 178
drive.mount('/content/drive')      # línea 179
!cp -r tfjs_target_dir/* "..."      # línea 186
```

El `!` es **sintaxis de shell de IPython**. En Python normal es un
`SyntaxError` inmediato, así que el script moría en la línea 165, justo
**después** de haber entrenado el modelo. Peor: el error solo aparecía tras
varios minutos de entrenamiento, cuando ya no había nada que hacer.

**Qué se hizo:** se eliminó todo el bloque. La conversión a TensorFlow.js se
hizo con código de Python estándar (`subprocess`, `shutil`, `pathlib`) y las
alternativas para Windows se documentaron en
`convertir_a_tfjs.py` / `convertir_a_tfjs.ipynb`.

### 2.2 `TF_USE_LEGACY_KERAS=1` (bloqueante)

```python
os.environ["TF_USE_LEGACY_KERAS"] = "1"   # línea 12 del original
```

El original was exporting from Colab, where Keras 2 era el default. Con
**TensorFlow 2.21 (Keras 3)** esa variable rompe el import:

```
ModuleNotFoundError: No module named 'tf_keras'
```

porque obliga a Keras a delegar en el paquete `tf_keras`, que ya no se instala
por defecto.

**Qué se hizo:** se eliminó la variable. Verificado que **todo** lo que usa el
script funciona igual en Keras 3: `Sequential`, `Flatten`, `Dense`, `Input`,
`compile()` y `save()`. La arquitectura y el compilador **no cambiaron**.

### 2.3 `.batch()` aplicado dos veces (bug de rendimiento, 32×)

El original aplicaba el batch **dos veces sobre la misma variable**:

```python
# línea 51
dataset_entreno = dataset_entreno.cache().shuffle(60000).batch(32).prefetch(...)

# línea 90  <-- vuelve a aplicar batch sobre el dataset YA loteado
dataset_entreno = dataset_entreno.repeat().shuffle(60000).batch(32)
```

`.batch(32).batch(32)` produce lotes de **32 × 32 = 1024** muestras. Como
`steps_per_epoch` seguía siendo `ceil(60000/32) = 1875`, el modelo veía:

```
1875 steps × 1024 muestras = 1 920 000 muestras por época
```

en lugar de 60 000. **32 veces más trabajo en CPU** sin ningún beneficio. El
mismo error estaba en `dataset_prueba` (líneas 52 y 91).

> Este patrón viene del tutorial oficial de TensorFlow, donde el primer
> `.batch()` se aplica a un dataset efímero que luego se descarta, no al que
> se entrena. Al copiar el código a Colab se replicó sobre la misma variable.

**Qué se hizo:** una sola cadena, con `repeat()` **antes** de `batch()`:

```python
dataset_entreno = (dataset_entreno
                   .cache()
                   .shuffle(num_ej_entrenamiento, reshuffle_each_iteration=True)
                   .repeat()          # <-- antes del batch
                   .batch(TAMANO_LOTE)
                   .prefetch(tf.data.AUTOTUNE))
```

`dataset_prueba` quedó con un único `.batch()`, y ya **no** se mezcla ni se
repite (eso solo tiene sentido en entrenamiento).

### 2.4 La gráfica de pérdida se dibujaba encima de la imagen (bug visual)

El original usaba la API de estado de pyplot, sin figuras separadas:

```python
plt.figure()             # línea 62  -> figura A
plt.imshow(imagen, ...)  # línea 63
plt.colorbar()           # línea 64
plt.show()               # línea 66

# ... más adelante, SIN crear figura nueva:
plt.xlabel("# Epoca")            # línea 101
plt.ylabel("Magnitud de pérdida") # línea 102
plt.plot(historial.history["loss"])  # línea 103  <-- se dibuja sobre la imagen
```

`plt.plot()` sin `plt.figure()` previo escribe en el **axes actual**, que seguía
siendo el de la imagen con su colorbar. El resultado era un cuadro con la foto
de un pantalón, una barra de color y encima la curva de pérdida encima. La
"gráfica de pérdida" **nunca existió como gráfica propia**.

**Qué se hizo:** se migró todo el plot a la API orientada a objetos
(`fig, ax = plt.subplots()`), donde cada figura es explícita:

```python
fig, ax = plt.subplots()
ax.plot(historial.history["loss"], label="entrenamiento")
ax.plot(historial.history.get("val_loss", []), label="pruebas")
ax.set_xlabel("# Epoca"); ax.set_ylabel("Magnitud de perdida")
ax.set_title("Perdida: entrenamiento vs pruebas")
ax.legend()
fig.tight_layout()
mostrar_o_guardar(fig, "02_perdida")
```

### 2.5 `shuffle(60000)` con valor mágico

El original tenía `60000` escrito a mano en la línea 51. Si se cambiara el
dataset, el shuffle quedaría mal dimensionado. Ahora se usa
`num_ej_entrenamiento` (el `60000` sale de `datos_entrenamiento.shape[0]`).

### 2.6 `plt.show()` bloquea en consola sin escritorio

`plt.show()` espera a que se cierre la ventana. En un servidor, contenedor o
Docker sin display, el script **se queda colgado para siempre** después de
entrenar.

**Qué se hizo:** la función `mostrar_o_guardar()` decide según
`MOSTRAR_GRAFICAS`:

| `MOSTRAR_GRAFICAS` | Backend | Comportamiento |
|---|---|---|
| `1` (default) | interactivo | Abre las ventanas de matplotlib |
| `0` | `Agg` (headless) | Guarda `graficas/01_imagen_entrada.png`, `02_perdida.png`, `03_predicciones.png` y **no abre nada** |

Cuando está en `0` también se hace `matplotlib.use("Agg")` **antes** de
importar `pyplot`, que es el orden que matplotlib exige. El `savefig` usa
`bbox_inches="tight"` para que las etiquetas no se corten.

---

## 3. Cambio por cambio

### 3.1 Bloqueantes

| # | Original | Actual | Motivo |
|---|---|---|---|
| 1 | `!pip install`, `!mkdir`, `!ls`, `!cp`, `!tensorflowjs_converter`, `from google.colab import drive`, `drive.mount()`, rutas `/content/drive` | eliminados; se usan `shutil`, `subprocess`, `pathlib` | `SyntaxError` en Python normal |
| 2 | `os.environ["TF_USE_LEGACY_KERAS"] = "1"` | eliminada | rompe con TF 2.21 / Keras 3 (`No module named 'tf_keras'`) |
| 3 | `import tensorflow_datasets as tfds` | eliminada | se importaba y **nunca se usaba**; los datos vienen de `tf.keras.datasets.fashion_mnist`. Ahorra ~500 MB de dependencias |
| 4 | `import tensorflow as tf` dos veces (líneas 14 y 29) | una sola | el segundo no hacía nada; era residuo del bloque de Colab |
| 5 | `import matplotlib.pyplot` y `import numpy as np` en medio del archivo (líneas 59 y 106) | todas las importaciones arriba | estilo y legibilidad |

### 3.2 Corrección de lógica

| # | Cambio | Detalle |
|---|---|---|
| 6 | **Doble `.batch()` eliminado** | `.repeat()` antes de `.batch()`. 32× menos trabajo en CPU. Ver [§2.3](#23-batch-aplicado-dos-veces-bug-de-rendimiento-32) |
| 7 | **`dataset_prueba` sin doble batch** | un solo `.cache().batch().prefetch()`; sin `shuffle` ni `repeat` |
| 8 | **Gráficas con la API OO** | `fig, ax = plt.subplots()` en lugar del estado global de pyplot. Ver [§2.4](#24-la-gráfica-de-pérdida-se-dibujaba-encima-de-la-imagen-bug-visual) |
| 9 | **`reshape((28,28))` eliminado** | `imagenes[0].numpy()` ya devuelve `(28, 28, 1)`; el `reshape` del original era un resto del tutorial y solo servía para `imshow` |
| 10 | **`imshow` con `cmap` y ejes explícitos** | se añadió `ax.grid(False)`, sin ticks, `fig.tight_layout()` y `suptitle` con la leyenda de colores (azul = correcta, rojo = incorrecta), que antes no existía |
| 11 | **`shuffle=False` en `model.fit()`** | el dataset ya viene mezclado con `.shuffle()`; Keras emitía warning por hacer doble mezcla |
| 12 | **`verbose=0` en los `predict()`** | el original inundaba la consola con la barra de progreso de Keras |

### 3.3 Observabilidad y reproducibilidad

| # | Cambio | Detalle |
|---|---|---|
| 13 | **`modelo.summary()` añadido** | antes no se imprimía la arquitectura. Ahora: `Total params: 42,310 (165.27 KB)` |
| 14 | **`modelo.evaluate()` sobre el set de pruebas** | el original **nunca medía nada**: entrenaba y guardaba sin decir si el modelo servía. Ahora imprime `Perdida` y `Exactitud` reales |
| 15 | **`val_loss` en la gráfica** | con `evaluate()` ya hay `val_loss` en el historial; se traza para ver sobreajuste |
| 16 | **Semillas fijadas** | `random.seed()`, `np.random.seed()` y `tf.random.set_seed()` con `SEMILLA=42`. Sin esto, dos ejecuciones dan exactitudes distintas |
| 17 | **Metadata del dataset impresa** | `Entrenamiento: (60000, 28, 28) -> 60000 imagenes` |
| 18 | **Rutas de salida parametrizadas** | `GRAFICAS_DIR`, `MODELO_H5`, `RUTA_SALIDA_TFJS` en vez de literales |
| 19 | **`sys.exit(0)`** | código de salida explícito para CI |
| 20 | **Docstring actualizado** | ya no dice "Automatically generated by Colab"; cita la fuente del tutorial y apunta a `.env` |

### 3.4 Configuración externa (`.env`)

Todos los números que estaban escritos a mano pasaron a variables de entorno
leídas con `python-dotenv`, con estos defaults:

| Variable | Default | Antes era |
|---|---|---|
| `EPOCHS` | `5` | `epochs=5` hardcodeado |
| `TAMANO_LOTE` | `32` | `TAMANO_LOTE = 32` hardcodeado |
| `SEMILLA` | `42` | no existía |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | no existía (logs ruidosos de TF) |
| `MOSTRAR_GRAFICAS` | `1` | `plt.show()` siempre |
| `GRAFICAS_DIR` | `graficas` | no existía (no se guardaba nada) |
| `MODELO_H5` | `modelo_exportado.h5` | `modelo.save('modelo_exportado.h5')` |
| `RUTA_SALIDA_TFJS` | `modelo` | `/content/drive/MyDrive/tfjs_model_2` |

**Detalle de implementación importante:** `load_dotenv()` se llama **antes** de
`import tensorflow` y **sin** `override`. Esto significa que:

- `TF_CPP_MIN_LOG_LEVEL` del `.env` sí aplica (necesita estar en el entorno
  antes de que TF cargue sus librerías nativas);
- si una variable ya está definida en la consola, **esa gana** sobre el `.env`,
  lo que permite pruebas rápidas sin editar el archivo.

### 3.5 El problema de `tensorflowjs` en Windows

El original hacía `!pip install tensorflowjs` y eso funcionaba en Colab
(Linux). **En Windows nativo es imposible**, y no por versión de Python:

> `tensorflowjs` importa `tensorflow-decision-forests` a nivel de módulo, y ese
> paquete **no publica wheels para `win_amd64`** en ninguna versión (solo
> `macosx` y `manylinux`).

Se probó `pip install --no-deps` con `tensorflowjs` 4.18, 4.19, 4.20, 4.21 y
4.22: **las cinco** fallan con
`ModuleNotFoundError: No module named 'tensorflow_decision_forests'`.

Por eso `tensorflowjs` **no está** en `requirements.txt`, y el script principal
solo invoca el conversor si lo encuentra en el `PATH`:

```python
if shutil.which(CONVERSOR) is None:
    print("[tfjs] tensorflowjs_converter no esta instalado en este interprete.")
    print("[tfjs] WINDOWS  -> sube modelo_exportado.h5 a Google Colab y ejecuta")
    print("[tfjs]              convertir_a_tfjs.ipynb, luego descarga la carpeta modelo/.")
    ...
else:
    subprocess.run([CONVERSOR, "--input_format=keras", MODELO_H5, str(dir_temporal)], check=True)
    # copia los archivos de tfjs_target_dir/ a RUTA_SALIDA_TFJS/
```

Se agregaron dos vías según el sistema:

| Entorno | Vía |
|---|---|
| **Windows** | `convertir_a_tfjs.ipynb` en Google Colab (sube el `.h5`, convierte, descarga `modelo.zip`) |
| **WSL2 / Linux / macOS** | `convertir_a_tfjs.py` con `requirements-tfjs.txt` en un venv aparte (Python 3.11, porque `tensorflowjs` pide `tensorflow~=2.15` y choca con `tensorflow==2.21`) |

> La extensión de Colab para VS Code **no** sirve para esto: ejecuta el
> notebook con un kernel Python **local**, así que `!pip install tensorflowjs`
> se ejecutaría en Windows y fallaría igual.

### 3.6 Archivos nuevos

| Archivo | Qué es |
|---|---|
| `requirements.txt` | Dependencias del entrenamiento, con versiones fijadas y verificadas |
| `requirements-tfjs.txt` | Dependencias **solo** del conversor a TF.js (Linux/macOS/WSL2) |
| `.env` | Configuración de la ejecución local |
| `.env.example` | Plantilla de `.env` para quien clonea el repo |
| `.gitignore` | Excluye `.venv/`, `__pycache__/`, `*.pyc`, `graficas/`, `modelo_exportado.h5` |
| `convertir_a_tfjs.py` | Conversor standalone (Linux/macOS/WSL2) |
| `convertir_a_tfjs.ipynb` | Conversor en Colab (Windows) |
| `README.md` | Instrucciones generales del proyecto |
| `CAMBIOS.md` | Este documento |

### 3.7 Lo que **NO** cambió

Importante para la evaluación: la lógica de la red es idéntica al original.

- Arquitectura: `Flatten → Dense(50, relu) → Dense(50, relu) → Dense(10, softmax)`
- Compilador: `adam` + `SparseCategoricalCrossentropy` + `accuracy`
- Preprocesamiento: `/255.0` + `expand_dims` a `(28,28,1)` (sin normalización extra)
- Dataset: `fashion_mnist` de Keras, 60 000 / 10 000
- Gráfica de predicciones: misma rejilla 5×5 con barras de probabilidad
- `clasificador_fashion_mnist_dibujo.html`: **sin modificar**
- `modelo/model.json` y `modelo/group1-shard1of1.bin`: **sin modificar**

El **accuracy reportado es real y medido**, no copiado: el entrenamiento se
ejecutó en local y dio **87.02 %** en pruebas con 5 épocas.

| Épocas | Exactitud en pruebas |
|---|---|
| 1 | 83.53 % |
| 2 | 86.42 % |
| 5 | **87.02 %** |

---

## 4. Cómo correrlo

### 4.1 Requisitos

- **Python 3.13** (probado con 3.13.15)
- No requiere GPU. TensorFlow 2.21 no usa GPU en Windows nativo.

> **Importante en Windows:** la ruta de esta carpeta supera los 260 caracteres y
> Windows no tiene activados los *long paths*, por lo que el venv **no debe
> crearse dentro del proyecto**. Si se crea dentro, la instalación de
> TensorFlow falla con `WinError 32` al copiar `include/`.

### 4.2 Instalar

```powershell
python -m venv C:\venvs\deep
C:\venvs\deep\Scripts\python.exe -m pip install --upgrade pip
C:\venvs\deep\Scripts\python.exe -m pip install -r requirements.txt
```

Versiones verificadas: `tensorflow==2.21.0`, `keras 3.15.1`, `numpy==2.5.3`,
`matplotlib==3.11.2`, `python-dotenv==1.2.3`.

En Linux/macOS el venv sí puede ir dentro (`.venv/` ya está en `.gitignore`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4.3 Entrenar

```powershell
C:\venvs\deep\Scripts\python.exe ".\clasificación_de_imágenes_con_python_y_tensorflow_redes_densas.py"
```

Salida esperada:

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

### 4.4 Probar rápido (1 época, sin ventanas)

```powershell
$env:EPOCHS="1"; $env:MOSTRAR_GRAFICAS="0"
C:\venvs\deep\Scripts\python.exe ".\clasificación_de_imágenes_con_python_y_tensorflow_redes_densas.py"
```

### 4.5 Usar la interfaz web

**No se puede abrir con doble clic.** Con `file://` el navegador bloquea la
carga de `model.json` por CORS y el modelo nunca carga. Hay que servirlo por
HTTP:

```powershell
C:\venvs\deep\Scripts\python.exe -m http.server 8000
```

Luego abrir <http://127.0.0.1:8000/clasificador_fashion_mnist_dibujo.html>,
dibujar una prenda y presionar **«Clasificar dibujo»**.

> **No hace falta convertir nada para entregar:** la carpeta `modelo/` del
> repositorio ya tiene `model.json` y `group1-shard1of1.bin`, y el HTML
> funciona con ellos. La conversión solo hace falta si **reentrenas** y quieres
> actualizar el modelo del navegador.

### 4.6 Reconvertir el modelo (opcional)

**Windows** → subir `modelo_exportado.h5` a
[colab.research.google.com](https://colab.research.google.com), ejecutar
`convertir_a_tfjs.ipynb`, descargar `modelo.zip` y reemplazar la carpeta
`modelo/`.

**WSL2 / Linux / macOS:**

```bash
python3.11 -m venv .venv-tfjs
./.venv-tfjs/bin/pip install -r requirements-tfjs.txt
./.venv-tfjs/bin/python convertir_a_tfjs.py
```

---

## 5. Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `WinError 32` al instalar TensorFlow | La ruta del proyecto supera 260 caracteres. Crear el venv **fuera**: `python -m venv C:\venvs\deep` |
| `No module named 'tf_keras'` | Quedó un `TF_USE_LEGACY_KERAS=1` en el entorno. Hay que quitarlo: `Remove-Item Env:\TF_USE_LEGACY_KERAS` |
| `No module named 'tensorflow_decision_forests'` | Se intentó instalar `tensorflowjs` en Windows. No es posible; usar el notebook de Colab |
| El script se cuelga sin imprimir nada | `MOSTRAR_GRAFICAS=1` con `plt.show()` esperando una ventana. Usar `MOSTRAR_GRAFICAS=0` |
| El HTML dice «No fue posible cargar el modelo» | Se abrió con doble clic (`file://`). Servir con `python -m http.server 8000` |
| Faltan pesos al predecir en el navegador | `model.json` y el `.bin` deben estar **juntos** dentro de `modelo/` |
| El `.env` no cambia nada | Las variables ya definidas en la consola tienen prioridad sobre el `.env` (por diseño) |
| El accuracy sale distinto al del README | No se fijaron las semillas: Revisa que `SEMILLA` esté definida. Si cambiaste `EPOCHS` o `TAMANO_LOTE`, el número cambia legítimamente |
