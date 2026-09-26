# -*- coding: utf-8 -*-
"""Convierte modelo_exportado.h5 (Keras) a TensorFlow.js.

NO funciona en Windows nativo: tensorflowjs depende de tensorflow-decision-forests,
que no publica wheels para win_amd64 en ninguna version.

  Windows        -> usa convertir_a_tfjs.ipynb en Google Colab
  WSL2 / Linux   -> python -m venv .venv-tfjs
                    ./.venv-tfjs/bin/pip install -r requirements-tfjs.txt
                    ./.venv-tfjs/bin/python convertir_a_tfjs.py
  macOS          -> igual que WSL2, con el venv en .venv-tfjs

Escribe model.json y los .bin en la carpeta indicada en RUTA_SALIDA_TFJS,
que es la que lee clasificador_fashion_mnist_dibujo.html (./modelo/model.json).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODELO_H5 = Path(os.getenv("MODELO_H5", "modelo_exportado.h5"))
RUTA_SALIDA_TFJS = Path(os.getenv("RUTA_SALIDA_TFJS", "modelo"))
DIR_TEMPORAL = Path("tfjs_target_dir")

if sys.platform == "win32":
    print(
        "[aviso] Estás en Windows nativo. tensorflowjs no se puede instalar aquí.\n"
        "         Usa convertir_a_tfjs.ipynb en Google Colab, o ejecutalo en WSL2."
    )

if not MODELO_H5.exists():
    sys.exit(f"[error] no existe {MODELO_H5}. Ejecuta primero el script de entrenamiento.")

CONVERSOR = "tensorflowjs_converter"
if shutil.which(CONVERSOR) is None:
    sys.exit(
        "[error] tensorflowjs_converter no esta en el PATH. Instala "
        "requirements-tfjs.txt en este interprete."
    )

if DIR_TEMPORAL.exists():
    shutil.rmtree(DIR_TEMPORAL)

subprocess.run(
    [CONVERSOR, "--input_format=keras", str(MODELO_H5), str(DIR_TEMPORAL)],
    check=True,
)

RUTA_SALIDA_TFJS.mkdir(parents=True, exist_ok=True)
for archivo in DIR_TEMPORAL.iterdir():
    destino = RUTA_SALIDA_TFJS / archivo.name
    shutil.copy2(archivo, destino)
    print(f"[tfjs] {destino}")

shutil.rmtree(DIR_TEMPORAL)
print(f"[tfjs] listo. Abre clasificador_fashion_mnist_dibujo.html en el navegador.")
