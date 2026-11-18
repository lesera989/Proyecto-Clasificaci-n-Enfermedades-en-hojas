# Proyecto de clasificación de enfermedades en hojas de café

Sistema académico de visión computacional para detectar enfermedades y estados de hojas de café robusta a partir de fotografías. El proyecto utiliza segmentación con U-Net para aislar la hoja y clasificadores de aprendizaje profundo para asignarla a una de seis categorías.

> Nota: las predicciones son estimaciones de un prototipo académico y no sustituyen una inspección fitosanitaria realizada por un profesional.

## Contenido

- [Propósito](#propósito)
- [Arquitectura general](#arquitectura-general)
- [Tecnologías](#tecnologías)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Dataset y preparación](#dataset-y-preparación)
- [Flujo de trabajo](#flujo-de-trabajo)
- [Clasificación disponible](#clasificación-disponible)
- [Modelos y métricas](#modelos-y-métricas)
- [Instalación](#instalación)
- [Uso](#uso)
- [Despliegue en internet](#despliegue-en-internet)
- [Archivos generados](#archivos-generados)
- [Limitaciones y consideraciones](#limitaciones-y-consideraciones)

## Propósito

El sistema procesa una fotografía de una hoja de café mediante dos etapas:

1. **Segmentación**: U-Net genera una máscara binaria y conserva únicamente los píxeles que corresponden a la hoja.
2. **Clasificación**: el mejor modelo clasificatorio compara la imagen original o segmentada y devuelve la clase más probable junto con las probabilidades de las seis categorías.

El proyecto utiliza el dataset [RoCoLe](https://github.com/carlos-jarillo/rocole): *A robusta coffee leaf images dataset*.

## Arquitectura general

```text
Fotografía
    |
    v
Redimensionado a 192 x 192
    |
    v
U-Net ------------------------> Máscara de la hoja
    |                                  |
    v                                  v
Imagen original o imagen con fondo negro
    |
    v
CNN / MobileNetV2 / EfficientNetB0
    |
    v
Probabilidades de las 6 clases
```

El programa separa los datos por grupos identificados como `CNP` para evitar que fotografías del mismo grupo terminen simultáneamente en entrenamiento y evaluación. La partición aproximada es 80 % entrenamiento, 10 % validación y 10 % prueba.

## Tecnologías

- **Python 3.11**: entorno de ejecución del proyecto.
- **TensorFlow/Keras**: construcción, entrenamiento y carga de U-Net y clasificadores.
- **MobileNetV2** y **EfficientNetB0**: transferencia de aprendizaje desde pesos entrenados con ImageNet.
- **CNN propia**: arquitectura convolucional desarrollada dentro del proyecto.
- **Pandas** y **OpenPyXL**: lectura del inventario XLSX y generación de CSV.
- **Scikit-learn**: división agrupada de datos y métricas de evaluación.
- **Pillow**: lectura EXIF, redimensionado, máscaras y composiciones de imágenes.
- **NumPy**: manipulación de tensores y máscaras.
- **Matplotlib**: gráficos de historial, matrices de confusión y comparaciones.
- **Gradio**: interfaz web interactiva para cargar una fotografía y mostrar resultados.

## Estructura del proyecto

```text
Proyecto_Cafe/
├── app.py                 Interfaz web, carga de modelos e inferencia
├── config.py              Rutas, clases, tamaño de imagen y configuración
├── datos.py               Preparación del dataset, máscaras y datasets de TF
├── entrenar.py            Entrenamiento de U-Net y clasificadores
├── evaluar.py             Evaluación final y generación de métricas/gráficas
├── modelos.py             Arquitecturas Keras
├── main.py                Punto de entrada con comandos de línea
├── README.md              Documentación del proyecto
├── requirements.txt       Dependencias sugeridas
├── .gitignore             Exclusión de artefactos locales y generados
├── outputs/               Artefactos de compatibilidad
└── resultados/
    ├── imagenes/          Fotografías procesadas
    ├── mascaras/          Máscaras reales extraídas de COCO
    ├── mascaras_predichas/ Máscaras generadas por U-Net
    ├── segmentadas/       Fotografías con fondo eliminado
    ├── modelos/           Modelos .keras y mejor_modelo.json
    ├── graficas/          Historiales, matrices y CSV de métricas
    └── inventario.csv     Rutas, clases y particiones del dataset
```

### Responsabilidad de cada módulo

- **`main.py`** expone las acciones `preparar`, `entrenar_unet`, `generar_segmentadas`, `entrenar_clasificadores`, `evaluar` e `interfaz`.
- **`config.py`** define una entrada de `192 x 192` píxeles, el lote de 8 imágenes, la semilla `42`, las seis clases y las rutas del proyecto.
- **`datos.py`** valida el XLSX y COCO, construye máscaras poligonales, corrige orientación EXIF, crea particiones sin mezclar grupos y genera los datasets de TensorFlow.
- **`modelos.py`** define U-Net, una CNN propia y clasificadores con MobileNetV2/EfficientNetB0.
- **`entrenar.py`** aplica checkpoints, early stopping, reducción de la tasa de aprendizaje, ponderación de clases y selección del mejor modelo por exactitud de validación.
- **`evaluar.py`** calcula IoU y Dice para U-Net, además de accuracy, balanced accuracy, precision, recall y F1 para los clasificadores.
- **`app.py`** carga U-Net y el clasificador indicado por `mejor_modelo.json`, segmenta la fotografía y muestra el diagnóstico y las probabilidades.

## Dataset y preparación

El proyecto espera esta información local de RoCoLe:

- Fotografías dentro de `photos/`.
- Inventario de clases en `RoCoLe-classes.xlsx`.
- Anotaciones COCO en `RoCoLE-coco.json`.

La ruta base está configurada actualmente en `config.py` como una ruta absoluta de Windows:

Para usarlo en otro equipo conviene reemplazarla por una ruta relativa o una variable de entorno. Por ejemplo:

```python
BASE = Path(__file__).resolve().parent.parent / "data" / "rocole"
```

La preparación valida que existan todas las clases, que no haya nombres duplicados, que todas las fotografías tengan anotación y que las máscaras tengan contenido válido. También guarda los errores en `resultados/errores.csv` y las advertencias de máscaras en `resultados/advertencias_mascaras.csv` si se producen.

## Flujo de trabajo

Ejecutar las acciones en este orden:

```powershell
python main.py preparar
python main.py entrenar_unet
python main.py generar_segmentadas
python main.py entrenar_clasificadores
python main.py evaluar
python main.py interfaz
```

La etapa `preparar` puede tardar varios minutos porque procesa todas las fotografías y genera máscaras. `generar_segmentadas` requiere haber entrenado U-Net. `entrenar_clasificadores` ejecuta seis combinaciones: tres arquitecturas con imágenes originales y segmentadas.

### 1. Preparación

```powershell
python main.py preparar
```

Genera el inventario y las imágenes normalizadas de la partición.

### 2. Entrenamiento de U-Net

```powershell
python main.py entrenar_unet
```

Entrena durante 20 épocas como máximo, con early stopping de paciencia 5 y checkpoint del mejor `val_loss`.

### 3. Segmentación

```powershell
python main.py generar_segmentadas
```

Aplica U-Net sobre el conjunto completo y añade las rutas de imágenes segmentadas al inventario.

### 4. Clasificadores

```powershell
python main.py entrenar_clasificadores
```

Entrena las variantes originales y segmentadas de:

- CNN propia
- MobileNetV2
- EfficientNetB0

Después guarda el modelo con mejor accuracy de validación en `mejor_modelo.json`.

### 5. Evaluación independiente

```powershell
python main.py evaluar
```

Evalúa U-Net y los seis clasificadores sobre el conjunto de prueba. Genera reportes CSV, matrices de confusión e histogramas comparativos.

### 6. Interfaz

```powershell
python main.py interfaz
```

Carga automáticamente U-Net y el clasificador seleccionado por `mejor_modelo.json`, abre el navegador y empieza un servidor local de Gradio. Por defecto el proyecto ejecuta `interfaz.launch(inbrowser=True, share=False)`.

## Clasificación disponible

| Identificador | Nombre mostrado |
|---|---|
| `healthy` | Hoja sana |
| `rust_level_1` | Roya nivel 1 |
| `rust_level_2` | Roya nivel 2 |
| `rust_level_3` | Roya nivel 3 |
| `rust_level_4` | Roya nivel 4 |
| `red_spider_mite` | Ácaro rojo |

## Modelos y métricas

Los artefactos actuales se encuentran en `resultados/modelos/`:

- `unet.keras`
- `cnn_original.keras`
- `cnn_segmentada.keras`
- `mobilenetv2_original.keras`
- `mobilenetv2_segmentada.keras`
- `efficientnetb0_original.keras`
- `efficientnetb0_segmentada.keras`
- `mejor_modelo.json`

Según el archivo actual, el mejor modelo seleccionado es **MobileNetV2 entrenado con imágenes segmentadas**, con accuracy de validación registrada de `0.5769`. La selección se realiza automáticamente durante el entrenamiento y puede cambiar si se vuelve a entrenar el proyecto.

La segmentación se mide con:

- IoU promedio e IoU global.
- Dice promedio y Dice global.

Los clasificadores se evalúan con:

- Accuracy.
- Balanced accuracy.
- Precision macro.
- Recall macro.
- F1 macro y F1 ponderado.
- Reportes por clase y matrices de confusión.

Los resultados se almacenan en `resultados/graficas/`.

## Instalación

### 1. Requisitos previos

- Windows 10/11, Linux o WSL.
- Python 3.10 o 3.11 recomendado.
- 8 GB de RAM como mínimo; 16 GB recomendado para entrenamiento.
- Una GPU compatible con TensorFlow acelera el entrenamiento, pero no es obligatoria.
- Descargar y extraer el dataset RoCoLe únicamente si se desea repetir la preparación o el entrenamiento.

### 2. Crear un entorno aislado

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En Linux o WSL:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Preparar las rutas

Esta ruta solo es necesaria para preparar o volver a entrenar el proyecto con el dataset original RoCoLe. No se necesita el dataset para usar los modelos ya entrenados.

El repositorio puede funcionar en otra máquina sin cambiar las rutas de los modelos, porque `config.py` construye `ROOT` y `MODELS` a partir de la ubicación real del proyecto.

Para repetir el entrenamiento en cualquier equipo, reemplaza la ruta absoluta de `BASE` por una ruta relativa o una variable de entorno:

```python
import os
from pathlib import Path

BASE = Path(os.environ.get(
    "ROCOLE_DIR",
    Path(__file__).resolve().parent / "data" / "rocole"
))
```

Coloca el dataset dentro del proyecto, si usas la ruta relativa:

```text
data/rocole/
├── photos/
└── annotations/
    ├── RoCoLe-classes.xlsx
    └── RoCoLE-coco.json
```

También puedes indicar una ruta externa desde una terminal:

```powershell
$env:ROCOLE_DIR="C:\Ruta\Al\RoCoLe"
python main.py preparar
```

En Linux o WSL:

```bash
export ROCOLE_DIR="/ruta/al/rocole"
python main.py preparar
```

## Uso

### Ejecutar en otra máquina sin volver a entrenar

Cuando clones el repositorio, confirma que existen los modelos ya entrenados:

```text
resultados/modelos/unet.keras
resultados/modelos/mejor_modelo.json
resultados/modelos/<clasificador>.keras
```

El clasificador exacto debe coincidir con `mejor_modelo.json`. En la configuración actual corresponde a `mobilenetv2_segmentada.keras`.

Instala las dependencias una sola vez y ejecuta solamente la interfaz:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py interfaz
```

No necesitas descargar el dataset RoCoLe, preparar imágenes, generar máscaras, entrenar U-Net o volver a entrenar los clasificadores.

### Uso general

Ejecuta la interfaz desde la raíz del proyecto:

```powershell
python main.py interfaz
```

Carga una fotografía clara de una hoja de café y pulsa **Analizar imagen**. La interfaz devuelve:

- Hoja segmentada.
- Máscara binaria creada por U-Net.
- Clase más probable.
- Probabilidad estimada.
- Modelo y tipo de imagen utilizados.
- Probabilidad de cada categoría.

## Archivos generados

El entrenamiento puede generar cientos de imágenes y varios modelos pesados. Por ello deben gestionarse con cuidado y normalmente deben ignorarse en Git:

- `resultados/`.
- `outputs/`.
- Cachés como `__pycache__/`.
- Entornos virtuales como `.venv/`.
- Pesos `.keras`, `.pth`, `.pt`, `.h5` y otros formatos binarios.
- Imágenes y máscaras generadas.

Los modelos utilizados en la interfaz pueden mantenerse fuera de Git temporalmente y subirse directamente al servicio de despliegue si son demasiado grandes.

## Limitaciones y consideraciones

- `config.py` contiene actualmente una ruta absoluta de Windows para el dataset original.
- El entrenamiento depende de una descarga local de RoCoLe y de los archivos de anotación.
- MobileNetV2 e EfficientNetB0 pueden intentar descargar pesos de ImageNet; si no hay conexión, los modelos se inicializan sin esos pesos.
- TensorFlow puede consumir mucha memoria al cargar U-Net y el clasificador simultáneamente.
- El conjunto de datos está desbalanceado, por lo que el proyecto usa pesos por clase durante el entrenamiento.
- Las probabilidades son resultados del modelo y no son un diagnóstico fitosanitario definitivo.
- Las fotografías deben seguir aproximadamente el formato y las condiciones visuales de RoCoLe para obtener resultados representativos.