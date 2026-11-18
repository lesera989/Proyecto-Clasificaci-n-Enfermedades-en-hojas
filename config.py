
from pathlib import Path

# Ruta original de RoCoLe
BASE = Path(
    r"C:\Users\lesly\Downloads\RoCoLe A robusta coffee leaf images dataset\RoCoLe A robusta coffee leaf images dataset"
)

PHOTOS = BASE / "photos"
ANNOTATIONS = BASE / "annotations"

EXCEL = ANNOTATIONS / "RoCoLe-classes.xlsx"
COCO = ANNOTATIONS / "RoCoLE-coco.json"

# Carpetas del proyecto
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "resultados"

IMAGES = OUT / "imagenes"
MASKS = OUT / "mascaras"
PRED_MASKS = OUT / "mascaras_predichas"
SEGMENTED = OUT / "segmentadas"
MODELS = OUT / "modelos"
PLOTS = OUT / "graficas"

INVENTORY = OUT / "inventario.csv"

# Configuracion del entrenamiento
SIZE = 192
BATCH = 8
SEED = 42

EPOCHS_UNET = 20
EPOCHS_CLASSIFIER = 15

CLASSES = [
    "healthy",
    "rust_level_1",
    "rust_level_2",
    "rust_level_3",
    "rust_level_4",
    "red_spider_mite"
]

CLASS_NAMES = {
    "healthy": "Hoja sana",
    "rust_level_1": "Roya nivel 1",
    "rust_level_2": "Roya nivel 2",
    "rust_level_3": "Roya nivel 3",
    "rust_level_4": "Roya nivel 4",
    "red_spider_mite": "Acaro rojo"
}

def crear_carpetas():
    for carpeta in [
        OUT, IMAGES, MASKS,
        PRED_MASKS, SEGMENTED,
        MODELS, PLOTS
    ]:
        carpeta.mkdir(parents=True, exist_ok=True)
