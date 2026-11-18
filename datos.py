
import json
import re
from urllib.parse import unquote

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from sklearn.model_selection import GroupShuffleSplit

from config import (
    PHOTOS, EXCEL, COCO, SIZE, SEED, BATCH,
    CLASSES, IMAGES, MASKS,
    INVENTORY, PLOTS, crear_carpetas
)

# Transformaciones equivalentes a la orientacion EXIF
TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90
}


def ajustar(imagen, es_mascara=False):
    """Redimensiona sin deformar, añadiendo bordes negros."""

    w, h = imagen.size
    escala = min(SIZE / w, SIZE / h)

    nw = max(1, round(w * escala))
    nh = max(1, round(h * escala))

    metodo = (
        Image.Resampling.NEAREST
        if es_mascara else Image.Resampling.BILINEAR
    )

    imagen = imagen.resize((nw, nh), metodo)

    modo = "L" if es_mascara else "RGB"
    salida = Image.new(modo, (SIZE, SIZE), 0)

    x = (SIZE - nw) // 2
    y = (SIZE - nh) // 2

    salida.paste(imagen, (x, y))
    return salida


def obtener_nombre_coco(url, nombres):
    """Extrae el nombre original de la URL de Labelbox."""

    nombre_url = unquote(str(url)).split("/")[-1].lower()

    coincidencias = [
        nombre for nombre in nombres
        if nombre_url == nombre.lower()
        or nombre_url.endswith("-" + nombre.lower())
    ]

    return coincidencias[0] if len(coincidencias) == 1 else None


def mascara_desde_coco(imagen, meta, anotaciones):
    """
    Genera una mascara usando el sistema de coordenadas
    COCO y aplica la orientacion EXIF cuando corresponde.
    """

    w_coco = int(meta["width"])
    h_coco = int(meta["height"])

    orientacion = int(imagen.getexif().get(274, 1))

    original = ImageOps.exif_transpose(imagen).convert("RGB")

    w_raw, h_raw = imagen.size
    w_final, h_final = original.size

    ratio_coco = w_coco / h_coco
    ratio_raw = w_raw / h_raw
    ratio_final = w_final / h_final

    # Determinar el sistema de coordenadas de COCO
    if abs(ratio_coco - ratio_raw) < 0.02:
        ancho, alto = w_raw, h_raw
        aplicar_exif = True

    elif abs(ratio_coco - ratio_final) < 0.02:
        ancho, alto = w_final, h_final
        aplicar_exif = False

    else:
        raise ValueError(
            f"Dimensiones incompatibles: "
            f"COCO={w_coco}x{h_coco}, "
            f"foto={w_raw}x{h_raw}"
        )

    mascara = Image.new("L", (ancho, alto), 0)
    dibujo = ImageDraw.Draw(mascara)

    sx = ancho / w_coco
    sy = alto / h_coco

    poligonos_validos = 0

    for ann in anotaciones:

        segmentos = ann.get("segmentation", [])

        if not isinstance(segmentos, list):
            continue

        for poligono in segmentos:

            if len(poligono) < 6 or len(poligono) % 2:
                continue

            puntos = [
                (
                    float(poligono[i]) * sx,
                    float(poligono[i + 1]) * sy
                )
                for i in range(0, len(poligono), 2)
            ]

            dibujo.polygon(puntos, fill=255)
            poligonos_validos += 1

    if poligonos_validos == 0:
        raise ValueError("Sin poligonos validos")

    if aplicar_exif and orientacion in TRANSPOSE:
        mascara = mascara.transpose(TRANSPOSE[orientacion])

    if mascara.size != original.size:
        raise ValueError("Mascara y fotografia no coinciden")

    return original, mascara


def dividir_grupos(datos):
    """
    Busca una division aproximada 80/10/10,
    sin compartir grupos entre particiones.
    """

    mejor = None
    mejor_puntaje = float("inf")

    distribucion = (
        datos["clase"].value_counts(normalize=True)
        .reindex(CLASSES, fill_value=0)
    )

    for intento in range(200):

        primero = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=SEED + intento
        )

        idx_train, idx_temp = next(
            primero.split(
                datos,
                groups=datos["grupo"]
            )
        )

        temporal = datos.iloc[idx_temp]

        segundo = GroupShuffleSplit(
            n_splits=1,
            test_size=0.50,
            random_state=SEED + 1000 + intento
        )

        idx_val_local, idx_test_local = next(
            segundo.split(
                temporal,
                groups=temporal["grupo"]
            )
        )

        idx_val = idx_temp[idx_val_local]
        idx_test = idx_temp[idx_test_local]

        particiones = {
            "entrenamiento": idx_train,
            "validacion": idx_val,
            "prueba": idx_test
        }

        puntaje = 0
        valido = True

        for indices in particiones.values():

            parte = datos.iloc[indices]

            if set(parte["clase"]) != set(CLASSES):
                valido = False
                break

            proporcion = (
                parte["clase"].value_counts(normalize=True)
                .reindex(CLASSES, fill_value=0)
            )

            puntaje += (
                proporcion - distribucion
            ).abs().sum()

        if valido and puntaje < mejor_puntaje:
            mejor = particiones
            mejor_puntaje = puntaje

    if mejor is None:
        raise RuntimeError(
            "No se encontro una division con todas las clases."
        )

    datos = datos.copy()
    datos["particion"] = ""

    for nombre, indices in mejor.items():
        datos.loc[indices, "particion"] = nombre

    return datos


def preparar():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    crear_carpetas()

    for ruta in [PHOTOS, EXCEL, COCO]:
        if not ruta.exists():
            raise FileNotFoundError(f"No existe: {ruta}")

    tabla = pd.read_excel(EXCEL)

    print("Registros encontrados:", len(tabla))
    print(tabla["Multiclass.Label"].value_counts())

    if tabla["File"].duplicated().any():
        raise ValueError("Hay nombres de imagen duplicados")

    if not set(tabla["Multiclass.Label"]) == set(CLASSES):
        raise ValueError("Las clases no coinciden con config.py")

    with open(COCO, "r", encoding="utf-8") as f:
        coco = json.load(f)

    fotos = {
        p.name.lower(): p
        for p in PHOTOS.rglob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    }

    nombres = tabla["File"].astype(str).tolist()

    mapa_coco = {}

    for meta in coco["images"]:
        nombre = obtener_nombre_coco(
            meta["file_name"], nombres
        )
        if nombre:
            mapa_coco[nombre] = meta

    anotaciones = {}

    for ann in coco["annotations"]:
        if ann.get("category_id") != 1:
            continue

        identificador = str(ann["image_id"])
        anotaciones.setdefault(
            identificador, []
        ).append(ann)

    registros = []
    errores = []
    advertencias = []

    for numero, (_, fila) in enumerate(tabla.iterrows(), 1):

        nombre = str(fila["File"])
        clase = str(fila["Multiclass.Label"])

        foto = fotos.get(nombre.lower())
        meta = mapa_coco.get(nombre)

        if foto is None or meta is None:
            errores.append((nombre, "Foto o anotacion faltante"))
            continue

        try:
            with Image.open(foto) as imagen:
                original, mascara = mascara_desde_coco(
                    imagen,
                    meta,
                    anotaciones.get(str(meta["id"]), [])
                )

            original = ajustar(original)
            mascara = ajustar(mascara, es_mascara=True)

            proporcion = (
                np.asarray(mascara) > 127
            ).mean()

            if proporcion == 0:
                raise ValueError("Mascara vacia")

            if proporcion < 0.005 or proporcion > 0.95:
                advertencias.append((nombre, proporcion))

            salida = foto.stem + ".png"

            ruta_original = IMAGES / salida
            ruta_mascara = MASKS / salida

            original.save(ruta_original)
            mascara.save(ruta_mascara)

            coincidencia = re.match(
                r"^(C\d+P\d+)", nombre, re.I
            )

            if not coincidencia:
                raise ValueError("No se pudo identificar el grupo")

            registros.append({
                "archivo": nombre,
                "clase": clase,
                "grupo": coincidencia.group(1).upper(),
                "original": str(ruta_original),
                "mascara": str(ruta_mascara),
                "segmentada": ""
            })

        except Exception as e:
            errores.append((nombre, str(e)))

        if numero % 200 == 0:
            print(f"Procesadas: {numero}/{len(tabla)}")

    if errores:
        pd.DataFrame(
            errores, columns=["archivo", "error"]
        ).to_csv(
            INVENTORY.parent / "errores.csv",
            index=False
        )

        raise RuntimeError(
            f"Se encontraron {len(errores)} errores. "
            "Revisa resultados/errores.csv"
        )

    datos = pd.DataFrame(registros)

    if len(datos) != len(tabla):
        raise RuntimeError("No se prepararon todas las imagenes")

    datos = dividir_grupos(datos)
    datos.to_csv(INVENTORY, index=False)

    if advertencias:
        pd.DataFrame(
            advertencias,
            columns=["archivo", "proporcion_mascara"]
        ).to_csv(
            INVENTORY.parent / "advertencias_mascaras.csv",
            index=False
        )

    # Visualizacion de muestras de todas las clases
    muestras = pd.concat([
        datos[datos["clase"] == clase].sample(
            n=min(2, sum(datos["clase"] == clase)),
            random_state=SEED
        )
        for clase in CLASSES
    ], ignore_index=True)

    fig, axes = plt.subplots(
        len(muestras), 3,
        figsize=(10, 3 * len(muestras))
    )

    for i, fila in muestras.iterrows():

        imagen = np.asarray(
            Image.open(fila["original"]).convert("RGB")
        )

        mascara = np.asarray(
            Image.open(fila["mascara"]).convert("L")
        ) > 127

        segmentada = imagen.copy()
        segmentada[~mascara] = 0

        axes[i, 0].imshow(imagen)
        axes[i, 1].imshow(mascara, cmap="gray")
        axes[i, 2].imshow(segmentada)

        axes[i, 0].set_ylabel(
            fila["clase"], fontsize=8
        )

        for ax in axes[i]:
            ax.axis("off")

    axes[0, 0].set_title("Original")
    axes[0, 1].set_title("Mascara")
    axes[0, 2].set_title("Segmentada")

    plt.tight_layout()
    plt.savefig(
        PLOTS / "verificar_mascaras.png",
        dpi=130
    )
    plt.close()

    print("\nPreparacion finalizada")
    print(pd.crosstab(
        datos["clase"], datos["particion"]
    ))
    print("\nGrupos:", datos["grupo"].nunique())
    print("Advertencias de mascaras:", len(advertencias))
    print("Revisa resultados/graficas/verificar_mascaras.png")


# =====================================================
# DATASETS PARA TENSORFLOW
# =====================================================

def leer_inventario():
    if not INVENTORY.exists():
        raise FileNotFoundError(
            "Ejecuta primero: python main.py preparar"
        )
    return pd.read_csv(INVENTORY).fillna("")


def dataset_segmentacion(tabla, entrenamiento=False):
    import tensorflow as tf

    rutas = tabla["original"].astype(str).tolist()
    mascaras = tabla["mascara"].astype(str).tolist()

    ds = tf.data.Dataset.from_tensor_slices(
        (rutas, mascaras)
    )

    if entrenamiento:
        ds = ds.shuffle(len(tabla), seed=SEED)

    def cargar(ruta, ruta_mascara):
        imagen = tf.io.decode_png(
            tf.io.read_file(ruta), channels=3
        )
        mascara = tf.io.decode_png(
            tf.io.read_file(ruta_mascara), channels=1
        )

        imagen = tf.cast(imagen, tf.float32) / 255.0
        mascara = tf.cast(mascara > 127, tf.float32)

        return imagen, mascara

    return ds.map(
        cargar,
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH).prefetch(tf.data.AUTOTUNE)


def dataset_clasificacion(
    tabla, columna="original", entrenamiento=False
):
    import tensorflow as tf

    rutas = tabla[columna].astype(str).tolist()
    etiquetas = [
        CLASSES.index(clase) for clase in tabla["clase"]
    ]

    ds = tf.data.Dataset.from_tensor_slices(
        (rutas, etiquetas)
    )

    if entrenamiento:
        ds = ds.shuffle(len(tabla), seed=SEED)

    def cargar(ruta, etiqueta):
        imagen = tf.io.decode_png(
            tf.io.read_file(ruta), channels=3
        )
        return tf.cast(imagen, tf.float32), etiqueta

    return ds.map(
        cargar,
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(BATCH).prefetch(tf.data.AUTOTUNE)
