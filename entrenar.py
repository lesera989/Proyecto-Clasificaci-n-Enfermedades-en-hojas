
import json
import gc

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    SEED,
    BATCH,
    CLASSES,
    MODELS,
    PLOTS,
    PRED_MASKS,
    SEGMENTED,
    INVENTORY,
    EPOCHS_UNET,
    EPOCHS_CLASSIFIER,
    crear_carpetas
)

from datos import (
    leer_inventario,
    dataset_segmentacion,
    dataset_clasificacion
)

from modelos import (
    crear_unet,
    crear_clasificador
)


# =====================================================
# CONFIGURACION GENERAL
# =====================================================

tf.keras.utils.set_random_seed(SEED)


def guardar_historial(historial, nombre):

    datos = pd.DataFrame(historial.history)

    datos.to_csv(
        PLOTS / f"historial_{nombre}.csv",
        index=False
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4)
    )

    axes[0].plot(
        datos["loss"],
        label="Entrenamiento"
    )

    axes[0].plot(
        datos["val_loss"],
        label="Validacion"
    )

    axes[0].set_title("Perdida")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()

    if "accuracy" in datos.columns:

        axes[1].plot(
            datos["accuracy"],
            label="Entrenamiento"
        )

        axes[1].plot(
            datos["val_accuracy"],
            label="Validacion"
        )

        axes[1].set_title("Exactitud")
        axes[1].set_xlabel("Epoca")
        axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS / f"historial_{nombre}.png",
        dpi=150
    )

    plt.close()


def crear_callbacks(ruta):

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ruta),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]


# =====================================================
# ETAPA 1: ENTRENAMIENTO DE U-NET
# =====================================================

def entrenar_unet():

    print("\n" + "=" * 55)
    print("ENTRENAMIENTO DE U-NET")
    print("=" * 55)

    crear_carpetas()

    tabla = leer_inventario()

    entrenamiento = tabla[
        tabla["particion"] == "entrenamiento"
    ]

    validacion = tabla[
        tabla["particion"] == "validacion"
    ]

    print(
        "Imagenes de entrenamiento:",
        len(entrenamiento)
    )

    print(
        "Imagenes de validacion:",
        len(validacion)
    )

    ds_train = dataset_segmentacion(
        entrenamiento,
        entrenamiento=True
    )

    ds_val = dataset_segmentacion(
        validacion,
        entrenamiento=False
    )

    modelo = crear_unet()

    modelo.summary()

    ruta = MODELS / "unet.keras"

    print("\nIniciando entrenamiento...")

    historial = modelo.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS_UNET,
        callbacks=crear_callbacks(ruta),
        verbose=1
    )

    guardar_historial(
        historial,
        "unet"
    )

    print("\nEntrenamiento finalizado")
    print("Modelo guardado en:", ruta)

    print(
        "Graficas guardadas en:",
        PLOTS
    )

    tf.keras.backend.clear_session()
    del modelo
    gc.collect()


# =====================================================
# ETAPA 2: GENERAR IMAGENES SEGMENTADAS
# =====================================================

def generar_segmentadas():

    print("\n" + "=" * 55)
    print("GENERACION DE IMAGENES SEGMENTADAS")
    print("=" * 55)

    crear_carpetas()

    ruta_modelo = MODELS / "unet.keras"

    if not ruta_modelo.exists():

        raise FileNotFoundError(
            "No existe unet.keras. "
            "Ejecuta primero: "
            "python main.py entrenar_unet"
        )

    tabla = leer_inventario()

    modelo = tf.keras.models.load_model(
        ruta_modelo,
        compile=False
    )

    ds = dataset_segmentacion(
        tabla,
        entrenamiento=False
    )

    contador = 0

    for imagenes, _ in ds:

        predicciones = modelo.predict_on_batch(
            imagenes
        )

        for prediccion in predicciones:

            fila = tabla.iloc[contador]

            imagen = np.asarray(
                Image.open(
                    fila["original"]
                ).convert("RGB")
            )

            mascara = (
                prediccion[:, :, 0] >= 0.5
            )

            mascara_png = (
                mascara.astype(np.uint8) * 255
            )

            segmentada = imagen.copy()

            segmentada[~mascara] = 0

            nombre = Path(
                fila["original"]
            ).name

            ruta_mascara = (
                PRED_MASKS / nombre
            )

            ruta_segmentada = (
                SEGMENTED / nombre
            )

            Image.fromarray(
                mascara_png
            ).save(ruta_mascara)

            Image.fromarray(
                segmentada
            ).save(ruta_segmentada)

            tabla.at[
                tabla.index[contador],
                "segmentada"
            ] = str(ruta_segmentada)

            contador += 1

            if contador % 100 == 0:

                print(
                    f"Segmentadas: "
                    f"{contador}/{len(tabla)}"
                )

    tabla.to_csv(
        INVENTORY,
        index=False
    )

    print("\nSegmentacion finalizada")
    print("Imagenes procesadas:", contador)
    print("Carpeta:", SEGMENTED)

    tf.keras.backend.clear_session()
    del modelo
    gc.collect()


# =====================================================
# PESOS PARA CLASES DESBALANCEADAS
# =====================================================

def calcular_pesos(tabla):

    cantidades = (
        tabla["clase"]
        .value_counts()
        .reindex(CLASSES)
    )

    total = len(tabla)
    numero_clases = len(CLASSES)

    pesos = {}

    for indice, clase in enumerate(CLASSES):

        cantidad = int(cantidades[clase])

        pesos[indice] = (
            total / (numero_clases * cantidad)
        )

    print("\nPesos por clase:")

    for indice, peso in pesos.items():

        print(
            CLASSES[indice],
            ":",
            round(peso, 3)
        )

    return pesos


# =====================================================
# ETAPA 3: ENTRENAR CLASIFICADORES
# =====================================================

def entrenar_clasificadores():

    print("\n" + "=" * 55)
    print("ENTRENAMIENTO DE CLASIFICADORES")
    print("=" * 55)

    crear_carpetas()

    tabla = leer_inventario()

    entrenamiento = tabla[
        tabla["particion"] == "entrenamiento"
    ]

    validacion = tabla[
        tabla["particion"] == "validacion"
    ]

    pesos = calcular_pesos(
        entrenamiento
    )

    arquitecturas = [
        "cnn",
        "mobilenetv2",
        "efficientnetb0"
    ]

    condiciones = [
        ("original", "original"),
        ("segmentada", "segmentada")
    ]

    resultados = []

    for arquitectura in arquitecturas:

        for tipo, columna in condiciones:

            print("\n" + "=" * 55)

            print(
                "Modelo:",
                arquitectura
            )

            print(
                "Tipo de imagen:",
                tipo
            )

            print("=" * 55)

            if columna == "segmentada":

                rutas = tabla[
                    columna
                ].astype(str)

                if (
                    (rutas == "").any()
                    or not all(
                        Path(ruta).exists()
                        for ruta in rutas
                    )
                ):

                    raise FileNotFoundError(
                        "Faltan imagenes segmentadas. "
                        "Ejecuta primero: "
                        "python main.py generar_segmentadas"
                    )

            ds_train = dataset_clasificacion(
                entrenamiento,
                columna=columna,
                entrenamiento=True
            )

            ds_val = dataset_clasificacion(
                validacion,
                columna=columna,
                entrenamiento=False
            )

            tf.keras.backend.clear_session()

            modelo = crear_clasificador(
                arquitectura
            )

            nombre = (
                arquitectura + "_" + tipo
            )

            ruta = (
                MODELS / f"{nombre}.keras"
            )

            historial = modelo.fit(
                ds_train,
                validation_data=ds_val,
                epochs=EPOCHS_CLASSIFIER,
                class_weight=pesos,
                callbacks=crear_callbacks(ruta),
                verbose=1
            )

            guardar_historial(
                historial,
                nombre
            )

            # Evaluar el modelo realmente guardado,
            # no solo las metricas del historial.

            mejor = tf.keras.models.load_model(
                ruta,
                compile=False
            )

            mejor.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
            )

            metricas = mejor.evaluate(
                ds_val,
                verbose=0,
                return_dict=True
            )

            exactitud = float(
                metricas["accuracy"]
            )

            perdida = float(
                metricas["loss"]
            )

            print(
                "\nAccuracy validacion:",
                round(exactitud, 4)
            )

            resultados.append({
                "modelo": arquitectura,
                "tipo": tipo,
                "val_accuracy": exactitud,
                "val_loss": perdida,
                "ruta_modelo": str(ruta)
            })

            pd.DataFrame(
                resultados
            ).to_csv(
                PLOTS / "comparacion_validacion.csv",
                index=False
            )

            del modelo
            del mejor
            gc.collect()

    comparacion = pd.DataFrame(
        resultados
    )

    comparacion = comparacion.sort_values(
        by=["val_accuracy", "val_loss"],
        ascending=[False, True]
    )

    mejor = comparacion.iloc[0]

    informacion = {
        "modelo": str(mejor["modelo"]),
        "tipo": str(mejor["tipo"]),
        "ruta_modelo": str(
            mejor["ruta_modelo"]
        ),
        "val_accuracy": float(
            mejor["val_accuracy"]
        ),
        "val_loss": float(
            mejor["val_loss"]
        )
    }

    with open(
        MODELS / "mejor_modelo.json",
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            informacion,
            archivo,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 55)
    print("COMPARACION FINAL")
    print("=" * 55)

    print(
        comparacion.to_string(
            index=False
        )
    )

    print("\nMejor modelo:")
    print(informacion)
