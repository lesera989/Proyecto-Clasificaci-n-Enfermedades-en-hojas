
import gc
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from config import (
    CLASSES,
    CLASS_NAMES,
    MODELS,
    PLOTS,
    crear_carpetas
)

from datos import (
    leer_inventario,
    dataset_segmentacion,
    dataset_clasificacion
)


# =====================================================
# EVALUACION DE U-NET
# =====================================================

def evaluar_unet(tabla_prueba):

    print("\n" + "=" * 55)
    print("EVALUACION DE U-NET")
    print("=" * 55)

    ruta = MODELS / "unet.keras"

    if not ruta.exists():
        print("No se encontro el modelo U-Net.")
        return

    modelo = tf.keras.models.load_model(
        ruta,
        compile=False
    )

    ds = dataset_segmentacion(
        tabla_prueba,
        entrenamiento=False
    )

    interseccion_total = 0
    union_total = 0
    suma_total = 0

    iou_imagenes = []
    dice_imagenes = []

    for imagenes, mascaras in ds:

        predicciones = modelo.predict_on_batch(
            imagenes
        )

        predicciones = (
            predicciones >= 0.5
        )

        reales = (
            mascaras.numpy() >= 0.5
        )

        for pred, real in zip(
            predicciones, reales
        ):

            pred = pred.astype(bool)
            real = real.astype(bool)

            interseccion = np.logical_and(
                pred, real
            ).sum()

            union = np.logical_or(
                pred, real
            ).sum()

            suma = (
                pred.sum() + real.sum()
            )

            iou = (
                interseccion / union
                if union > 0 else 1.0
            )

            dice = (
                2 * interseccion / suma
                if suma > 0 else 1.0
            )

            iou_imagenes.append(iou)
            dice_imagenes.append(dice)

            interseccion_total += interseccion
            union_total += union
            suma_total += suma

    iou_global = (
        interseccion_total / union_total
        if union_total > 0 else 1.0
    )

    dice_global = (
        2 * interseccion_total / suma_total
        if suma_total > 0 else 1.0
    )

    resultados = {
        "iou_promedio": float(
            np.mean(iou_imagenes)
        ),
        "dice_promedio": float(
            np.mean(dice_imagenes)
        ),
        "iou_global": float(iou_global),
        "dice_global": float(dice_global)
    }

    pd.DataFrame(
        [resultados]
    ).to_csv(
        PLOTS / "metricas_unet_prueba.csv",
        index=False
    )

    print("\nResultados U-Net en prueba:")

    for nombre, valor in resultados.items():
        print(f"{nombre}: {valor:.4f}")

    del modelo
    tf.keras.backend.clear_session()
    gc.collect()


# =====================================================
# EVALUACION DE CLASIFICADORES
# =====================================================

def evaluar_clasificador(
    arquitectura,
    tipo,
    tabla_prueba
):

    nombre = f"{arquitectura}_{tipo}"

    print("\n" + "=" * 55)
    print("EVALUANDO:", nombre)
    print("=" * 55)

    ruta = MODELS / f"{nombre}.keras"

    if not ruta.exists():

        print("No se encontro:", ruta)
        return None

    columna = (
        "original"
        if tipo == "original"
        else "segmentada"
    )

    ds = dataset_clasificacion(
        tabla_prueba,
        columna=columna,
        entrenamiento=False
    )

    modelo = tf.keras.models.load_model(
        ruta,
        compile=False
    )

    probabilidades = modelo.predict(
        ds,
        verbose=1
    )

    y_pred = np.argmax(
        probabilidades,
        axis=1
    )

    y_true = np.array([
        CLASSES.index(clase)
        for clase in tabla_prueba["clase"]
    ])

    etiquetas = list(
        range(len(CLASSES))
    )

    nombres = [
        CLASS_NAMES[clase]
        for clase in CLASSES
    ]

    accuracy = accuracy_score(
        y_true, y_pred
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true, y_pred
        )
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=etiquetas,
            average="macro",
            zero_division=0
        )
    )

    _, _, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=etiquetas,
            average="weighted",
            zero_division=0
        )
    )

    print("\nRESULTADOS:")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"Precision macro: {precision:.4f}"
    )

    print(
        f"Recall macro: {recall:.4f}"
    )

    print(
        f"F1-score macro: {f1:.4f}"
    )

    print("\nREPORTE POR CLASE:")

    reporte = classification_report(
        y_true,
        y_pred,
        labels=etiquetas,
        target_names=nombres,
        output_dict=True,
        zero_division=0
    )

    reporte_df = pd.DataFrame(
        reporte
    ).transpose()

    print(
        reporte_df.to_string()
    )

    reporte_df.to_csv(
        PLOTS / f"reporte_{nombre}.csv"
    )

    # Matriz de confusion

    matriz = confusion_matrix(
        y_true,
        y_pred,
        labels=etiquetas
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    visualizacion = (
        ConfusionMatrixDisplay(
            confusion_matrix=matriz,
            display_labels=nombres
        )
    )

    visualizacion.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
        colorbar=False
    )

    plt.title(
        f"Matriz de confusion - {nombre}"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS / f"matriz_confusion_{nombre}.png",
        dpi=180
    )

    plt.close()

    resultado = {
        "modelo": arquitectura,
        "tipo": tipo,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "f1_weighted": float(f1_weighted)
    }

    del modelo
    tf.keras.backend.clear_session()
    gc.collect()

    return resultado


# =====================================================
# COMPARACION FINAL
# =====================================================

def evaluar():

    crear_carpetas()

    print("\n" + "=" * 55)
    print("EVALUACION FINAL DEL PROYECTO")
    print("=" * 55)

    tabla = leer_inventario()

    prueba = tabla[
        tabla["particion"] == "prueba"
    ].copy()

    print(
        "\nImagenes de prueba:",
        len(prueba)
    )

    print("\nDistribucion de clases:")

    print(
        prueba["clase"].value_counts()
    )

    # Evaluar segmentacion

    evaluar_unet(prueba)

    # Evaluar los seis clasificadores

    arquitecturas = [
        "cnn",
        "mobilenetv2",
        "efficientnetb0"
    ]

    tipos = [
        "original",
        "segmentada"
    ]

    resultados = []

    for arquitectura in arquitecturas:

        for tipo in tipos:

            resultado = evaluar_clasificador(
                arquitectura,
                tipo,
                prueba
            )

            if resultado is not None:

                resultados.append(
                    resultado
                )

                pd.DataFrame(
                    resultados
                ).to_csv(
                    PLOTS / "comparacion_prueba.csv",
                    index=False
                )

    if not resultados:

        raise RuntimeError(
            "No se pudo evaluar ningun modelo."
        )

    comparacion = pd.DataFrame(
        resultados
    )

    comparacion = comparacion.sort_values(
        by="f1_macro",
        ascending=False
    )

    print("\n" + "=" * 55)
    print("COMPARACION FINAL - PRUEBA")
    print("=" * 55)

    print(
        comparacion.to_string(
            index=False
        )
    )

    # Grafica comparativa

    etiquetas = [
        f"{fila.modelo}\n{fila.tipo}"
        for fila in comparacion.itertuples()
    ]

    x = np.arange(
        len(comparacion)
    )

    ancho = 0.35

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    ax.bar(
        x - ancho / 2,
        comparacion["accuracy"],
        ancho,
        label="Accuracy"
    )

    ax.bar(
        x + ancho / 2,
        comparacion["f1_macro"],
        ancho,
        label="F1 macro"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        etiquetas,
        fontsize=9
    )

    ax.set_ylim(0, 1)

    ax.set_ylabel("Puntuacion")

    ax.set_title(
        "Comparacion de clasificadores - Prueba"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS / "comparacion_prueba.png",
        dpi=180
    )

    plt.close()

    print("\nEvaluacion finalizada.")

    print(
        "Resultados guardados en:",
        PLOTS
    )


if __name__ == "__main__":
    evaluar()
