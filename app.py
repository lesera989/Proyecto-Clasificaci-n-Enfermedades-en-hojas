
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import gradio as gr

from PIL import Image, ImageOps

from config import (
    SIZE,
    CLASSES,
    CLASS_NAMES,
    MODELS
)

from datos import ajustar


# =====================================================
# 1. CARGAR MODELOS ENTRENADOS
# =====================================================

def cargar_modelos():

    print("\n========================================")
    print("CARGANDO MODELOS ENTRENADOS")
    print("========================================")

    ruta_unet = MODELS / "unet.keras"
    ruta_info = MODELS / "mejor_modelo.json"

    # Verificar que existan los archivos

    if not ruta_unet.exists():
        raise FileNotFoundError(
            f"No se encontro U-Net: {ruta_unet}"
        )

    if not ruta_info.exists():
        raise FileNotFoundError(
            f"No se encontro: {ruta_info}"
        )

    # Leer informacion del mejor clasificador

    with open(
        ruta_info,
        "r",
        encoding="utf-8"
    ) as archivo:

        informacion = json.load(archivo)

    nombre_modelo = informacion["modelo"]
    tipo_imagen = informacion["tipo"]

    # Construir la ruta del clasificador

    ruta_clasificador = (
        MODELS /
        f"{nombre_modelo}_{tipo_imagen}.keras"
    )

    if not ruta_clasificador.exists():
        raise FileNotFoundError(
            f"No se encontro el clasificador: "
            f"{ruta_clasificador}"
        )

    # Cargar U-Net

    print("\nCargando U-Net...")

    unet = tf.keras.models.load_model(
        ruta_unet,
        compile=False
    )

    # Cargar clasificador

    print("Cargando clasificador...")

    clasificador = tf.keras.models.load_model(
        ruta_clasificador,
        compile=False
    )

    print("\nModelos cargados correctamente")

    print("Segmentacion: U-Net")
    print("Clasificacion:", nombre_modelo)
    print("Tipo de imagen:", tipo_imagen)

    return unet, clasificador, informacion


# =====================================================
# 2. PREPROCESAR IMAGEN
# =====================================================

def preparar_imagen(imagen):

    # Corregir orientacion EXIF

    imagen = ImageOps.exif_transpose(
        imagen
    ).convert("RGB")

    # Ajustar al mismo tamaño del entrenamiento

    imagen = ajustar(imagen)

    # Convertir a arreglo NumPy

    imagen_array = np.asarray(
        imagen,
        dtype=np.uint8
    )

    return imagen_array


# =====================================================
# 3. SEGMENTACION CON U-NET
# =====================================================

def segmentar_hoja(imagen_array, unet):

    # Normalizar valores entre 0 y 1

    entrada = (
        imagen_array.astype(np.float32)
        / 255.0
    )

    # Agregar dimension del lote

    entrada = np.expand_dims(
        entrada,
        axis=0
    )

    # Predecir mascara

    prediccion = unet.predict(
        entrada,
        verbose=0
    )

    # Obtener mascara binaria

    mascara = (
        prediccion[0, :, :, 0] >= 0.5
    )

    # Aplicar mascara a la fotografia

    imagen_segmentada = imagen_array.copy()

    imagen_segmentada[~mascara] = 0

    # Imagen de mascara para visualizar

    mascara_visual = (
        mascara.astype(np.uint8) * 255
    )

    return (
        imagen_segmentada,
        mascara_visual
    )


# =====================================================
# 4. CLASIFICACION DE ENFERMEDADES
# =====================================================

def clasificar_hoja(
    imagen_array,
    clasificador
):

    # Los clasificadores reciben imagenes
    # con valores entre 0 y 255.
    # El preprocesamiento esta dentro del modelo.

    entrada = imagen_array.astype(
        np.float32
    )

    entrada = np.expand_dims(
        entrada,
        axis=0
    )

    # Prediccion

    probabilidades = clasificador.predict(
        entrada,
        verbose=0
    )[0]

    # Clase con mayor probabilidad

    indice = int(
        np.argmax(probabilidades)
    )

    clase = CLASSES[indice]

    nombre_clase = CLASS_NAMES[clase]

    confianza = float(
        probabilidades[indice] * 100
    )

    # Probabilidades para Gradio

    resultados = {}

    for i, clase_actual in enumerate(CLASSES):

        nombre = CLASS_NAMES[clase_actual]

        resultados[nombre] = float(
            probabilidades[i]
        )

    return (
        nombre_clase,
        confianza,
        resultados
    )


# =====================================================
# 5. ANALIZAR UNA FOTOGRAFIA
# =====================================================

def analizar_imagen(
    imagen,
    unet,
    clasificador,
    informacion
):

    if imagen is None:

        return (
            None,
            None,
            {},
            "Por favor, carga una fotografia."
        )

    try:

        # Preparar fotografia

        imagen_array = preparar_imagen(
            imagen
        )

        # Segmentar hoja

        imagen_segmentada, mascara = (
            segmentar_hoja(
                imagen_array,
                unet
            )
        )

        # Seleccionar imagen para clasificacion

        tipo = informacion["tipo"]

        if tipo == "segmentada":

            entrada_clasificacion = (
                imagen_segmentada
            )

        else:

            entrada_clasificacion = (
                imagen_array
            )

        # Clasificar

        nombre_clase, confianza, resultados = (
            clasificar_hoja(
                entrada_clasificacion,
                clasificador
            )
        )

        # Crear mensaje

        mensaje = (
            f"Clasificacion: {nombre_clase}\n"
            f"Probabilidad estimada: "
            f"{confianza:.2f}%\n"
            f"Modelo: {informacion['modelo']}\n"
            f"Tipo de imagen: {tipo}"
        )

        return (
            Image.fromarray(imagen_segmentada),
            Image.fromarray(mascara),
            resultados,
            mensaje
        )

    except Exception as error:

        print("Error durante el analisis:", error)

        raise gr.Error(
            f"No se pudo analizar la imagen: {error}"
        )


# =====================================================
# 6. CREAR INTERFAZ GRAFICA
# =====================================================

def iniciar_interfaz():

    # Cargar los modelos una sola vez

    unet, clasificador, informacion = (
        cargar_modelos()
    )

    # Funcion para conectar Gradio

    def ejecutar_analisis(imagen):

        return analizar_imagen(
            imagen,
            unet,
            clasificador,
            informacion
        )

    # Crear interfaz

    with gr.Blocks(
        title="Clasificacion de enfermedades del cafe"
    ) as interfaz:

        gr.Markdown(
            """
            # Clasificación de enfermedades en hojas de café

            Sistema de visión computacional basado
            en aprendizaje profundo.

            **Segmentación:** U-Net

            **Clasificación:** MobileNetV2

            **Base de datos:** RoCoLe

            **Categorías:**
            - Hoja sana
            - Roya nivel 1
            - Roya nivel 2
            - Roya nivel 3
            - Roya nivel 4
            - Ácaro rojo
            """
        )

        # Entrada

        with gr.Row():

            with gr.Column():

                imagen_entrada = gr.Image(
                    label="Cargar fotografia de la hoja",
                    type="pil"
                )

                boton_analizar = gr.Button(
                    "Analizar imagen",
                    variant="primary"
                )

                boton_limpiar = gr.Button(
                    "Limpiar"
                )

            # Salidas visuales

            with gr.Column():

                imagen_segmentada = gr.Image(
                    label="Hoja segmentada"
                )

                imagen_mascara = gr.Image(
                    label="Mascara generada por U-Net"
                )

        # Resultados

        gr.Markdown(
            "## Resultado de clasificacion"
        )

        resultado_texto = gr.Textbox(
            label="Diagnostico estimado",
            lines=4,
            interactive=False
        )

        probabilidades = gr.Label(
            label="Probabilidades por clase",
            num_top_classes=6
        )

        # Evento de analisis

        boton_analizar.click(
            fn=ejecutar_analisis,
            inputs=[imagen_entrada],
            outputs=[
                imagen_segmentada,
                imagen_mascara,
                probabilidades,
                resultado_texto
            ]
        )

        # Evento de limpieza

        boton_limpiar.click(
            fn=lambda: (
                None,
                None,
                None,
                {},
                ""
            ),
            inputs=[],
            outputs=[
                imagen_entrada,
                imagen_segmentada,
                imagen_mascara,
                probabilidades,
                resultado_texto
            ]
        )

        # Informacion del sistema

        gr.Markdown(
            """
            ---

            **Nota:** Este sistema es un prototipo
            académico de clasificación de enfermedades
            en hojas de café.

            Las predicciones son estimaciones
            computacionales y no sustituyen
            una evaluación fitosanitaria profesional.
            """
        )

    # Iniciar servidor

    print("\n========================================")
    print("INTERFAZ LISTA")
    print("========================================")

    print(
        "Modelo:",
        informacion["modelo"]
    )

    print(
        "Tipo:",
        informacion["tipo"]
    )

    print("\nAbriendo interfaz en el navegador...")

    interfaz.launch(
        inbrowser=True,
        share=False
    )


# =====================================================
# 7. EJECUCION DIRECTA
# =====================================================

if __name__ == "__main__":

    iniciar_interfaz()
