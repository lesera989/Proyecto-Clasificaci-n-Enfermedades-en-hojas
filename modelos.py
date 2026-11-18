
import tensorflow as tf
from tensorflow.keras import layers as L

from config import SIZE, CLASSES


# =====================================================
# BLOQUES DE CONVOLUCION
# =====================================================

def bloque_conv(x, filtros):

    x = L.Conv2D(
        filtros, 3, padding="same",
        activation="relu"
    )(x)

    x = L.BatchNormalization()(x)

    x = L.Conv2D(
        filtros, 3, padding="same",
        activation="relu"
    )(x)

    x = L.BatchNormalization()(x)

    return x


# =====================================================
# MODELO U-NET PARA SEGMENTACION
# =====================================================

def crear_unet():

    entradas = L.Input(
        shape=(SIZE, SIZE, 3)
    )

    c1 = bloque_conv(entradas, 16)
    p1 = L.MaxPooling2D()(c1)

    c2 = bloque_conv(p1, 32)
    p2 = L.MaxPooling2D()(c2)

    c3 = bloque_conv(p2, 64)
    p3 = L.MaxPooling2D()(c3)

    c4 = bloque_conv(p3, 128)
    p4 = L.MaxPooling2D()(c4)

    centro = bloque_conv(p4, 256)

    u4 = L.UpSampling2D()(centro)
    u4 = L.Concatenate()([u4, c4])
    c5 = bloque_conv(u4, 128)

    u3 = L.UpSampling2D()(c5)
    u3 = L.Concatenate()([u3, c3])
    c6 = bloque_conv(u3, 64)

    u2 = L.UpSampling2D()(c6)
    u2 = L.Concatenate()([u2, c2])
    c7 = bloque_conv(u2, 32)

    u1 = L.UpSampling2D()(c7)
    u1 = L.Concatenate()([u1, c1])
    c8 = bloque_conv(u1, 16)

    salidas = L.Conv2D(
        1, 1, activation="sigmoid"
    )(c8)

    modelo = tf.keras.Model(
        entradas, salidas, name="UNet_Cafe"
    )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.BinaryIoU(
                target_class_ids=[1],
                threshold=0.5,
                name="iou"
            )
        ]
    )

    return modelo


# =====================================================
# AUMENTO DE DATOS
# =====================================================

def aumento_datos():

    return tf.keras.Sequential([
        L.RandomFlip("horizontal_and_vertical"),
        L.RandomRotation(0.10),
        L.RandomZoom(0.10)
    ], name="Aumento_Datos")


# =====================================================
# CLASIFICADOR CNN
# =====================================================

def crear_cnn():

    entradas = L.Input(
        shape=(SIZE, SIZE, 3)
    )

    x = aumento_datos()(entradas)
    x = L.Rescaling(1.0 / 255.0)(x)

    for filtros in [32, 64, 128]:

        x = L.Conv2D(
            filtros, 3,
            padding="same",
            activation="relu"
        )(x)

        x = L.BatchNormalization()(x)
        x = L.MaxPooling2D()(x)

    x = L.GlobalAveragePooling2D()(x)

    x = L.Dense(
        128, activation="relu"
    )(x)

    x = L.Dropout(0.4)(x)

    salidas = L.Dense(
        len(CLASSES),
        activation="softmax"
    )(x)

    return tf.keras.Model(
        entradas, salidas,
        name="CNN_Cafe"
    )


# =====================================================
# CLASIFICADORES CON TRANSFER LEARNING
# =====================================================

def crear_clasificador(nombre):

    nombre = nombre.lower()

    if nombre == "cnn":
        modelo = crear_cnn()

    elif nombre in ["mobilenetv2", "efficientnetb0"]:

        if nombre == "mobilenetv2":
            arquitectura = (
                tf.keras.applications.MobileNetV2
            )

        else:
            arquitectura = (
                tf.keras.applications.EfficientNetB0
            )

        try:
            base = arquitectura(
                include_top=False,
                weights="imagenet",
                input_shape=(SIZE, SIZE, 3)
            )

            print(
                "Pesos ImageNet cargados:",
                nombre
            )

        except Exception as error:

            print(
                "ADVERTENCIA: no se pudieron "
                "cargar los pesos ImageNet."
            )
            print(error)
            print(
                "El modelo se inicializara "
                "sin preentrenamiento."
            )

            base = arquitectura(
                include_top=False,
                weights=None,
                input_shape=(SIZE, SIZE, 3)
            )

        base.trainable = False

        entradas = L.Input(
            shape=(SIZE, SIZE, 3)
        )

        x = aumento_datos()(entradas)

        if nombre == "mobilenetv2":

            x = L.Rescaling(
                scale=1.0 / 127.5,
                offset=-1.0
            )(x)

        # EfficientNetB0 ya incorpora su
        # propio preprocesamiento interno.

        x = base(x, training=False)

        x = L.GlobalAveragePooling2D()(x)

        x = L.Dense(
            128, activation="relu"
        )(x)

        x = L.Dropout(0.4)(x)

        salidas = L.Dense(
            len(CLASSES),
            activation="softmax"
        )(x)

        modelo = tf.keras.Model(
            entradas, salidas,
            name=nombre + "_Cafe"
        )

    else:
        raise ValueError(
            "Modelo desconocido: " + nombre
        )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modelo
