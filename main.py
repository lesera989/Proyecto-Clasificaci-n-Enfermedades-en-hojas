
import argparse

from config import crear_carpetas


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Proyecto de clasificacion y segmentacion "
            "de hojas de cafe - RoCoLe"
        )
    )

    parser.add_argument(
        "accion",
        choices=[
            "preparar",
            "entrenar_unet",
            "generar_segmentadas",
            "entrenar_clasificadores",
            "evaluar",
            "interfaz"
        ],
        help="Etapa del proyecto que se ejecutara"
    )

    args = parser.parse_args()

    crear_carpetas()

    if args.accion == "preparar":

        from datos import preparar
        preparar()

    elif args.accion == "entrenar_unet":

        from entrenar import entrenar_unet
        entrenar_unet()

    elif args.accion == "generar_segmentadas":

        from entrenar import generar_segmentadas
        generar_segmentadas()

    elif args.accion == "entrenar_clasificadores":

        from entrenar import entrenar_clasificadores
        entrenar_clasificadores()

    elif args.accion == "evaluar":

        from evaluar import evaluar
        evaluar()

    elif args.accion == "interfaz":

        from app import iniciar_interfaz
        iniciar_interfaz()


if __name__ == "__main__":
    main()
