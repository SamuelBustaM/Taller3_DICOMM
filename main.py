import numpy as np
import pandas as pd
import cv2
import pydicom
from pathlib import Path

class ProcesadorDICOM:
    """
    Clase encargada de automatizar el flujo de procesamiento DICOM:
    carga de archivos, extracción de metadatos, análisis de imágenes,
    procesamiento con OpenCV y exportación de resultados.
    """

    # Tags DICOM que serán extraídos y organizados en el DataFrame final.
    TAGS = [
        ("PatientID",        "PatientID"),
        ("PatientName",      "PatientName"),
        ("StudyInstanceUID", "StudyInstanceUID"),
        ("StudyDescription", "StudyDescription"),
        ("StudyDate",        "StudyDate"),
        ("Modality",         "Modality"),
        ("Rows",             "Rows"),
        ("Columns",          "Columns"),
    ]

    def __init__(self, directorio, salida="output"):
        """
        Inicializa rutas de entrada/salida y crea automáticamente
        las carpetas donde se almacenarán los resultados procesados.
        """
        self.directorio = Path(directorio)
        self.salida = Path(salida)

        self.salida_ecualizada = self.salida / "equalized"
        self.salida_bordes = self.salida / "edges"

        self.salida_ecualizada.mkdir(parents=True, exist_ok=True)
        self.salida_bordes.mkdir(parents=True, exist_ok=True)

        self.datasets = []
        self.df = pd.DataFrame()

    # 1. Carga de archivos DICOM

    def cargar_archivos(self):
        """
        Busca archivos DICOM en el directorio indicado y carga cada
        dataset utilizando pydicom. Los archivos inválidos se omiten.
        """
        archivos = [p for p in self.directorio.rglob("*") if p.is_file()]
        print(f"Archivos encontrados: {len(archivos)}")

        cargados = 0

        for ruta in archivos:
            try:
                ds = pydicom.dcmread(str(ruta))
                ds.filepath = ruta

                self.datasets.append(ds)
                cargados += 1

            except Exception as e:
                print(f"  [OMITIDO] {ruta.name}: {e}")

        print(f"Archivos DICOM cargados: {cargados}")

    # 2. Extracción de metadatos

    def extraer_metadatos(self):
        """
        Extrae los tags definidos en TAGS y construye un DataFrame
        con la información de cada archivo DICOM.
        """
        filas = []

        for ds in self.datasets:

            fila = {"Archivo": ds.filepath.name}

            for col, keyword in self.TAGS:

                try:
                    valor = getattr(ds, keyword, None)

                    fila[col] = (
                        str(valor).strip()
                        if valor is not None
                        else "N/A"
                    )

                except Exception:
                    fila[col] = "N/A"

            filas.append(fila)

        self.df = pd.DataFrame(filas)

        print(f"Metadatos extraídos: {len(self.df)} filas")

    # 3. Cálculo de intensidad promedio

    def calcular_intensidad_promedio(self):
        """
        Calcula el promedio de intensidad de píxeles para cada imagen
        DICOM utilizando NumPy y agrega el resultado al DataFrame.
        """
        intensidades = []

        for ds in self.datasets:

            try:
                if not hasattr(ds, "PixelData"):
                    intensidades.append(None)
                    continue

                # Evitar formatos comprimidos problemáticos
                transfer_syntax = getattr(
                    ds.file_meta,
                    "TransferSyntaxUID",
                    ""
                )

                transfer_syntax = str(transfer_syntax)

                if "JPEG" in transfer_syntax or "1.2.840.10008.1.2.4" in transfer_syntax:
                    intensidades.append(None)
                    continue

                promedio = np.mean(ds.pixel_array)

                intensidades.append(
                    round(float(promedio), 4)
                )

            except Exception:
                intensidades.append(None)

        self.df["IntensidadPromedio"] = intensidades

        print("Columna 'IntensidadPromedio' agregada.")

    # 4. Procesamiento de imágenes con OpenCV

    def procesar_imagenes(self):
        """
        Aplica normalización, ecualización de histograma y detección
        de bordes con Canny a las imágenes DICOM válidas.
        """
        CANNY_LOW = 50
        CANNY_HIGH = 150

        procesadas = 0

        for ds in self.datasets:

            nombre_base = ds.filepath.stem

            try:
                if not hasattr(ds, "PixelData"):
                    continue

                transfer_syntax = getattr(
                    ds.file_meta,
                    "TransferSyntaxUID",
                    ""
                )

                transfer_syntax = str(transfer_syntax)

                if "JPEG" in transfer_syntax or "1.2.840.10008.1.2.4" in transfer_syntax:
                    print(f"  [Omitido] Compresión no soportada: {ds.filepath.name}")
                    continue

                pixels = ds.pixel_array

                # Conversión básica para imágenes RGB o multiframe.
                if pixels.ndim == 3:

                    if pixels.shape[-1] == 3:
                        pixels = cv2.cvtColor(
                            pixels,
                            cv2.COLOR_RGB2GRAY
                        )

                    else:
                        pixels = pixels[0]

                # Normalización al rango [0,255].
                arr = pixels.astype(np.float32)

                minv, maxv = arr.min(), arr.max()

                if maxv != minv:

                    img_uint8 = (
                        (arr - minv) / (maxv - minv) * 255
                    ).astype(np.uint8)

                else:

                    img_uint8 = np.zeros_like(
                        arr,
                        dtype=np.uint8
                    )

                # Procesamiento de contraste y bordes.
                img_ecualizada = cv2.equalizeHist(img_uint8)

                img_bordes = cv2.Canny(
                    img_ecualizada,
                    CANNY_LOW,
                    CANNY_HIGH
                )

                # Exportación de imágenes procesadas.
                cv2.imwrite(
                    str(self.salida_ecualizada / f"{nombre_base}_eq.png"),
                    img_ecualizada
                )

                cv2.imwrite(
                    str(self.salida_bordes / f"{nombre_base}_edges.png"),
                    img_bordes
                )

                procesadas += 1

                print(f"  Procesada: {ds.filepath.name}")

            except Exception as e:

                print(
                    f"  [Omitido] Sin píxeles en "
                    f"'{ds.filepath.name}': {e}"
                )

        print(f"Imágenes procesadas y guardadas: {procesadas}")

    # 5. Exportación y resumen

    def exportar_csv(self):
        """
        Exporta el DataFrame final a formato CSV.
        """
        ruta_csv = self.salida / "metadatos_dicom.csv"

        self.df.to_csv(
            ruta_csv,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"CSV guardado en: {ruta_csv}")

    def mostrar_resumen(self):
        """
        Muestra en consola un resumen de los metadatos procesados.
        """
        print("\n" + "=" * 60)

        print("  Resumen de metadatos con DICOM")

        print("=" * 60)

        print(self.df.to_string(index=False))

        print("=" * 60)

        print(f"Total de archivos procesados: {len(self.df)}\n")

    # Método principal

    def ejecutar(self):
        """
        Ejecuta el flujo completo del procesamiento DICOM.
        """
        print(" Iniciando ProcesadorDICOM ")

        self.cargar_archivos()

        self.extraer_metadatos()

        self.calcular_intensidad_promedio()

        self.procesar_imagenes()

        self.exportar_csv()

        self.mostrar_resumen()

        print("Procesamiento finalizado")


# Configuración de rutas de entrada y salida.
directorio_dicom = "data_store/data"
carpeta_salida = "output"

# Ejecución principal del procesamiento.
procesador = ProcesadorDICOM(
    directorio_dicom,
    carpeta_salida
)

procesador.ejecutar()
