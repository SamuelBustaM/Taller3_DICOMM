# Procesador DICOM — Informática 2: Unidad 3

**Taller Evaluativo — Introducción a la Informática Médica**

**Integrantes**
Samuel Bustamante Muñoz - s.bustamante@udea.edu.co

---

## 1. Descripción del Proyecto

Este proyecto implementa una aplicación en Python orientada a objetos (ProcesadorDICOM) que automatiza el flujo de trabajo con archivos médicos en formato DICOM.

El programa realiza los siguientes pasos de forma organizada:

1. Carga de archivos DICOM desde un directorio (con manejo de errores).
2. Extracción de metadatos estándar (PatientID, StudyDate, Modality, etc.).
3. Estructuración de los datos en un DataFrame de Pandas.
4. Análisis de intensidad promedio de píxeles con NumPy.
5. Procesamiento de imágenes con OpenCV: normalización, ecualización de histograma y detección de bordes con Canny.
6. Exportación de resultados a CSV y guardado de imágenes PNG.

---

## 2. Entorno de ejecución

### ¿Por qué se usa un entorno virtual?

Un entorno virtual (`venv`) es una instalación aislada de Python que contiene sus propias librerías, independiente del sistema operativo y de otros proyectos. Esto evita conflictos de versiones entre dependencias de distintos proyectos y garantiza que el código se ejecute siempre en el mismo entorno, sin importar la máquina donde se corra.

### Crear y activar el entorno virtual

Desde la terminal, dentro de la carpeta del proyecto:

```bash
# 1. Crear el entorno virtual en una carpeta llamada 'venv'
python -m venv venv

# 2. Activar el entorno virtual
venv\Scripts\activate
```

Una vez activo, el prompt de la terminal mostrará `(venv)` al inicio, indicando que el entorno está listo.

### Instalar las dependencias

Con el entorno virtual activo, se instalan todas las librerías necesarias:

```bash
pip install numpy pandas opencv-python matplotlib pydicom
```

---

## 3. Instalación y uso

### Ejecutar el procesador

El código está configurado para leer archivos DICOM desde la carpeta `data_store/data/` y guardar los resultados en `output`.

```python
directorio_dicom = "data_store/data"   # carpeta con archivos .dcm
carpeta_salida   = "output"            # carpeta donde se guardan los resultados
```

### Estructura de salida

```
output/
├── equalized/              ← Imágenes con ecualización de histograma (.png)
├── edges/                  ← Imágenes con bordes detectados por Canny (.png)
└── metadatos_dicom.csv     ← Tabla con todos los metadatos extraídos
```

### Estructura del repositorio

```
TALLER3_DICOM/
├── main.py                 ← Script principal (clase ProcesadorDICOM)
├── data_store/
│   └── data/               ← Carpeta para archivos DICOM de entrada
└── output/                 ← Se genera automáticamente al ejecutar
    ├── equalized/
    ├── edges/
    └── metadatos_dicom.csv
```

---

## 4. Preguntas teóricas

### 4.1 ¿Por qué DICOM y HL7 son cruciales para la interoperabilidad en salud y en qué se diferencian?

La interoperabilidad en sistemas de salud es, la capacidad de que distintos equipos, software e instituciones intercambien información de forma coherente dependiendo fundamentalmente de estándares abiertos y ampliamente adoptados. DICOM y HL7 son los dos pilares principales, aunque operan en capas distintas.

*DICOM (Digital Imaging and Communications in Medicine)* es un estándar diseñado específicamente para el manejo de imágenes médicas y su información asociada. Define tanto el formato del archivo (cómo se almacena una imagen de tomografía, resonancia magnética, radiografía, etc.) como el protocolo de red para transmitirla entre dispositivos (modalidades, PACS, estaciones de trabajo). Cada archivo DICOM contiene incrustados en su cabecera los metadatos del paciente y del estudio (nombre, ID, fecha, modalidad, parámetros de adquisición), lo que garantiza que la imagen nunca esté desvinculada de su contexto clínico. Sin DICOM, un tomógrafo de una marca no podría enviar imágenes a un visualizador de otra marca.

*HL7 (Health Level Seven)*, en sus versiones 2.x y la más moderna FHIR (Fast Healthcare Interoperability Resources), es un estándar orientado al intercambio de mensajes clínicos y administrativos entre sistemas de información hospitalaria (HIS, LIS, RIS, EHR). Gestiona datos como órdenes médicas, resultados de laboratorio, altas, diagnósticos, medicaciones y registros de pacientes. HL7 FHIR, en particular, usa recursos basados en API REST y JSON/XML, lo que facilita la integración con aplicaciones web modernas.

**Diferencia conceptual:**

| Aspecto | DICOM | HL7 / FHIR |
|---------|-------|------------|
| Dominio principal | Imágenes médicas | Información clínica y administrativa |
| Qué transporta | Píxeles + metadatos de imagen | Órdenes, resultados, registros clínicos |
| Protocolo de red | DIMSE (propio) / DICOMweb | HTTP/REST (FHIR), mensajes v2.x |
| Formato de archivo | .dcm (binario con cabecera) | JSON, XML, HL7 v2 (texto delimitado) |
| Ejemplo de uso | Enviar una radiografía del TAC al PACS | Notificar la orden de un estudio al RIS |

En la práctica clínica ambos coexisten: cuando un médico solicita una tomografía, HL7 lleva la orden desde el HIS al RIS; cuando el equipo adquiere las imágenes, DICOM las transfiere al PACS; y cuando el radiólogo valida el informe, HL7 devuelve el resultado al HIS. Son complementarios, no competidores.

---

### 4.2 Ventajas, limitaciones y escenarios de uso de la ecualización de histograma y la detección de bordes con Canny en imágenes médicas

#### Ecualización de histograma

La ecualización de histograma redistribuye la intensidad de los píxeles para que la imagen utilice todo el rango dinámico disponible [0,255]. El efecto práctico es un mayor contraste global.

**Ventajas:**
- Mejora la visibilidad de estructuras con bajo contraste en radiografías y tomografías (útil para revisión preliminar y control de calidad).
- Es computacionalmente muy eficiente y simple de aplicar.
- Funciona bien como paso de preprocesamiento antes de algoritmos de segmentación o detección.

**Limitaciones:**
- Amplifica el ruido junto con la señal; en imágenes con ruido de quantum (radiografías de bajo kVp) puede hacer el ruido más prominente.
- Altera los valores de intensidad Hounsfield (en tomografías), por lo que no debe aplicarse antes de análisis cuantitativos de densidad tisular.
- La versión global (equalizeHist) puede sobreexponer regiones ya brillantes. La variante CLAHE (cv2.createCLAHE) es más adecuada para imágenes médicas porque aplica la ecualización de forma localizada, preservando mejor el contexto local.

**Escenarios donde es útil:**
- Preprocesamiento visual antes de clasificadores de aprendizaje automático.
- Mejora de contraste en radiografías de tórax para detectar consolidaciones.
- Visualización rápida en sistemas de revisión sin calibración de ventana/nivel.

**Escenarios donde puede ser perjudicial:**
- Análisis cuantitativo de densidad ósea o grasa (distorsiona los valores reales).
- Imágenes de resonancia magnética donde las variaciones de intensidad tienen significado diagnóstico.
- Cualquier flujo en el que los valores de píxel deban conservar su significado físico original.

---

#### Detección de bordes con Canny

El algoritmo de Canny detecta cambios abruptos de intensidad (gradientes), marcando los contornos de estructuras en la imagen.

**Justificación de los umbrales utilizados (threshold1=50, threshold2=150):**
- La razón 1:3 entre umbrales es la recomendada por el propio Canny para suprimir bordes irrelevantes.
- threshold1=50, conserva bordes suaves (tejidos blandos, contornos de órganos).
- threshold2=150, descarta ruido y retiene bordes fuertes (hueso, contraste vascular).
- Este par es un punto de partida estándar en imágenes de rayos X y TC; deben ajustarse según la modalidad específica.

**Ventajas:**
- Resalta estructuras anatómicas como bordes óseos, siluetas cardíacas y vasculares.
- Muy útil como preprocesamiento para algoritmos de segmentación automática.
- Reduce la información a representaciones esqueleto/contorno, útil para análisis de forma.

**Limitaciones:**
- Muy sensible al ruido; sin un suavizado previo adecuado (Gaussiano) puede producir falsos bordes.
- Los umbrales deben ajustarse caso a caso según la modalidad (CT, MRI, RX), lo que dificulta la automatización sin validación clínica.
- No diferencia entre bordes clínicamente relevantes y artefactos (implantes metálicos, ruido).
- En imágenes de resonancia magnética con alto nivel de ruido térmico puede producir resultados de baja calidad.

**Escenarios donde es útil:**
- Detección automática del contorno del corazón en radiografías de tórax.
- Preprocesamiento para medir ángulos o distancias anatómicas en ortopedia.
- Segmentación de huesos en tomografías para cirugía asistida por imagen.

**Escenarios donde puede ser perjudicial:**
- Diagnóstico directo sin supervisión médica: los bordes detectados no equivalen a diagnóstico.
- Imágenes con artefactos metálicos donde el algoritmo produce falsos bordes prominentes.
- Imágenes de resonancia funcional (fMRI) donde el interés es la intensidad de señal, no los bordes.

---

## 5. Dificultades encontradas e importancia de las herramientas Python

### Dificultades encontradas

1. **Tags faltantes por anonimización:** Muchos archivos DICOM de acceso público tienen tags como "PatientName" o "PatientID" removidos o reemplazados. Fue necesario implementar manejo defensivo para todos los tags con valores por defecto ("N/A").

2. **Archivos DICOM sin datos de píxeles:** Modalidades como SR (Structured Report) o PR (Presentation State) no contienen "pixel_array". El manejo con bloques "try/except" fue importante para no interrumpir el procesamiento completo.

3. **Profundidad de bits y normalización:** Las imágenes DICOM en 12 o 16 bits no pueden usarse directamente con OpenCV. La normalización a uint8 debe realizarse preservando el rango dinámico real de la imagen para no perder información de contraste.

4. **Imágenes multifotograma:** Algunos archivos DICOM contienen múltiples frames (estudios dinámicos). Fue necesario seleccionar el primer frame para el procesamiento simple.

### Importancia de las herramientas Python para análisis de datos médicos

Python se ha convertido en el lenguaje estándar en informática médica e investigación clínica por varias razones:

- **pydicom** permite leer y manipular el estándar DICOM sin depender de software propietario, democratizando el acceso a imágenes médicas para investigación y desarrollo.
- **NumPy** proporciona operaciones vectorizadas sobre arreglos de píxeles de forma eficiente, fundamental cuando se manejan volúmenes de tomografías de cientos de megabytes.
- **Pandas** facilita la estructuración, exploración y exportación de grandes conjuntos de metadatos clínicos, integrándose directamente con bases de datos y pipelines de análisis estadístico.
- **OpenCV** aporta una biblioteca madura de algoritmos de procesamiento de imagen usada tanto en investigación como en producción, con excelente rendimiento y documentación.
- El ecosistema Python completo permite escalar desde el preprocesamiento básico hasta modelos de inteligencia artificial para diagnóstico asistido, todo en el mismo lenguaje y entorno.

La combinación de estas herramientas de software libre elimina barreras económicas y de licencia, permitiendo que investigadores, hospitales públicos y países en desarrollo implementen soluciones de informática médica de calidad sin costos de licenciamiento.
