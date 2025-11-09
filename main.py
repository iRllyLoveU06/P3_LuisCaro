# ---  Importación de Librerías ---
import os
import cv2  # Necesario para las banderas de binarización y morfología
import pandas as pd
import matplotlib.pyplot as plt
from clases_dicom import DicomManager, Estudiolmaginologico

def mostrar_menu():
    """Imprime el menú principal en la consola."""
    print("\n--- 🏥 Menú Principal: Procesador DICOM ---")
    print("1. Cargar nueva carpeta DICOM (Crear Estudio)")
    print("2. Mostrar cortes 3D (Transversal, Sagital, Coronal) de un estudio")
    print("3. Aplicar ZOOM (Recorte y Redimensión) a un corte")
    print("4. Aplicar Segmentación (Binarización) a un corte")
    print("5. Aplicar Transformación Morfológica a un corte")
    print("6. Convertir estudio DICOM a NIFTI")
    print("7. Exportar metadatos de estudios cargados a CSV")
    print("0. Salir")