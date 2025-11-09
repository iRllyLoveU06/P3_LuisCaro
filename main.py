# ---  Importación de Librerías ---
import os
import cv2  # Necesario para las banderas de binarización y morfología
import pandas as pd
import matplotlib.pyplot as plt
from clases_dicom import DicomManager, Estudiolmaginologico


# Un diccionario para almacenar los objetos creados, la llave es la ruta de la carpeta y el valor será un objeto EstudioImaginologico
estudios_cargados = {}
#--- definimos varias funciones para realizar el bucle y sistema mas sencillo

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




def seleccionar_estudio():
    """
    Función auxiliar para mostrar los estudios cargados y
    permitir al usuario seleccionar uno para trabajar.
    """
    if not estudios_cargados:
        print("\n[Error] No hay estudios cargados. Por favor, cargue un estudio primero (Opción 1).")
        return None

    print("\n--- Estudios Cargados ---")
    # Convertimos las llaves del diccionario (rutas) a una lista
    lista_estudios = list(estudios_cargados.keys())
    
    for i, ruta in enumerate(lista_estudios):
        print(f"  [{i+1}] {ruta}")
    
    try:
        opcion = int(input("Seleccione el número del estudio: "))
        if 1 <= opcion <= len(lista_estudios):
            # Devolvemos el objeto Estudiolmaginologico seleccionado
            ruta_seleccionada = lista_estudios[opcion - 1]
            return estudios_cargados[ruta_seleccionada]
        else:
            print("[Error] Selección fuera de rango.")
            return None
    except ValueError:
        print("[Error] Entrada inválida. Debe ser un número.")
        return None