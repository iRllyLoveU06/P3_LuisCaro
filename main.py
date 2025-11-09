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
    """Imprime el menú :3 """
    print("\n--- Menú Principal: Procesador DICOM ---")
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
    
def cargar_nuevo_estudio():
    """
    Opción 1: Pide una ruta, crea los objetos DicomManager y
    Estudiolmaginologico, y los almacena en el diccionario del inicio :3
    """
    ruta_carpeta = input("Ingrese la ruta de la carpeta con los archivos DICOM: ")
    
    if not os.path.isdir(ruta_carpeta):
        print(f"[Error] La ruta '{ruta_carpeta}' no es una carpeta válida.")
        return

    print("Cargando y procesando... por favor espere.")
    try:
        # Creamos el DicomManager 
        manager = DicomManager(ruta_carpeta)
        
        # si manager.volumen_3d será None, la reconstrucción falló 
        if manager.volumen_3d is None:
            print("[Error] No se pudo reconstruir el volumen 3D. Verifique los archivos.")
            return
            
        # Creamos el Estudiolmaginologico
        estudio = Estudiolmaginologico(manager)
        
        # almacenamos el objeto 
        estudios_cargados[ruta_carpeta] = estudio
        
        print(f"\n¡Estudio cargado exitosamente!")
        print(f"  - Modalidad: {estudio.study_modality}")
        print(f"  - Descripción: {estudio.study_description}")
        print(f"  - Forma 3D: {estudio.forma}")
        
    except Exception as e:
        print(f"[Error] Ocurrió un problema al cargar el estudio: {e}")


def mostrar_cortes_3d():
    """
    Opción 2: Muestra los 3 cortes principales (T, S, C).
    """
    estudio_seleccionado = seleccionar_estudio()
    if estudio_seleccionado is None:
        return

    # Obtenemos los cortes desde el manager (que está dentro del estudio)
    trans, sag, cor = estudio_seleccionado.manager_dicom.obtener_cortes_principales()

    if trans is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6)) #consampiramos los lienzos 
        
        ax1.imshow(trans, cmap='gray', aspect='auto')
        ax1.set_title("Corte Transversal (Axial)")
        ax1.axis('off')
        
        ax2.imshow(sag.T, cmap='gray', aspect='auto') # Usamos .T (Transpuesta) para orientación correcta
        ax2.set_title("Corte Sagital")
        ax2.axis('off')

        ax3.imshow(cor.T, cmap='gray', aspect='auto') # Usamos .T (Transpuesta) para orientación correcta
        ax3.set_title("Corte Coronal")
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()

def aplicar_zoom():
    """
    Opción 3: Pide parámetros y llama al metodo_zoom.
    """
    estudio_seleccionado = seleccionar_estudio()
    if estudio_seleccionado is None:
        return

    print(f"El volumen tiene {estudio_seleccionado.forma[0]} cortes (índice 0 a {estudio_seleccionado.forma[0]-1})")
    try:
        idx = int(input("Ingrese el índice del corte a procesar: "))
        if not (0 <= idx < estudio_seleccionado.forma[0]):
            print("[Error] Índice de corte fuera de rango.")
            return
            
        print(f"El corte tiene dimensiones: {estudio_seleccionado.forma[1]} Alto x {estudio_seleccionado.forma[2]} Ancho")
        x = int(input("Ingrese coordenada X de inicio (esquina sup-izq): "))
        y = int(input("Ingrese coordenada Y de inicio (esquina sup-izq): "))
        w = int(input("Ingrese Ancho (W) del recorte: "))
        h = int(input("Ingrese Alto (H) del recorte: "))
        nombre = input("Nombre del archivo de salida (ej: zoom.png): ")

        estudio_seleccionado.metodo_zoom(idx, x, y, w, h, nombre)

    except ValueError:
        print("[Error] Entrada inválida. Todos los valores deben ser números enteros.")
    except Exception as e:
        print(f"[Error] No se pudo aplicar el zoom: {e}")