# ---  Importación de Librerías ---
import os
import cv2  # Necesario para las banderas de binarización y morfología
import pandas as pd
import matplotlib.pyplot as plt
from clases_dicom import DicomManager, Estudiolmaginologico
import numpy as np


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


def aplicar_segmentacion():
    """
    Opción 4: Pide tipo de binarización y llama a funcion_segmentacion.
    """
    estudio_seleccionado = seleccionar_estudio()
    if estudio_seleccionado is None:
        return
        
    try:
        idx = int(input(f"Ingrese el índice del corte (0 a {estudio_seleccionado.forma[0]-1}): "))
        if not (0 <= idx < estudio_seleccionado.forma[0]):
            print("[Error] Índice de corte fuera de rango.")
            return

        print("\n--- Tipos de Binarización ---")
        print("1. Binario (cv2.THRESH_BINARY)")
        print("2. Binario Invertido (cv2.THRESH_BINARY_INV)")
        print("3. Truncado (cv2.THRESH_TRUNC)")
        print("4. A Cero (cv2.THRESH_TOZERO)")
        print("5. A Cero Invertido (cv2.THRESH_TOZERO_INV)")
        
        opcion = input("Seleccione el tipo de binarización (1-5): ")

        # Mapeamos la entrada del usuario a la bandera real de OpenCV
        mapeo_binarizacion = {
            "1": cv2.THRESH_BINARY,
            "2": cv2.THRESH_BINARY_INV,
            "3": cv2.THRESH_TRUNC,
            "4": cv2.THRESH_TOZERO,
            "5": cv2.THRESH_TOZERO_INV
        }
        
        bandera_seleccionada = mapeo_binarizacion.get(opcion)
        
        if bandera_seleccionada is None:
            print("[Error] Opción de binarización no válida.")
            return

        # Pedimos el umbral.
        umbral = int(input("Ingrese el valor del umbral (0-255, ej: 127): "))
        
        # Llamamos al método de la clase
        estudio_seleccionado.funcion_segmentacion(idx, bandera_seleccionada, umbral)
        
    except ValueError:
        print("[Error] Entrada inválida. Debe ser un número entero.")

def convertir_a_nifti():
    """
    Opción 6: Llama al método de conversión a NIFTI del DicomManager.
    """
    estudio_seleccionado = seleccionar_estudio()
    if estudio_seleccionado is None:
        return
        
    nombre_salida = input("Ingrese el nombre del archivo de salida (agregue "".nii"" o "".nii.gz""): ")
    if not (nombre_salida.endswith(".nii") or nombre_salida.endswith(".nii.gz")):
        print("Advertencia: El nombre de archivo no termina en .nii o .nii.gz")
        
    try:
        # El método de conversión está en el DicomManager
        estudio_seleccionado.manager_dicom.convertir_a_nifti(nombre_salida)
    except Exception as e:
        print(f"[Error] No se pudo convertir a NIFTI: {e}")

def exportar_metadata_csv():
    """
    Opción 7: Extrae metadatos de TODOS los estudios cargados
    y los guarda en un archivo CSV.
    """
    if not estudios_cargados:
        print("\n[Error] No hay estudios cargados para exportar.")
        return

    lista_de_estudios_data = []
    
    # Iteramos sobre el diccionario de estudios cargados
    for ruta, estudio in estudios_cargados.items():
        datos_fila = {
            "Ruta Carpeta": ruta,
            "Modalidad": estudio.study_modality,
            "Fecha Estudio": estudio.study_date,
            "Hora Estudio": estudio.study_time,
            "Hora Serie": estudio.series_time,
            "Duracion Estudio": estudio.tiempo_duracion_estudio,
            "Descripcion": estudio.study_description,
            "Forma 3D": str(estudio.forma)
        }
        lista_de_estudios_data.append(datos_fila)

    # Creamos el DataFrame de Pandas
    df = pd.DataFrame(lista_de_estudios_data)
    
    # Guardamos en CSV
    nombre_csv = "metadata_estudios.csv"
    df.to_csv(nombre_csv, index=False)
    
    print(f"\n¡Metadatos exportados exitosamente a '{nombre_csv}'!")
    print(df)
    

def aplicar_morfologia():
    """
    Opción 5: Pide tamaño de kernel y operación, y llama a transformacion_morfologica.
    """
    estudio_seleccionado = seleccionar_estudio()
    if estudio_seleccionado is None:
        return

    try:
        idx = int(input(f"Ingrese el índice del corte (0 a {estudio_seleccionado.forma[0]-1}): "))
        if not (0 <= idx < estudio_seleccionado.forma[0]):
            print("[Error] Índice de corte fuera de rango.")
            return
            
        tam_kernel = int(input("Ingrese el tamaño del kernel (ej: 3, 5, 7): "))

        print("\n--- Tipos de Operación Morfológica ---")
        print("1. Erosión (cv2.erode)")
        print("2. Dilatación (cv2.dilate)")
        print("3. Apertura (cv2.morphologyEx con cv2.MORPH_OPEN)")
        print("4. Cierre (cv2.morphologyEx con cv2.MORPH_CLOSE)")
        
        opcion = input("Seleccione la operación (1-4): ")

        # Mapeamos la entrada a la función/parámetros de OpenCV
        operacion_cv = None
        if opcion == "1":
            operacion_cv = cv2.erode
        elif opcion == "2":
            operacion_cv = cv2.dilate
        elif opcion == "3":
            # Las operaciones 'Open' y 'Close' usan una función base diferente
            kernel = np.ones((tam_kernel, tam_kernel), np.uint8)
            corte = estudio_seleccionado.imagen_3d[idx, :, :]
            corte_uint8 = estudio_seleccionado._normalizar_a_uint8(corte)
            
            img_resultante = cv2.morphologyEx(corte_uint8, cv2.MORPH_OPEN, kernel)
            
            plt.imshow(img_resultante, cmap='gray')
            plt.title("Morfología: Apertura")
            plt.show()
            cv2.imwrite("morf_apertura.png", img_resultante)
            print("Imagen guardada como morf_apertura.png")
            return # Salimos de la función
            
        elif opcion == "4":
            kernel = np.ones((tam_kernel, tam_kernel), np.uint8)
            corte = estudio_seleccionado.imagen_3d[idx, :, :]
            corte_uint8 = estudio_seleccionado._normalizar_a_uint8(corte)
            
            img_resultante = cv2.morphologyEx(corte_uint8, cv2.MORPH_CLOSE, kernel)
            
            plt.imshow(img_resultante, cmap='gray')
            plt.title("Morfología: Cierre")
            plt.show()
            cv2.imwrite("morf_cierre.png", img_resultante)
            print("Imagen guardada como morf_cierre.png")
            return # Salimos de la función
        else:
            print("[Error] Opción no válida.")
            return
        
        # Llamamos al método de la clase (solo para Erosión y Dilatación)
        estudio_seleccionado.transformacion_morfologica(idx, tam_kernel, operacion_cv)

    except ValueError:
        print("[Error] Entrada inválida. Debe ser un número entero.")

#---  Bucle Principal de veritas ---
def main():
    """
    Función principal que ejecuta el bucle del menú.
    """
    while True:
        mostrar_menu()
        opcion = input("Seleccione una opción: ")

        if opcion == '1':
            cargar_nuevo_estudio()
        
        elif opcion == '2':
            mostrar_cortes_3d()
            
        elif opcion == '3':
            aplicar_zoom()

        elif opcion == '4':
            aplicar_segmentacion()

        elif opcion == '5':
            aplicar_morfologia()

        elif opcion == '6':
            convertir_a_nifti()
            
        elif opcion == '7':
            exportar_metadata_csv()

        elif opcion == '0':
            print("Saliendo del programa...")
            break
            
        else:
            print("[Error] Opción no válida. Por favor, intente de nuevo.")

if __name__ == "__main__":
    main()