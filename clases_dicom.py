# ---  Importación de Librerías ---
import os  # Para interactuar con el sistema operativo (leer carpetas)
import pydicom  # Para leer los archivos DICOM
import numpy as np  # Para el manejo de arreglos 
import cv2  # OpenCV para procesamiento de imagen
import matplotlib.pyplot as plt  # Para mostrar las imágenes
import nibabel as nib  # Para crear y guardar archivos NIFTI
from datetime import datetime  # Para calcular la diferencia de tiempo]
from scipy.ndimage import zoom
import pandas as pd


class DicomManager:
    """
    Esta clase se encarga de gestionar la carga y reconstrucción
    de una serie de archivos DICOM desde una carpeta.
    """
    def __init__(self, carpeta_path):
        self.carpeta_path = carpeta_path
        self.slices_dicom = []  # Lista para guardar los slices leídos
        self.volumen_3d = None  # Matriz 3D que se reconstruirá
        self.metadatos_primer_slice = None # Guardamos la metadata del primer slice
        self._cargar_archivos_dicom()
        self._reconstruir_volumen_3d()

        # Llamamos a los métodos de carga y reconstrucción al inicializar
        self._cargar_archivos_dicom()
        self._reconstruir_volumen_3d()

    def _cargar_archivos_dicom(self):
        """
        Método privado para cargar y ordenar los archivos DICOm
        El orden es crucial para una correcta reconstrucción 3d
        """
        print(f"Cargando archivos desde: {self.carpeta_path}")
        archivos = [f for f in os.listdir(self.carpeta_path) if f.endswith('.dcm')]
        
        if not archivos:
            print("Advertencia: No se encontraron archivos .dcm en la carpeta.")
            return

        # Leemos todos los archivos DICOM
        for nombre_archivo in archivos:
            ruta_completa = os.path.join(self.carpeta_path, nombre_archivo)
            try:
                ds = pydicom.dcmread(ruta_completa)
                self.slices_dicom.append(ds)
            except Exception as e:
                print(f"No se pudo leer el archivo {nombre_archivo}: {e}")

        # Ordenamos los slices. Usamos 'InstanceNumber' (Número de instancia)
        # que generalmente indica el orden correcto del corte.
        self.slices_dicom.sort(key=lambda x: int(x.InstanceNumber))
        
        if self.slices_dicom:
            # Guardamos los metadatos del primer slice para fácil acceso
            self.metadatos_primer_slice = self.slices_dicom[0]

    def _reconstruir_volumen_3d(self):
        """
        Método privado para apilar los slices 2D en un solo volumen 3D (matriz NumPy).
        """
        if not self.slices_dicom:
            print("Error: No hay slices DICOM para reconstruir.")
            return

        # Obtenemos las dimensiones del primer slice
        filas = int(self.metadatos_primer_slice.Rows)
        columnas = int(self.metadatos_primer_slice.Columns)
        num_slices = len(self.slices_dicom)

        # Creamos una matriz 3D vacía con las dimensiones correctas
        # Usamos np.float64 para permitir la conversión a Hounsfield (que puede tener decimales)
        self.volumen_3d = np.zeros((num_slices, filas, columnas), dtype=np.float64)

        # Llenamos la matriz 3D con los datos de píxeles de cada slice
        for i, slice_dicom in enumerate(self.slices_dicom):
            img_2d = slice_dicom.pixel_array.astype(np.float64)
            
            # Aplicamos la conversión a Hounsfield (o unidades reales)
            # Esto es fundamental para que los valores de píxeles sean médicamente correctos
            if 'RescaleSlope' in slice_dicom and 'RescaleIntercept' in slice_dicom:
                slope = float(slice_dicom.RescaleSlope)
                intercept = float(slice_dicom.RescaleIntercept)
                img_2d = (img_2d * slope) + intercept
            
            self.volumen_3d[i, :, :] = img_2d
        
        print("Volumen 3D reconstruido.")

    def obtener_cortes_principales(self):
        """
        Obtiene los 3 cortes (transversal, sagital, coronal) del volumen 3D.
        """
        if self.volumen_3d is None:
            print("Error: El volumen 3D no ha sido reconstruido.")
            return None, None, None

        # Obtenemos los índices del centro del volumen
        idx_transversal = self.volumen_3d.shape[0] // 2
        idx_coronal = self.volumen_3d.shape[1] // 2
        idx_sagital = self.volumen_3d.shape[2] // 2

        # Extraemos los cortes
        # Corte Transversal (Axial): Es un corte directo del stack
        corte_transversal = self.volumen_3d[idx_transversal, :, :]
        
        # Corte Coronal "de frente"
        corte_coronal = self.volumen_3d[:, idx_coronal, :] #Z, X
        
        # Corte Sagital "de lado"
        corte_sagital = self.volumen_3d[:, :, idx_sagital] #Z, Y

        return corte_transversal, corte_sagital, corte_coronal

    def convertir_a_nifti(self, ruta_archivo_salida):
        """
        Convierte el volumen 3D reconstruido a formato NIFTI.
        """
        if self.volumen_3d is None:
            print("Error: No hay volumen 3D para convertir.")
            return

        # NIFTI requiere una matriz 'affine' que describe la orientación y
        # espaciado. Crear una matriz affine completa es complejo.
        # Creamos una matriz 'affine' simple (identidad)
        # que solo guarda los datos del voxel.
        
        # Intentamos obtener el espaciado de los píxeles (mm)
        try:
            pixel_spacing = self.metadatos_primer_slice.PixelSpacing
            slice_thickness = float(self.metadatos_primer_slice.SliceThickness)
            
            affine_matrix = np.array([
                [-float(pixel_spacing[1]), 0, 0, 0],
                [0, -float(pixel_spacing[0]), 0, 0],
                [0, 0, slice_thickness, 0],
                [0, 0, 0, 1]
            ])
        except Exception:
            print("No se pudo obtener metadata de espaciado. Usando matriz identidad.")
            affine_matrix = np.eye(4) # Matriz identidad básica

        # Creamos el objeto imagen NIFTI
        nifti_img = nib.Nifti1Image(self.volumen_3d, affine_matrix)
        
        # Guardamos el archivo
        nib.save(nifti_img, ruta_archivo_salida)
        print(f"Archivo guardado en formato NIFTI en: {ruta_archivo_salida}")



    
class Estudiolmaginologico:
    """
    Clase que representa un estudio imaginológico
    Almacena metadatos clave y la matriz 3D
    Provee métodos para procesamiento de imagen con OpenCV
    """
    
    def __init__(self, manager_dicom: DicomManager):
        """
        Constructor. Recibe un objeto DicomManager que ya ha cargado
        y reconstruido los datos.
        """
        if manager_dicom.metadatos_primer_slice is None:
            raise ValueError("El DicomManager no tiene metadatos cargados.")

        # Guardamos el manager para acceder a metadatos (ej. PixelSpacing)
        self.manager_dicom = manager_dicom
        
        # Extraemos los metadatos requeridos 
        meta = manager_dicom.metadatos_primer_slice
        self.study_date = meta.get('StudyDate', 'N/A')
        self.study_time = meta.get('StudyTime', '000000')
        self.study_modality = meta.get('Modality', 'N/A')
        self.study_description = meta.get('StudyDescription', 'N/A')
        self.series_time = meta.get('SeriesTime', '000000')

        # Atributos de la imagen 3D 
        self.imagen_3d = manager_dicom.volumen_3d
        self.forma = self.imagen_3d.shape
        
        # Cálculo del tiempo de duración 
        self.tiempo_duracion_estudio = self._calcular_diferencia_tiempo(
            self.study_time, self.series_time
        )

    def _calcular_diferencia_tiempo(self, t_inicio_str, t_fin_str):
        """
        Método privado para calcular la diferencia entre dos
        marcas de tiempo en formato DICOM (HHMMSS.ffffff).
        """
        try:
            # El formato DICOM puede tener o no fracciones de segundo.
            # Lo limpiamos quedándonos solo con HHMMSS.
            t_inicio_limpio = t_inicio_str.split('.')[0]
            t_fin_limpio = t_fin_str.split('.')[0]
            
            # Formato de hora
            formato_hora = "%H%M%S"
            
            # Convertimos los strings a objetos 'datetime'
            t1 = datetime.strptime(t_inicio_limpio, formato_hora)
            t2 = datetime.strptime(t_fin_limpio, formato_hora)
            
            diferencia = t2 - t1
            return str(diferencia)
            
        except ValueError:
            # Maneja casos donde el formato de hora es inválido o está vacío
            return "Error en cálculo de tiempo"

    def _normalizar_a_uint8(self, img):
        """
        Método privado para normalizar una imagen (corte)
        y convertirla a uint8 (0-255), como requiere OpenCV
        """
        # img - min(img)
        img_norm = img - np.min(img)
        
        # max(img) - min(img)
        denominador = np.max(img) - np.min(img)
        
        # Evitamos división por cero si la imagen es plana (todos los píxeles iguales)
        if denominador == 0:
            return np.zeros(img.shape, dtype=np.uint8)
            
        # (img - min) / (max - min)
        img_norm = img_norm / denominador
        
        # Multiplicamos por 255
        img_norm = img_norm * 255.0
        
        # Convertimos a tipo uint8 (entero sin signo de 8 bits)
        img_uint8 = img_norm.astype(np.uint8)
        
        return img_uint8


    def metodo_zoom(self, indice_corte, x, y, w, h, nombre_archivo_recorte, nombre_archivo_plot):
        """
        MODIFICADO: Se ajusta la posición del texto de las dimensiones
        para que aparezca DENTRO del cuadro, cumpliendo el requisito.
        """
        #  Obtener y normalizar
        corte_original = self.imagen_3d[indice_corte, :, :]
        corte_norm_uint8 = self._normalizar_a_uint8(corte_original)
        
        # Convertir a BGR para dibujar a color
        corte_bgr = cv2.cvtColor(corte_norm_uint8, cv2.COLOR_GRAY2BGR)

        # 3. Dibujar el cuadro (ROI)
        pt1 = (x, y) # Coordenada superior izquierda
        pt2 = (x + w, y + h) # Coordenada inferior derecha
        color_verde = (0, 255, 0)
        grosor = 2
        cv2.rectangle(corte_bgr, pt1, pt2, color_verde, grosor)

        #  Añadir texto con dimensiones en milímetros (mm)
        try:
            meta = self.manager_dicom.slices_dicom[indice_corte]
            pixel_spacing = meta.PixelSpacing # Es [spacing_filas (Y), spacing_cols (X)]
            
            dim_mm_w = w * float(pixel_spacing[1]) # Ancho (columnas)
            dim_mm_h = h * float(pixel_spacing[0]) # Alto (filas)
            
            texto = f"{dim_mm_w:.1f} x {dim_mm_h:.1f} mm"
        
            pos_texto = (x + 5, y + 20) 
            
            cv2.putText(corte_bgr, texto, pos_texto, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_verde, 1, cv2.LINE_AA)
        except Exception as e:
            print(f"No se pudo añadir texto de dimensiones: {e}")

        #Recortar la región de la imagen normalizada (uint8)
        region_recortada = corte_norm_uint8[y:y+h, x:x+w]

        # Redimensionar (resize) el recorte
        # (Usamos w*2 y h*2 como ejemplo de redimensión)
        recorte_redimensionado = cv2.resize(region_recortada, (w * 2, h * 2), 
                                          interpolation=cv2.INTER_LINEAR)

        # Mostrar en dos subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(cv2.cvtColor(corte_bgr, cv2.COLOR_BGR2RGB)) # Convertir a RGB para Matplotlib
        ax1.set_title("Original con ROI y Texto")
        ax1.axis('off')
        
        ax2.imshow(recorte_redimensionado, cmap='gray')
        ax2.set_title("Región Recortada y Redimensionada (Zoom)")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

        # Guardar la imagen recortada y el plot
        cv2.imwrite(nombre_archivo_recorte, recorte_redimensionado)
        fig.savefig(nombre_archivo_plot)
        print(f"Imagen recortada guardada como: {nombre_archivo_recorte}")
        print(f"Plot de comparación guardado como: {nombre_archivo_plot}")

    def funcion_segmentacion(self, indice_corte, tipo_binarizacion_cv2, umbral, nombre_archivo_salida):
        """
        Aplica binarización (segmentación simple) a un corte.

        """
        # Obtenemos y normalizamos el corte
        corte = self.imagen_3d[indice_corte, :, :]
        corte_uint8 = self._normalizar_a_uint8(corte)

        # Aplicamos el umbral (binarización)
        # cv2.threshold devuelve el valor del umbral usado y la imagen binarizada
        # Usamos un umbral fijo (ej. 127) o podríamos usar cv2.THRESH_OTSU
        valor_umbral, img_binarizada = cv2.threshold(
            corte_uint8, 
            umbral, # Valor del umbral
            255,    # Valor máximo (para píxeles que superan el umbral)
            tipo_binarizacion_cv2 # Tipo de binarización
        )

        # Mostramos la imagen resultante
        plt.figure(figsize=(8, 8))
        plt.imshow(img_binarizada, cmap='gray')
        plt.title(f"Resultado de Binarización (Tipo: {tipo_binarizacion_cv2})")
        plt.axis('off')
        plt.show()

        #guardamos
        
        cv2.imwrite(nombre_archivo_salida, img_binarizada)
        fig.savefig(f"plot_{nombre_archivo_salida}")
        print(f"Imagen binarizada guardada como: {nombre_archivo_salida}")
        print(f"Plot de binarización guardado como: plot_{nombre_archivo_salida}")

        return img_binarizada

    def transformacion_morfologica(self, indice_corte, tamano_kernel, operacion_cv2_flag, nombre_archivo_salida):
        """
        Aplica una transformación morfológica (Erosión, Dilatación, etc.)
        'operacion_cv2' debe ser una función de OpenCV,
        ej: cv2.erode, cv2.dilate
        """
        # Obtenemos y normalizamos el corte 
        corte_uint8 = self._normalizar_a_uint8(self.imagen_3d[indice_corte, :, :])
        #el fokin kernel
        kernel = np.ones((tamano_kernel, tamano_kernel), np.uint8)


        # Aplicamos la operación morfológica
        # La 'operacion_cv2' se pasa como un argumento de función
        # ej: operacion_cv2 = cv2.erode
        try:
            img_resultante = cv2.morphologyEx(corte_uint8, operacion_cv2_flag, kernel)
            
            # Mostramos y guardamos la imagen resultante 
            plt.figure(figsize=(8, 8))
            plt.imshow(img_resultante, cmap='gray')
            plt.title(f"Transformación Morfológica (Kernel: {tamano_kernel}x{tamano_kernel})")
            plt.axis('off')
            plt.show()
            
            # Guardamos la imagen
            cv2.imwrite(nombre_archivo_salida, img_resultante)
            fig.savefig(f"plot_{nombre_archivo_salida}")
            print(f"Imagen morfológica guardada como: {nombre_archivo_salida}")
            print(f"Plot morfológico guardado como: plot_{nombre_archivo_salida}")

        except Exception as e:
            print(f"Error al aplicar la operación morfológica: {e}")



class SistemaGestionDICOM:
    """
    Esta clase encapsula toda la lógica de la aplicación (el menú)
    y gestiona el estado (los estudios cargados).
    """
    
    def __init__(self):
        """
        Constructor. Inicializa el diccionario que almacenará
        los objetos Estudiolmaginologico.
        """
        self.estudios_cargados = {} # Reemplaza la variable global

    def mostrar_menu(self):
        print("\n" + "="*40)
        print("   🏥 Sistema de Gestión DICOM (POO) 🏥")
        print("="*40)
        print("1. Cargar nueva carpeta DICOM (Crear Estudio)")
        print("2. Mostrar cortes 3D (Transversal, Sagital, Coronal) de un estudio")
        print("3. Aplicar ZOOM (Recorte y Redimensión) a un corte")
        print("4. Aplicar Segmentación (Binarización) a un corte")
        print("5. Aplicar Transformación Morfológica a un corte")
        print("6. Convertir estudio DICOM a NIFTI")
        print("7. Exportar metadatos de estudios cargados a CSV")
        print("0. Salir")
        print("-"*40)

    def seleccionar_estudio(self):
        """
        Método auxiliar para mostrar los estudios cargados y
        permitir al usuario seleccionar uno para trabajar.
        """
        if not self.estudios_cargados:
            print("\n[Error] No hay estudios cargados. Por favor, cargue un estudio primero (Opción 1).")
            return None

        print("\n--- 📚 Estudios Cargados ---")
        lista_estudios = list(self.estudios_cargados.keys())
        
        for i, ruta in enumerate(lista_estudios):
            print(f"  [{i+1}] {ruta}")
        
        try:
            opcion = int(input("Seleccione el número del estudio: "))
            if 1 <= opcion <= len(lista_estudios):
                ruta_seleccionada = lista_estudios[opcion - 1]
                return self.estudios_cargados[ruta_seleccionada]
            else:
                print("[Error] Selección fuera de rango.")
                return None
        except ValueError:
            print("[Error] Entrada inválida. Debe ser un número.")
            return None

    def cargar_nuevo_estudio(self):
        """
        Opción 1: Pide una ruta, crea los objetos DicomManager y
        Estudiolmaginologico, y los almacena en 'self.estudios_cargados'.
        """
        ruta_carpeta = input("Ingrese la ruta de la carpeta con los archivos DICOM: ")
        
        if not os.path.isdir(ruta_carpeta):
            print(f"[Error] La ruta '{ruta_carpeta}' no es una carpeta válida.")
            return

        print("Cargando y procesando... por favor espere.")
        try:
            manager = DicomManager(ruta_carpeta)
            if manager.volumen_3d is None:
                print("[Error] No se pudo reconstruir el volumen 3D. Verifique los archivos.")
                return
            
            estudio = Estudiolmaginologico(manager)
            self.estudios_cargados[ruta_carpeta] = estudio # Almacena en el atributo de clase
            
            print(f"\n¡Estudio cargado exitosamente!")
            print(f"  - Modalidad: {estudio.study_modality}")
            print(f"  - Descripción: {estudio.study_description}")
            print(f"  - Forma 3D: {estudio.forma}")
            
        except Exception as e:
            print(f"[Error] Ocurrió un problema al cargar el estudio: {e}")

    def mostrar_cortes_3d(self):
        """
        Opción 2: Muestra y guarda los 3 cortes principales (T, S, C).
        """
        estudio_seleccionado = self.seleccionar_estudio()
        if estudio_seleccionado is None:
            return

        # 1. Obtener metadatos de espaciado
        try:
            meta = estudio_seleccionado.manager_dicom.metadatos_primer_slice
            pixel_spacing = meta.PixelSpacing
            slice_thickness = float(meta.SliceThickness) 
            
            aspecto_trans = float(pixel_spacing[0]) / float(pixel_spacing[1]) 
            aspecto_sag_T = float(pixel_spacing[0]) / slice_thickness
            aspecto_cor = slice_thickness / float(pixel_spacing[1])
        
        except Exception as e:
            print(f"Advertencia: No se pudo leer el espaciado. Se usará 'auto'. ({e})")
            aspecto_trans = 'auto'; aspecto_sag_T = 'auto'; aspecto_cor = 'auto'
        
        # 2. Obtenemos los cortes
        trans, sag, cor = estudio_seleccionado.manager_dicom.obtener_cortes_principales()

        if trans is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            ax1.imshow(trans, cmap='gray', aspect=aspecto_trans, interpolation='bilinear')
            ax1.set_title("Corte Transversal (Axial)"); ax1.axis('off')
            
            ax2.imshow(sag.T, cmap='gray', aspect=aspecto_sag_T, interpolation='bilinear') 
            ax2.set_title("Corte Sagital"); ax2.axis('off')

            ax3.imshow(cor, cmap='gray', aspect=aspecto_cor, interpolation='bilinear') 
            ax3.set_title("Corte Coronal"); ax3.axis('off')
            
            plt.tight_layout()
            plt.show()

            # --- NUEVA FUNCIÓN DE GUARDADO ---
            try:
                nombre_salida = input("Ingrese nombre para guardar el plot (ej: cortes_3d.png): ")
                fig.savefig(nombre_salida)
                print(f"Plot de cortes 3D guardado como: {nombre_salida}")
            except Exception as e:
                print(f"No se pudo guardar el plot: {e}")

    def aplicar_zoom(self):
        """
        Opción 3: Pide parámetros y llama al metodo_zoom.
        """
        estudio_seleccionado = self.seleccionar_estudio()
        if estudio_seleccionado is None:
            return

        try:
            print(f"El volumen tiene {estudio_seleccionado.forma[0]} cortes (índice 0 a {estudio_seleccionado.forma[0]-1})")
            idx = int(input("Ingrese el índice del corte a procesar: "))
            if not (0 <= idx < estudio_seleccionado.forma[0]):
                print("[Error] Índice de corte fuera de rango."); return
                
            print(f"El corte tiene dimensiones: {estudio_seleccionado.forma[1]} Alto x {estudio_seleccionado.forma[2]} Ancho")
            x = int(input("Ingrese coordenada X de inicio: ")); y = int(input("Ingrese coordenada Y de inicio: "))
            w = int(input("Ingrese Ancho (W) del recorte: ")); h = int(input("Ingrese Alto (H) del recorte: "))
            
            # --- NUEVA FUNCIÓN DE GUARDADO ---
            nombre_recorte = input("Nombre del archivo para la imagen recortada (ej: zoom.png): ")
            nombre_plot = input("Nombre del archivo para el plot de comparación (ej: comparacion_zoom.png): ")

            estudio_seleccionado.metodo_zoom(idx, x, y, w, h, nombre_recorte, nombre_plot)

        except ValueError:
            print("[Error] Entrada inválida. Todos los valores deben ser números enteros.")
        except Exception as e:
            print(f"[Error] No se pudo aplicar el zoom: {e}")

    def aplicar_segmentacion(self):
        """
        Opción 4: Pide tipo de binarización y llama a funcion_segmentacion.
        """
        estudio_seleccionado = self.seleccionar_estudio()
        if estudio_seleccionado is None:
            return
            
        try:
            idx = int(input(f"Ingrese el índice del corte (0 a {estudio_seleccionado.forma[0]-1}): "))
            if not (0 <= idx < estudio_seleccionado.forma[0]):
                print("[Error] Índice de corte fuera de rango."); return

            print("\n--- Tipos de Binarización ---")
            print("1. Binario (cv2.THRESH_BINARY)"); print("2. Binario Invertido (cv2.THRESH_BINARY_INV)")
            print("3. Truncado (cv2.THRESH_TRUNC)"); print("4. A Cero (cv2.THRESH_TOZERO)")
            print("5. A Cero Invertido (cv2.THRESH_TOZERO_INV)")
            
            opcion = input("Seleccione el tipo de binarización (1-5): ")

            mapeo_binarizacion = {
                "1": cv2.THRESH_BINARY, "2": cv2.THRESH_BINARY_INV, "3": cv2.THRESH_TRUNC,
                "4": cv2.THRESH_TOZERO, "5": cv2.THRESH_TOZERO_INV
            }
            bandera_seleccionada = mapeo_binarizacion.get(opcion)
            
            if bandera_seleccionada is None:
                print("[Error] Opción de binarización no válida."); return

            umbral = int(input("Ingrese el valor del umbral (0-255, ej: 127): "))
            
            # --- NUEVA FUNCIÓN DE GUARDADO ---
            nombre_salida = input("Nombre del archivo para guardar la imagen segmentada (ej: binaria.png): ")
            
            estudio_seleccionado.funcion_segmentacion(idx, bandera_seleccionada, umbral, nombre_salida)
            
        except ValueError:
            print("[Error] Entrada inválida. Debe ser un número entero.")

    def aplicar_morfologia(self):
        """
        Opción 5: Pide tamaño de kernel y operación, y llama a la
        función morfológica correspondiente.
        """
        estudio_seleccionado = self.seleccionar_estudio()
        if estudio_seleccionado is None:
            return

        try:
            idx = int(input(f"Ingrese el índice del corte (0 a {estudio_seleccionado.forma[0]-1}): "))
            if not (0 <= idx < estudio_seleccionado.forma[0]):
                print("[Error] Índice de corte fuera de rango."); return
                
            tam_kernel = int(input("Ingrese el tamaño del kernel (ej: 3, 5, 7): "))

            print("\n--- Tipos de Operación Morfológica ---")
            print("1. Erosión"); print("2. Dilatación"); print("3. Apertura"); print("4. Cierre")
            
            opcion = input("Seleccione la operación (1-4): ")
            
            # --- NUEVA FUNCIÓN DE GUARDADO ---
            nombre_salida = input("Nombre del archivo para guardar la imagen morfológica (ej: erosion.png): ")
            
            if opcion == "1":
                estudio_seleccionado.transformacion_morfologica(idx, tam_kernel, cv2.erode, nombre_salida)
            elif opcion == "2":
                estudio_seleccionado.transformacion_morfologica(idx, tam_kernel, cv2.dilate, nombre_salida)
            elif opcion == "3":
                estudio_seleccionado.transformacion_morfologica_ex(idx, tam_kernel, cv2.MORPH_OPEN, nombre_salida)
            elif opcion == "4":
                estudio_seleccionado.transformacion_morfologica_ex(idx, tam_kernel, cv2.MORPH_CLOSE, nombre_salida)
            else:
                print("[Error] Opción no válida.")
                
        except ValueError:
            print("[Error] Entrada inválida. Debe ser un número entero.")
            
    def convertir_a_nifti(self):
        """
        Opción 6: Llama al método de conversión a NIFTI del DicomManager.
        """
        estudio_seleccionado = self.seleccionar_estudio()
        if estudio_seleccionado is None:
            return
            
        nombre_salida = input("Ingrese el nombre del archivo de salida (ej: mi_estudio.nii.gz): ")
        if not (nombre_salida.endswith(".nii") or nombre_salida.endswith(".nii.gz")):
            print("Advertencia: Se recomienda usar extensión .nii o .nii.gz")
            
        try:
            estudio_seleccionado.manager_dicom.convertir_a_nifti(nombre_salida)
        except Exception as e:
            print(f"[Error] No se pudo convertir a NIFTI: {e}")

    def exportar_metadata_csv(self):
        """
        Opción 7: Extrae metadatos de TODOS los estudios cargados
        y los guarda en un archivo CSV.
        """
        if not self.estudios_cargados:
            print("\n[Error] No hay estudios cargados para exportar.")
            return

        lista_de_estudios_data = []
        for ruta, estudio in self.estudios_cargados.items():
            datos_fila = {
                "Ruta Carpeta": ruta, "Modalidad": estudio.study_modality,
                "Fecha Estudio": estudio.study_date, "Hora Estudio": estudio.study_time,
                "Hora Serie": estudio.series_time, "Duracion Estudio": estudio.tiempo_duracion_estudio,
                "Descripcion": estudio.study_description, "Forma 3D": str(estudio.forma)
            }
            lista_de_estudios_data.append(datos_fila)

        df = pd.DataFrame(lista_de_estudios_data)
        nombre_csv = "metadata_estudios.csv"
        df.to_csv(nombre_csv, index=False)
        
        print(f"\n¡Metadatos exportados exitosamente a '{nombre_csv}'!")
        print(df)

    def run(self):
        """
        Función principal que ejecuta el bucle del menú.
        """
        while True:
            self.mostrar_menu()
            opcion = input("Seleccione una opción: ")

            if opcion == '1':
                self.cargar_nuevo_estudio()
            elif opcion == '2':
                self.mostrar_cortes_3d()
            elif opcion == '3':
                self.aplicar_zoom()
            elif opcion == '4':
                self.aplicar_segmentacion()
            elif opcion == '5':
                self.aplicar_morfologia()
            elif opcion == '6':
                self.convertir_a_nifti()
            elif opcion == '7':
                self.exportar_metadata_csv()
            elif opcion == '0':
                print("Saliendo del programa...")
                break
            else:
                print("[Error] Opción no válida. Por favor, intente de nuevo.")