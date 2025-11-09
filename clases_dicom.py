# ---  Importación de Librerías ---
import os  # Para interactuar con el sistema operativo (leer carpetas)
import pydicom  # Para leer los archivos DICOM
import numpy as np  # Para el manejo de arreglos 
import cv2  # OpenCV para procesamiento de imagen
import matplotlib.pyplot as plt  # Para mostrar las imágenes
import nibabel as nib  # Para crear y guardar archivos NIFTI
from datetime import datetime  # Para calcular la diferencia de tiempo]

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
        corte_coronal = self.volumen_3d[:, idx_coronal, :]
        
        # Corte Sagital "de lado"
        corte_sagital = self.volumen_3d[:, :, idx_sagital]

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

    def metodo_zoom(self, indice_corte, x, y, w, h, nombre_archivo_salida):
        """
        Recorta, redimensiona y dibuja un cuadro sobre un corte.
        """

        # Obtenemos el corte original (slice) desde el volumen 3D
        corte_original = self.imagen_3d[indice_corte, :, :]

        # Normalizamos a uint8 
        corte_norm_uint8 = self._normalizar_a_uint8(corte_original)

        # Convertimos a BGR para poder dibujar en color 
        # (OpenCV espera 3 canales de color para dibujar en color)
        corte_bgr = cv2.cvtColor(corte_norm_uint8, cv2.COLOR_GRAY2BGR)

        # Dibujar el cuadro (ROI - Región de Interés) 
        pt1 = (x, y) # Coordenada superior izquierda
        pt2 = (x + w, y + h) # Coordenada inferior derecha
        color_verde = (0, 255, 0)
        grosor = 2
        cv2.rectangle(corte_bgr, pt1, pt2, color_verde, grosor)

        # Añadir texto con dimensiones en milímetros (mm) 
        try:
            # Obtenemos el espaciado de píxeles (mm)
            meta = self.manager_dicom.slices_dicom[indice_corte]
            pixel_spacing = meta.PixelSpacing # Es una lista [spacing_filas, spacing_cols]
            
            dim_mm_w = w * float(pixel_spacing[1]) # Ancho (columnas)
            dim_mm_h = h * float(pixel_spacing[0]) # Alto (filas)
            
            texto = f"{dim_mm_w:.1f} x {dim_mm_h:.1f} mm"
            
            # Coordenadas para el texto (un poco arriba del cuadro)
            pos_texto = (x, y - 10)
            cv2.putText(corte_bgr, texto, pos_texto, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_verde, 1)
        except Exception as e:
            print(f"No se pudo añadir texto de dimensiones: {e}")

        # Recortar la región de la imagen normalizada (no de la BGR)
        region_recortada = corte_norm_uint8[y:y+h, x:x+w]

        # Redimensionar (resize) el recorte 
        # Vamos a duplicar su tamaño como ejemplo de redimensionamiento
        nuevo_ancho = w * 2
        nueva_altura = h * 2
        recorte_redimensionado = cv2.resize(region_recortada, (nuevo_ancho, nueva_altura), 
                                          interpolation=cv2.INTER_LINEAR)

        # Mostrar en dos subplots 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(corte_bgr)
        ax1.set_title("Imagen Original con Cuadro (ROI)")
        ax1.axis('off')
        
        ax2.imshow(recorte_redimensionado, cmap='gray')
        ax2.set_title("Región Recortada y Redimensionada (Zoom)")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

        # Guardar la imagen recortada
        cv2.imwrite(nombre_archivo_salida, recorte_redimensionado)
        print(f"Imagen recortada guardada como: {nombre_archivo_salida}")

    def funcion_segmentacion(self, indice_corte, tipo_binarizacion_cv2, umbral=127):
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

        return img_binarizada

    def transformacion_morfologica(self, indice_corte, tamano_kernel, operacion_cv2):
        """
        Aplica una transformación morfológica (Erosión, Dilatación, etc.)
        'operacion_cv2' debe ser una función de OpenCV,
        ej: cv2.erode, cv2.dilate
        """
        # Obtenemos y normalizamos el corte 
        corte = self.imagen_3d[indice_corte, :, :]
        corte_uint8 = self._normalizar_a_uint8(corte)

        # Creamos el 'kernel' 
        # Es una matriz cuadrada de 'unos' del tamaño dado
        kernel = np.ones((tamano_kernel, tamano_kernel), np.uint8)

        # Aplicamos la operación morfológica
        # La 'operacion_cv2' se pasa como un argumento de función
        # ej: operacion_cv2 = cv2.erode
        try:
            img_resultante = operacion_cv2(corte_uint8, kernel, iterations=1)
            
            # Mostramos y guardamos la imagen resultante 
            plt.figure(figsize=(8, 8))
            plt.imshow(img_resultante, cmap='gray')
            plt.title(f"Transformación Morfológica (Kernel: {tamano_kernel}x{tamano_kernel})")
            plt.axis('off')
            plt.show()
            
            # Guardamos la imagen
            nombre_archivo = f"morfologia_k{tamano_kernel}.png"
            cv2.imwrite(nombre_archivo, img_resultante)
            print(f"Imagen morfológica guardada como: {nombre_archivo}")

        except Exception as e:
            print(f"Error al aplicar la operación morfológica: {e}")