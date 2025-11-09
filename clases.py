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
