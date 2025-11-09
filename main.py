# ---  Importación de Librerías ---
import os
import cv2  # Necesario para las banderas de binarización y morfología
import pandas as pd
import matplotlib.pyplot as plt
from clases_dicom import DicomManager, Estudiolmaginologico, SistemaGestionDICOM
import numpy as np
from scipy.ndimage import zoom

# Punto de Entrada del Programa
if __name__ == "__main__":
    sistema = SistemaGestionDICOM()
    sistema.run()