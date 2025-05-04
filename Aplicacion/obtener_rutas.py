import os
import sys

def resource_path(relative_path):
    """
    Obtiene la ruta absoluta al recurso, funcionando tanto para desarrollo como para el ejecutable empaquetado.
    """
    try:
        # PyInstaller crea una carpeta temporal _MEIPASS al ejecutar el archivo empaquetado
        base_path = sys._MEIPASS
    except AttributeError:
        # Estamos ejecutando el script directamente (no empaquetado)
        base_path = os.path.abspath(".")
        # Ajuste para desarrollo
        base_path = os.path.dirname(base_path)
    return os.path.join(base_path, relative_path)
