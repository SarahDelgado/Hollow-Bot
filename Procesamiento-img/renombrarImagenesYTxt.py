import os

# Código para renombrar imágenes y sus txt correspondientes

def renombrar_archivos(directorio, extension):
    """
    Renombra todos los archivos en el directorio especificado que tengan la extensión dada,
    agregando el sufijo '_v2' al nombre base de cada archivo.

    Args:
        directorio (str): Ruta del directorio que contiene los archivos a renombrar.
        extension (str): Extensión de los archivos que se desean renombrar (por ejemplo, ".json" o ".txt").
    """
    # Itera sobre todos los archivos de un directorio
    for archivo in os.listdir(directorio):
        # Comprueba que sea del tipo establecido por parámetro, es decir, que tenga al extensión deseada
        if archivo.endswith(extension):
            nombre_base, ext = os.path.splitext(archivo) # Extrae el nombre del archivo sin la extensión
            nuevo_nombre = f"{nombre_base}_v2{ext}" # Añade al nombre el sufijo _v2
            ruta_actual = os.path.join(directorio, archivo) # Extrae la ruta actual del archivo
            ruta_nueva = os.path.join(directorio, nuevo_nombre) # Crea la nueva ruta del archivo con el nuevo nombre
            os.rename(ruta_actual, ruta_nueva) # Renombra el archivo
            print(f"Renombrado: {archivo} → {nuevo_nombre}")


# Directorios donde están los archivos
directorio_png = "dataset/json"
directorio_txt = "txt"

# Renombrar archivos en ambos directorios
renombrar_archivos(directorio_png, ".json")
renombrar_archivos(directorio_txt, ".txt")
