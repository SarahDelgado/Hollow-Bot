import os

# Código para renombrar imágenes y sus txt correspondientes

# Directorios donde están los archivos
directorio_png = "dataset/frames"
directorio_txt = "txt"

# Función para renombrar archivos en un directorio
def renombrar_archivos(directorio, extension):
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

# Renombrar archivos en ambos directorios
renombrar_archivos(directorio_png, ".png")
renombrar_archivos(directorio_txt, ".txt")
