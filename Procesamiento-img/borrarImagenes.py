import os

# Código para borrar las imágenes que no han sido etiquetadas y no van a ser utilizadas en el entrenamiento

# Ruta de la carpeta dónde se encuentran las imágenes y JSONs
carpeta = os.path.relpath("dataset/frames")

# Lee cada archivo de la carpeta
for archivo in os.listdir(carpeta):
    # Comprueba si tiene la extensión PNG de imagen, sino lo ignora
    if archivo.endswith(".png"):
        nombre_base = os.path.splitext(archivo)[0] # Extrae el nombre de la imagen sin la extensión
        json_path = os.path.join(carpeta, f"{nombre_base}.json") # Carga el JSON asociado a la imagen

        # Comprueba si se ha cargado algún JSON y si no es asi borra la imagen
        if not os.path.exists(json_path):
            os.remove(os.path.join(carpeta, archivo))
            print(f"Eliminado: {archivo}")
