import os
import json
import glob

# Define las clases manualmente o carga desde un archivo
class_names = ["character"]  # Reemplaza con tus clases

# Directorios
labelme_path = "dataset/frames"  # Carpeta donde están los JSON
yolo_path = "txt"  # Carpeta destino de los TXT
os.makedirs(yolo_path, exist_ok=True)

# Procesar cada JSON
for json_file in glob.glob(os.path.join(labelme_path, "*.json")):
    with open(json_file, "r") as f:
        data = json.load(f)

    img_width = data["imageWidth"]
    img_height = data["imageHeight"]

    yolo_labels = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in class_names:
            continue  # Ignora etiquetas desconocidas

        class_id = class_names.index(label)
        points = shape["points"]

        # Extraer bounding box (xmin, ymin, xmax, ymax)
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)

        # Convertir a formato YOLO (normalizado)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

    # Guardar en archivo .txt con mismo nombre que imagen
    txt_filename = os.path.join(yolo_path, os.path.basename(json_file).replace(".json", ".txt"))
    with open(txt_filename, "w") as txt_file:
        txt_file.write("\n".join(yolo_labels))