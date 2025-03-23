import cv2
import numpy as np
import mss
import pygetwindow as gw
import os
import csv
from datetime import datetime
from collections import defaultdict
from math import dist  # Usamos la función dist para calcular la distancia entre puntos
from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO("best_modelo_boss_10h.pt")

# Nombres de las clases
class_names = ["cristal_boss", "boss_ray", "ray_from_above", "base"]

# Colores para cada clase (BGR)
class_colors = {
    "cristal_boss": (0, 255, 0),
    "boss_ray": (255, 0, 0),
    "ray_from_above": (0, 0, 255),
    "base": (0, 0, 0)
}

# Nombre de la ventana del juego
game_window_title = "Hollow Knight"

# Directorio para guardar imágenes
output_dir = "detections"
os.makedirs(output_dir, exist_ok=True)

# Archivo CSV para almacenar coordenadas
csv_file = "deteccionesrayo1.0.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "class_base", "x1_base", "y1_base", "x2_base", "y2_base",
                         "class_rayo", "x1_rayo", "y1_rayo", "x2_rayo", "y2_rayo"])  # Encabezados del CSV


def get_game_window():
    """ Obtiene la posición y dimensiones de la ventana del juego. """
    windows = gw.getWindowsWithTitle(game_window_title)
    if windows:
        game_window = windows[0]
        return {
            "top": game_window.top,
            "left": game_window.left,
            "width": game_window.width,
            "height": game_window.height
        }
    return None


def capture_screen(monitor):
    """ Captura la pantalla de la ventana del juego. """
    with mss.mss() as sct:
        img = sct.grab(monitor)
        frame = np.array(img)[:, :, :3]  # Quitar canal alfa
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convertir a formato OpenCV
        return frame


def obtener_distancia(p1, p2):
    """ Calcula la distancia Euclidiana entre dos puntos (centro de las cajas delimitadoras). """
    cx1, cy1 = (p1[0] + p1[2]) / 2, (p1[1] + p1[3]) / 2  # Coordenadas del centro de la base
    cx2, cy2 = (p2[0] + p2[2]) / 2, (p2[1] + p2[3]) / 2  # Coordenadas del centro del rayo
    return dist((cx1, cy1), (cx2, cy2))


while True:
    game_window = get_game_window()

    if game_window:
        frame = capture_screen(game_window)

        # Detectar entidades en la imagen
        results = model(frame)

        detecciones = defaultdict(list)  # Diccionario para agrupar bases y rayos en un mismo timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Listas para almacenar las bases y rayos detectados
        bases = []
        rayos = []

        # Recorrer todas las detecciones y almacenar las bases y rayos
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())
                class_id = int(cls)
                class_name = class_names[class_id]
                color = class_colors.get(class_name, (255, 255, 255))

                # Dibujar rectángulo y texto
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Guardar bases y rayos en las listas correspondientes
                if class_name == "base":
                    bases.append((x1, y1, x2, y2))
                elif class_name == "ray_from_above":
                    rayos.append((x1, y1, x2, y2))

        # Ahora emparejamos la base y el rayo más cercanos
        if bases and rayos:
            # Se asume que emparejamos la base más cercana a cada rayo (o viceversa)
            for base in bases:
                rayo_mas_cercano = min(rayos, key=lambda rayo: obtener_distancia(base, rayo))
                # Guardar la detección en el CSV
                with open(csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, "base", *base, "ray_from_above", *rayo_mas_cercano])

            # Guardar imagen del frame solo si hay al menos una base y un rayo
            img_filename = f"{output_dir}/{timestamp}.png"
            cv2.imwrite(img_filename, frame)

        # Mostrar la imagen con detección
        cv2.imshow("Detección de Entidades", frame)

    else:
        print("Ventana del juego no encontrada.")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
