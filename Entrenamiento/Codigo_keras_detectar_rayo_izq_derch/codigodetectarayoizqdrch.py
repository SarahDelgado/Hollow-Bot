import cv2
import numpy as np
import mss
import joblib
import pygetwindow as gw
from ultralytics import YOLO
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


scaler = joblib.load("scaler.pkl")
# Cargar el modelo de detección
model = YOLO("best_modelo_boss_10h.pt")

# Cargar el modelo de predicción de dirección del rayo
direction_model = tf.keras.models.load_model("modelo_rayo_derecha_izquierda.h5")

# Nombres de las clases
class_names = ["cristal_boss", "boss_ray", "ray_from_above", "base"]

# Colores para cada clase
class_colors = {
    "cristal_boss": (0, 255, 0),  # Verde
    "boss_ray": (255, 0, 0),  # Azul
    "ray_from_above": (0, 0, 255),  # Rojo
    "base": (0, 0, 0)  # Negro
}

# Nombre de la ventana del juego
game_window_title = "Hollow Knight"

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

while True:
    game_window = get_game_window()

    if game_window:  # Si la ventana está activa
        frame = capture_screen(game_window)

        # Detectar entidades en la imagen
        results = model(frame)

        bases = []
        rayos = []

        # Extraer detecciones
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box.tolist())  # Convertir coordenadas a enteros
                class_id = int(cls)  # Obtener ID de la clase
                class_name = class_names[class_id]  # Obtener el nombre de la clase
                color = class_colors.get(class_name, (255, 255, 255))  # Color por defecto: blanco

                # Guardar bases y rayos detectados
                if class_name == "base":
                    bases.append((x1, y1, x2, y2))
                elif class_name == "ray_from_above":
                    rayos.append((x1, y1, x2, y2))

                # Dibujar el rectángulo con el color de la clase
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Poner el nombre de la clase sobre el cuadro
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Si hay bases y rayos detectados, predecir dirección
        for rayo in rayos:
            x1_r, y1_r, x2_r, y2_r = rayo
            centro_rayo = (x1_r + x2_r) / 2

            for base in bases:
                x1_b, y1_b, x2_b, y2_b = base
                centro_base = (x1_b + x2_b) / 2

                # Preparar entrada para la red neuronal
                entrada = np.array([x1_b, y1_b, x2_b, y2_b, x1_r, y1_r, x2_r, y2_r]).reshape(1, -1)
                entrada = scaler.transform(entrada)

                # Predecir la dirección
                prediccion = direction_model.predict(entrada)
                direccion = "→ Derecha" if prediccion > 0.5 else "← Izquierda"

                # Dibujar la dirección sobre el rayo
                cv2.putText(frame, direccion, (x1_r, y1_r - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Mostrar la imagen con la detección
        cv2.imshow("Detección de Entidades", frame)

    else:
        print("Ventana del juego no encontrada.")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
