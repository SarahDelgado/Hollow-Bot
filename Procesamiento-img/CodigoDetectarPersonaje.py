import cv2
import numpy as np
import mss
import pygetwindow as gw
from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO("hollow_knight_best.pt")

# Nombre de la ventana del juego
#game_window_title = "Hollow Knight"
game_window_title = "Hollow Knight - Any% No Major Glitches in 32:24"
def get_game_window():
    """ Obtiene la posición y dimensiones de la ventana del juego. """
    windows = gw.getWindowsWithTitle(game_window_title)
    if windows:
        game_window = windows[0]  # Tomamos la primera coincidencia
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

        # Detectar el personaje en la imagen
        results = model(frame)

        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = box.tolist()  # Convertir a lista de Python
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Mostrar la imagen con la detección
        cv2.imshow("Detección de Personaje", frame)

    else:
        print("Ventana del juego no encontrada.")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
