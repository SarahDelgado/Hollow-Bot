import cv2
import mss
import numpy as np
from ultralytics import YOLO
import pyautogui
import pygetwindow as gw

# Función para detectar objetos en la imagen sin dibujar cuadros
def detect_objects(frame, model, class_names):
    """
    Detecta objetos en la imagen usando el modelo YOLO.
    Retorna un diccionario con las coordenadas de cada clase.
    """
    results = model(frame)  # Realiza la detección con YOLO
    coordinates = {class_name: [] for class_name in class_names}  # Diccionario para almacenar coordenadas

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())  # Convertir coordenadas a enteros
            class_id = int(cls)  # Obtener ID de la clase
            class_name = class_names[class_id]  # Obtener el nombre de la clase
            coordinates[class_name].append((x1, y1, x2, y2))

    return coordinates


    # Función para mover al jugador usando el teclado
def move_player(direction):
    """
    Simula la pulsación de teclas para mover al jugador en la dirección especificada.
    Las direcciones posibles son: 'up', 'down', 'left', 'right'.
    """
    print(f"Moviendo: {direction}")
    if direction == "up":
        pyautogui.keyDown('Z')
        pyautogui.keyUp('Z')  # Simula la pulsación de la tecla Z para saltar
    elif direction == "down":
        pyautogui.keyDown('down')
        pyautogui.keyUp('down')  # Simula la pulsación de la flecha hacia abajo
    elif direction == "left":
        pyautogui.keyDown('left')
        pyautogui.keyUp('left') # Simula la pulsación de la flecha hacia izquierda
    elif direction == "right":
        pyautogui.keyDown('right')
        pyautogui.keyUp('right')  # Simula la pulsación de la flecha hacia derecha


# Función para calcular y ejecutar los movimientos del jugador
def evade_attack(boss_coordinates, player_coordinates):
    """
    Calcula la mejor dirección de movimiento para esquivar ataques y mantenerse a salvo.
    """
    if not player_coordinates:
        return  # Si no se detecta al jugador, no hacer nada

    player_x1, player_y1, player_x2, player_y2 = player_coordinates[0]
    player_x = (player_x1 + player_x2) // 2  # Centro del jugador en X
    player_y = (player_y1 + player_y2) // 2  # Centro del jugador en Y

    # Evitar colisión con el boss
    if "cristal_boss" in boss_coordinates and boss_coordinates["cristal_boss"]:
        boss_x1, boss_y1, boss_x2, boss_y2 = boss_coordinates["cristal_boss"][0]
        boss_x = (boss_x1 + boss_x2) // 2

        if abs(player_x - boss_x) < 50:  # Si está demasiado cerca
            move_player("right" if player_x < boss_x else "left")
            return

    # Evitar ataques del boss
    for ray in boss_coordinates.get("boss_ray", []) + boss_coordinates.get("ray_from_above", []):
        ray_x1, ray_y1, ray_x2, ray_y2 = ray
        if ray_y1 < player_y < ray_y2:
            if player_x < ray_x1:
                move_player("right")
            elif player_x > ray_x2:
                move_player("left")
            else:
                move_player("up")
            return


def capture_game_window(window_name):
    """ Captura la ventana del juego por su nombre. """
    windows = gw.getWindowsWithTitle(window_name)  # Busca la ventana
    if not windows:
        print(f"No se encontró la ventana '{window_name}'")
        return None

    game_window = windows[0]  # Tomar la primera coincidencia
    left, top, width, height = game_window.left, game_window.top, game_window.width, game_window.height

    with mss.mss() as sct:
        monitor = {"top": top, "left": left, "width": width, "height": height}
        img = sct.grab(monitor)
        frame = np.array(img)[:, :, :3]  # Convertir a imagen OpenCV
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Cargar modelos entrenados
model_boss = YOLO("best_boss.pt")  # Modelo para detectar al boss y sus ataques
model_player = YOLO("best_character.pt")  # Modelo para detectar al jugador

# Nombres de las clases en cada modelo
boss_class_names = ["cristal_boss", "boss_ray", "ray_from_above", "base"]
player_class_name = "character"  # Suponemos que el modelo solo detecta el personaje

# Bucle principal
while True:
    frame = capture_game_window("Hollow Knight")
    boss_coordinates = detect_objects(frame, model_boss, boss_class_names)
    player_coordinates = detect_objects(frame, model_player, [player_class_name])
    evade_attack(boss_coordinates, player_coordinates.get(player_class_name, []))

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Presiona 'q' para salir
        break

cv2.destroyAllWindows()
