import cv2
import mss
import numpy as np
from ultralytics import YOLO
import pyautogui
import pygetwindow as gw
import time
import joblib
from keras.models import load_model
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor
import threading

# Inicializa modelos
modelo_rayo = load_model("mejor_modelo_direccion_rayo.keras")
scaler = joblib.load("scaler_rayo.pkl")
model_boss = YOLO("best_boss.pt")
model_player = YOLO("best_character.pt")
boss_class_names = ["cristal_boss", "boss_ray", "ray_from_above", "base"]
player_class_name = "character"

# Compartido entre hilos
shared_frame = None
shared_detections = {"boss": {}, "player": []}
lock = threading.Lock()

def esperar_frase_clave(frase_esperada="Inicia"):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            texto = recognizer.recognize_google(audio, language='es-ES')
            if frase_esperada.lower() in texto.lower():
                break
        except (sr.UnknownValueError, sr.RequestError):
            pass

def detect_objects(frame, model, class_names):
    results = model(frame)
    coordinates = {class_name: [] for class_name in class_names}
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())
            class_id = int(cls)
            class_name = class_names[class_id]
            coordinates[class_name].append((x1, y1, x2, y2))
    return coordinates

def predecir_direccion_rayo(x1_base, y1_base, x2_base, y2_base, x1_rayo, y1_rayo, x2_rayo, y2_rayo):
    features = np.array([[x1_base, y1_base, x2_base, y2_base, x1_rayo, y1_rayo, x2_rayo, y2_rayo]])
    features_scaled = scaler.transform(features)
    prediction = modelo_rayo.predict(features_scaled)
    return "izquierda" if np.argmax(prediction[0]) == 0 else "derecha"

def move_player(direction):
    if direction == "up":
        pyautogui.keyDown('Z')
        time.sleep(0.5)
        pyautogui.keyUp('Z')
        pyautogui.keyDown('Z')
        time.sleep(0.5)
        pyautogui.keyUp('Z')
    elif direction in ("left", "right"):
        pyautogui.keyDown(direction)
    elif direction == "Attack":
        pyautogui.keyDown('X')
        pyautogui.keyUp('X')
    elif direction == "Dash":
        pyautogui.keyDown('C')
        pyautogui.keyUp('C')

moving_direction = None
is_moving = False

def start_movement(direction):
    global moving_direction, is_moving
    if moving_direction != direction or not is_moving:
        stop_movement()
        pyautogui.keyDown(direction)
        moving_direction = direction
        is_moving = True

def stop_movement():
    global moving_direction, is_moving
    if is_moving and moving_direction:
        pyautogui.keyUp(moving_direction)
        is_moving = False
        moving_direction = None

def evade_attack(boss_coords, player_coords):
    global is_moving, moving_direction
    if not player_coords:
        stop_movement()
        return

    player_x1, player_y1, player_x2, player_y2 = player_coords[0]
    player_x = (player_x1 + player_x2) // 2
    player_y = (player_y1 + player_y2) // 2

    if "cristal_boss" in boss_coords and boss_coords["cristal_boss"]:
        boss_x1, boss_y1, boss_x2, boss_y2 = boss_coords["cristal_boss"][0]
        boss_x = (boss_x1 + boss_x2) // 2
        boss_y = (boss_y1 + boss_y2) // 2

        if abs(player_x - boss_x) < 60 and boss_y < (player_y + 20):
            stop_movement()
            move_player("Dash")
            return

    if "boss_ray" in boss_coords and boss_coords["boss_ray"]:
        stop_movement()
        move_player("up")
        return

    if "ray_from_above" in boss_coords and boss_coords["ray_from_above"]:
        for ray in boss_coords["ray_from_above"]:
            x1_rayo, y1_rayo, x2_rayo, y2_rayo = ray
            centro_rayo = (x1_rayo + x2_rayo) // 2
            ancho_rayo = abs(x2_rayo - x1_rayo) // 2

            if abs(player_x - centro_rayo) < ancho_rayo + 10:
                direccion_rayo = "desconocida"
                if "base" in boss_coords and boss_coords["base"]:
                    x1_base, y1_base, x2_base, y2_base = boss_coords["base"][0]
                    direccion_rayo = predecir_direccion_rayo(
                        x1_base, y1_base, x2_base, y2_base, x1_rayo, y1_rayo, x2_rayo, y2_rayo
                    )

                hay_rayo_izq = any((rx1 + rx2) // 2 < player_x for (rx1, _, rx2, _) in boss_coords["ray_from_above"])
                hay_rayo_der = any((rx1 + rx2) // 2 > player_x for (rx1, _, rx2, _) in boss_coords["ray_from_above"])

                if "cristal_boss" in boss_coords and boss_coords["cristal_boss"]:
                    boss_x = (boss_coords["cristal_boss"][0][0] + boss_coords["cristal_boss"][0][2]) // 2
                    if direccion_rayo == "izquierda" and not hay_rayo_der and boss_x > player_x:
                        start_movement("right")
                    elif direccion_rayo == "derecha" and not hay_rayo_izq and boss_x < player_x:
                        start_movement("left")
                    else:
                        stop_movement()
                else:
                    stop_movement()
                return

    if "cristal_boss" in boss_coords and boss_coords["cristal_boss"]:
        boss_x = (boss_coords["cristal_boss"][0][0] + boss_coords["cristal_boss"][0][2]) // 2
        dist = abs(player_x - boss_x)
        if dist < 140:
            direction = "left" if player_x < boss_x else "right"
            start_movement(direction)
        elif dist > 160:
            direction = "right" if player_x < boss_x else "left"
            start_movement(direction)
        else:
            stop_movement()
            move_player("Attack")
    else:
        stop_movement()

def capture_game_window(window_name="Hollow Knight"):
    windows = gw.getWindowsWithTitle(window_name)
    if not windows:
        return None
    game_window = windows[0]
    with mss.mss() as sct:
        monitor = {
            "top": game_window.top, "left": game_window.left,
            "width": game_window.width, "height": game_window.height
        }
        img = sct.grab(monitor)
        return np.array(img)[:, :, :3]

def visual_thread():
    global shared_frame, shared_detections
    while True:
        time.sleep(1/30)
        lock.acquire()
        if shared_frame is None:
            lock.release()
            continue
        frame_copy = shared_frame.copy()
        boss_coords = shared_detections["boss"]
        player_coords = shared_detections["player"]
        lock.release()

        # Dibujar jugador
        for (x1, y1, x2, y2) in player_coords:
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_copy, "character", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Dibujar enemigos
        for label, boxes in boss_coords.items():
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Detección en tiempo real", frame_copy)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

# Lanzar hilo de visualización
threading.Thread(target=visual_thread, daemon=True).start()

# Iniciar detección tras comando por voz
esperar_frase_clave("Inicia")

# Bucle principal de detección
while True:
    frame = capture_game_window()
    if frame is None:
        continue

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_boss = executor.submit(detect_objects, frame, model_boss, boss_class_names)
        future_player = executor.submit(detect_objects, frame, model_player, [player_class_name])
        boss_coordinates = future_boss.result()
        player_coordinates = future_player.result().get(player_class_name, [])

    # Compartir datos para visualización
    lock.acquire()
    shared_frame = frame
    shared_detections = {"boss": boss_coordinates, "player": player_coordinates}
    lock.release()

    # Ejecutar lógica de evasión
    evade_attack(boss_coordinates, player_coordinates)
