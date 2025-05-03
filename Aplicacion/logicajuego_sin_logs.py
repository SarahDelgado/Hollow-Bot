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


def esperar_frase_clave(frase_esperada="Inicia"):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        print(f"🎤 Esperando que digas: '{frase_esperada}'...")

        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            texto = recognizer.recognize_google(audio, language='es-ES')
            print(f"🗣️ Dijiste: {texto}")

            if frase_esperada.lower() in texto.lower():
                print("✅ Frase reconocida. Iniciando aplicación...")
                break  # Salir del bucle y continuar
            else:
                print("❌ Esa no es la frase. Intenta de nuevo.\n")

        except sr.UnknownValueError:
            print("❗ No se entendió lo que dijiste. Intenta otra vez.\n")
        except sr.RequestError as e:
            print(f"❗ Error con el servicio de reconocimiento: {e}")
            exit()


esperar_frase_clave("Inicia")

modelo_rayo = load_model("mejor_modelo_direccion_rayo.keras")
scaler = joblib.load("scaler_rayo.pkl")

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
    direccion = "izquierda" if np.argmax(prediction[0]) == 0 else "derecha"
    return direccion

def move_player(direction):
    if direction == "up":
        pyautogui.keyDown('Z')
        time.sleep(0.5)
        pyautogui.keyUp('Z')
        pyautogui.keyDown('Z')
        time.sleep(0.5)
        pyautogui.keyUp('Z')
    elif direction == "left":
        pyautogui.keyDown('left')
    elif direction == "right":
        pyautogui.keyDown('right')
    elif direction == "Attack":
        pyautogui.keyDown('X')
        pyautogui.keyUp('X')
    elif direction == "Dash":
        pyautogui.keyDown('C')
        pyautogui.keyUp('C')

last_attack_time = 0
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

def evade_attack(boss_coordinates, player_coordinates):
    global is_moving, moving_direction
    player_x = player_y = boss_x = boss_y = 0

    if "cristal_boss" in boss_coordinates and boss_coordinates["cristal_boss"] and player_coordinates:
        boss_x1, boss_y1, boss_x2, boss_y2 = boss_coordinates["cristal_boss"][0]
        player_x1, player_y1, player_x2, player_y2 = player_coordinates[0]
        boss_x = (boss_x1 + boss_x2) // 2
        boss_y = (boss_y1 + boss_y2) // 2
        player_x = (player_x1 + player_x2) // 2
        player_y = (player_y1 + player_y2) // 2

        if abs(player_x - boss_x) < 60 and boss_y < (player_y + 20):
            stop_movement()
            move_player("Dash")
            return

    if "boss_ray" in boss_coordinates and boss_coordinates["boss_ray"]:
        stop_movement()
        move_player("up")
        return

    if "ray_from_above" in boss_coordinates and boss_coordinates["ray_from_above"]:
        for ray in boss_coordinates["ray_from_above"]:
            x1_rayo, y1_rayo, x2_rayo, y2_rayo = ray
            centro_rayo = (x1_rayo + x2_rayo) // 2
            ancho_rayo = abs(x2_rayo - x1_rayo) // 2

            if abs(player_x - centro_rayo) < ancho_rayo + 10:
                if "base" in boss_coordinates and boss_coordinates["base"]:
                    x1_base, y1_base, x2_base, y2_base = boss_coordinates["base"][0]
                    direccion_rayo = predecir_direccion_rayo(
                        x1_base, y1_base, x2_base, y2_base, x1_rayo, y1_rayo, x2_rayo, y2_rayo
                    )
                else:
                    direccion_rayo = "desconocida"

                hay_rayo_izquierda = any(
                    (rx1 + rx2) // 2 < player_x for (rx1, _, rx2, _) in boss_coordinates["ray_from_above"]
                )
                hay_rayo_derecha = any(
                    (rx1 + rx2) // 2 > player_x for (rx1, _, rx2, _) in boss_coordinates["ray_from_above"]
                )

                if "cristal_boss" in boss_coordinates and boss_coordinates["cristal_boss"]:
                    boss_x1, _, boss_x2, _ = boss_coordinates["cristal_boss"][0]
                    boss_x_centro = (boss_x1 + boss_x2) // 2

                    if direccion_rayo == "izquierda" and not hay_rayo_derecha and boss_x_centro > player_x:
                        start_movement("right")
                    elif direccion_rayo == "derecha" and not hay_rayo_izquierda and boss_x_centro < player_x:
                        start_movement("left")
                    else:
                        stop_movement()
                else:
                    stop_movement()
                return

    if player_coordinates and "cristal_boss" in boss_coordinates and boss_coordinates["cristal_boss"]:
        player_x1, player_y1, player_x2, player_y2 = player_coordinates[0]
        boss_x1, boss_y1, boss_x2, boss_y2 = boss_coordinates["cristal_boss"][0]
        player_x = (player_x1 + player_x2) // 2
        boss_x = (boss_x1 + boss_x2) // 2
        distancia_deseada = 150
        margen_tolerancia = 10
        distancia_actual = abs(player_x - boss_x)

        if distancia_actual < (distancia_deseada - margen_tolerancia):
            stop_movement()
            direction = "left" if player_x < boss_x else "right"
            start_movement(direction)
        elif distancia_actual > (distancia_deseada + margen_tolerancia):
            stop_movement()
            direction = "right" if player_x < boss_x else "left"
            start_movement(direction)
        else:
            stop_movement()
            move_player("Attack")
    else:
        stop_movement()

def capture_game_window(window_name):
    windows = gw.getWindowsWithTitle(window_name)
    if not windows:
        return None
    game_window = windows[0]
    left, top, width, height = game_window.left, game_window.top, game_window.width, game_window.height
    with mss.mss() as sct:
        monitor = {"top": top, "left": left, "width": width, "height": height}
        img = sct.grab(monitor)
        frame = np.array(img)[:, :, :3]
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

model_boss = YOLO("best_boss.pt")
model_player = YOLO("best_character.pt")
boss_class_names = ["cristal_boss", "boss_ray", "ray_from_above", "base"]
player_class_name = "character"

while True:
    frame = capture_game_window("Hollow Knight")
    if frame is None:
        continue

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_boss = executor.submit(detect_objects, frame, model_boss, boss_class_names)
        future_player = executor.submit(detect_objects, frame, model_player, [player_class_name])
        boss_coordinates = future_boss.result()
        player_coordinates = future_player.result().get(player_class_name, [])

    evade_attack(boss_coordinates, player_coordinates)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
