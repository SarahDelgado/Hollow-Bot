# ------------------- IMPORTACIONES ------------------- #
import os                                         # Proporciona funciones para interactuar con el sistema operativo, como manipulación de rutas de archivos.
import numpy as np                                # Biblioteca fundamental para computación numérica en Python, utilizada aquí para operaciones con arrays y matrices.
import cv2                                        # OpenCV (Open Source Computer Vision Library), utilizada para tareas de procesamiento de imágenes y visión artificial.
from ultralytics import YOLO                      # Framework para utilizar modelos YOLO (You Only Look Once) para la detección de objetos en tiempo real.
import tensorflow as tf                           # Biblioteca de código abierto para aprendizaje automático y desarrollo de redes neuronales, utilizada aquí para cargar el modelo Keras.
import joblib                                     # Proporciona funcionalidades para la serialización y deserialización de objetos de Python, utilizada para cargar el scaler.
import pyautogui                                  # Permite controlar el teclado y el ratón, útil para interactuar con la ventana del juego (simular pulsaciones de teclas).
import pygetwindow as gw                          # Permite obtener información y controlar ventanas del sistema operativo, utilizado para encontrar la ventana del juego.
import time                                       # Proporciona funciones relacionadas con el tiempo, como pausas en la ejecución.
import mss                                        # Para capturar capturas de pantalla de forma rápida.
from concurrent.futures import ThreadPoolExecutor # Permite ejecutar tareas en paralelo utilizando un grupo de hilos.
from obtener_rutas import resource_path           # Función personalizada para obtener las rutas de los archivos


# ------------------- VARIABLES GLOBALES ------------------- #
last_attack_time = 0      # Marca el tiempo de la última acción de ataque (podría usarse para limitar la frecuencia de ataque).
moving_direction = None   # Almacena la dirección actual en la que el personaje se está moviendo (None si no se está moviendo).
is_moving = False         # Booleano que indica si el personaje está actualmente en movimiento.


# ------------------- FUNCIONES DE DETECCIÓN ------------------- #
def detect_objects(frame, model, class_names):
    """
    Realiza la detección de objetos en un frame utilizando un modelo YOLO.

    Args:
        frame (numpy.ndarray): El frame de la imagen en el que se realizará la detección.
        model (YOLO): El modelo YOLO cargado para la detección de objetos.
        class_names (list): Una lista de los nombres de las clases de objetos que el modelo puede detectar.

    Returns:
        dict: Un diccionario donde las claves son los nombres de las clases y los valores son listas de las
              coordenadas (x1, y1, x2, y2) de los bounding boxes de los objetos detectados de esa clase.
    """
    results = model(frame)  # Realiza la predicción utilizando el modelo en el frame.
    coordinates = {class_name: [] for class_name in class_names}  # Inicializa un diccionario para almacenar las coordenadas.
    for result in results:  # Itera sobre los resultados de la detección.
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):  # Itera sobre los bounding boxes y las clases detectadas.
            x1, y1, x2, y2 = map(int, box.tolist())  # Convierte las coordenadas del bounding box a enteros.
            class_id = int(cls)  # Obtiene el ID de la clase detectada.
            class_name = class_names[class_id]  # Obtiene el nombre de la clase a partir del ID.
            coordinates[class_name].append((x1, y1, x2, y2))  # Añade las coordenadas al diccionario correspondiente a la clase.
    return coordinates

def predecir_direccion_rayo(x1_base, y1_base, x2_base, y2_base, x1_rayo, y1_rayo, x2_rayo, y2_rayo):
    """
    Predice la dirección (izquierda o derecha) en la que se debe esquivar un rayo del jefe
    utilizando un modelo Keras previamente entrenado.

    Args:
        x1_base (int): Coordenada x1 del bounding box de la base del rayo
        y1_base (int): Coordenada y1 del bounding box de la base del rayo.
        x2_base (int): Coordenada x2 del bounding box de la base del rayo.
        y2_base (int): Coordenada y2 del bounding box de la base del rayo.
        x1_rayo (int): Coordenada x1 del bounding box del rayo.
        y1_rayo (int): Coordenada y1 del bounding box del rayo.
        x2_rayo (int): Coordenada x2 del bounding box del rayo.
        y2_rayo (int): Coordenada y2 del bounding box del rayo.

    Returns:
        str: La dirección predicha para esquivar el rayo ("izquierda" o "derecha").
    """
    features = np.array([[x1_base, y1_base, x2_base, y2_base, x1_rayo, y1_rayo, x2_rayo, y2_rayo]])  # Crea un array con las características de entrada.
    features_scaled = scaler.transform(features)  # Normaliza las características utilizando el scaler cargado.
    prediction = modelo_rayo.predict(features_scaled)  # Realiza la predicción utilizando el modelo Keras.
    direccion = "izquierda" if np.argmax(prediction[0]) == 0 else "derecha"  # Determina la dirección basada en el índice del valor máximo en la predicción.
    return direccion


# ------------------- FUNCIONES DE CONTROL DEL PERSONAJE PRINCIPAL ------------------- #
def move_player(direction):
    """
    Simula la pulsación de teclas para mover al personaje principal en una dirección específica.

    Args:
        direction (str): La dirección en la que se desea mover al personaje principal ("up", "left", "right", "Attack", "Dash").
    """
    if direction == "up":
        # Simula dos saltos cortos presionando 'Z' dos veces.
        pyautogui.keyDown('Z')
        time.sleep(0.5)
        pyautogui.keyUp('Z')
        pyautogui.keyDown('Z')
        time.sleep(0.5)
        pyautogui.keyUp('Z')
    elif direction == "left":
        pyautogui.keyDown('left')  # Mantiene presionada la tecla de flecha izquierda.
    elif direction == "right":
        pyautogui.keyDown('right') # Mantiene presionada la tecla de flecha derecha.
    elif direction == "Attack":
        pyautogui.keyDown('X')    # Simula una pulsación de la tecla 'X' (ataque).
        pyautogui.keyUp('X')
    elif direction == "Dash":
        pyautogui.keyDown('C')    # Simula una pulsación de la tecla 'C' (dash).
        pyautogui.keyUp('C')

def start_movement(direction):
    """
    Inicia el movimiento del personaje principal en una dirección específica, deteniendo el movimiento anterior si es necesario.

    Args:
        direction (str): La dirección en la que se desea iniciar el movimiento ("left" o "right").
    """
    global moving_direction, is_moving
    if moving_direction != direction or not is_moving:
        stop_movement()  # Detiene el movimiento actual antes de iniciar uno nuevo.
        pyautogui.keyDown(direction)  # Mantiene presionada la tecla de la dirección.
        moving_direction = direction  # Actualiza la dirección actual del movimiento.
        is_moving = True              # Marca que el personaje principal está en movimiento.

def stop_movement():
    """
    Detiene el movimiento del personaje principal si actualmente se está moviendo.
    """
    global moving_direction, is_moving
    if is_moving and moving_direction:
        pyautogui.keyUp(moving_direction)  # Levanta la tecla de la dirección actual.
        is_moving = False                  # Marca que el personaje principal ya no está en movimiento.
        moving_direction = None            # Resetea la dirección del movimiento.


# ------------------- LÓGICA DE EVASIÓN ------------------- #
def evade_attack(boss_coordinates, player_coordinates):
    """
    Implementa la lógica para que el personaje principal evada los ataques del jefe basándose en las coordenadas detectadas.

    Args:
        boss_coordinates (dict): Un diccionario con las coordenadas de los objetos del jefe detectados.
        player_coordinates (list): Una lista con las coordenadas del personaje principal detectado.
    """
    global is_moving, moving_direction
    player_x = player_y = boss_x = boss_y = 0

    # Evade el cristal que el boss lanza directamente al personaje principal
    if "cristal_boss" in boss_coordinates and boss_coordinates["cristal_boss"] and player_coordinates:
        boss_x1, boss_y1, boss_x2, boss_y2 = boss_coordinates["cristal_boss"][0]
        player_x1, player_y1, player_x2, player_y2 = player_coordinates[0]
        boss_x = (boss_x1 + boss_x2) // 2
        boss_y = (boss_y1 + boss_y2) // 2
        player_x = (player_x1 + player_x2) // 2
        player_y = (player_y1 + player_y2) // 2

        # Si el cristal está cerca horizontalmente y debajo del personaje principal, realiza un dash para evadir.
        if abs(player_x - boss_x) < 60 and boss_y < (player_y + 20):
            stop_movement()
            move_player("Dash")
            return

    # Evade el rayo horizontal que lanza el jefe
    if "boss_ray" in boss_coordinates and boss_coordinates["boss_ray"]:
        stop_movement()
        move_player("up")  # Intenta saltar para evitar el rayo horizontal.
        return

    # Evade los rayos que caen desde arriba
    if "ray_from_above" in boss_coordinates and boss_coordinates["ray_from_above"]:
        for ray in boss_coordinates["ray_from_above"]:
            x1_rayo, y1_rayo, x2_rayo, y2_rayo = ray
            centro_rayo = (x1_rayo + x2_rayo) // 2
            ancho_rayo = abs(x2_rayo - x1_rayo) // 2

            # Si el personaje principal está dentro del ancho del rayo (con un pequeño margen).
            if abs(player_x - centro_rayo) < ancho_rayo + 10:
                # Intenta predecir la dirección del rayo basándose en la posición de la base del jefe.
                if "base" in boss_coordinates and boss_coordinates["base"]:
                    x1_base, y1_base, x2_base, y2_base = boss_coordinates["base"][0]
                    direccion_rayo = predecir_direccion_rayo(
                        x1_base, y1_base, x2_base, y2_base, x1_rayo, y1_rayo, x2_rayo, y2_rayo
                    )
                else:
                    direccion_rayo = "desconocida"

                # Verifica si hay otros rayos cerca a la izquierda o derecha del personaje principal.
                hay_rayo_izquierda = any(
                    (rx1 + rx2) // 2 < player_x for (rx1, _, rx2, _) in boss_coordinates["ray_from_above"]
                )
                hay_rayo_derecha = any(
                    (rx1 + rx2) // 2 > player_x for (rx1, _, rx2, _) in boss_coordinates["ray_from_above"]
                )

                # Intenta moverse en la dirección predicha si no hay otro rayo bloqueando ese lado.
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

    # Lógica para mantener una distancia óptima del jefe y atacar
    if player_coordinates and "cristal_boss" in boss_coordinates and boss_coordinates["cristal_boss"]:
        player_x1, player_y1, player_x2, player_y2 = player_coordinates[0]
        boss_x1, boss_y1, boss_x2, boss_y2 = boss_coordinates["cristal_boss"][0]
        player_x = (player_x1 + player_x2) // 2
        boss_x = (boss_x1 + boss_x2) // 2
        distancia_deseada = 150  # Distancia horizontal deseada entre el personaje principal y el jefe.
        margen_tolerancia = 10  # Margen de tolerancia para la distancia.
        distancia_actual = abs(player_x - boss_x)

        # Si el personaje principal está demasiado cerca, se aleja.
        if distancia_actual < (distancia_deseada - margen_tolerancia):
            stop_movement()
            direction = "left" if player_x < boss_x else "right"
            start_movement(direction)
        # Si el personaje principal está demasiado lejos, se acerca.
        elif distancia_actual > (distancia_deseada + margen_tolerancia):
            stop_movement()
            direction = "right" if player_x < boss_x else "left"
            start_movement(direction)
        # Si está a la distancia deseada, ataca.
        else:
            stop_movement()
            move_player("Attack")
    # Si no se detecta jefe, detiene el movimiento.
    else:
        stop_movement()


# ------------------- FUNCIÓN DE CAPTURA DE PANTALLA ------------------- #
def capture_game_window(window_name):
    """
    Captura una porción específica de la pantalla correspondiente a la ventana del juego.

    Args:
        window_name (str): El título de la ventana del juego que se desea capturar.

    Returns:
        numpy.ndarray or None: Un frame de la ventana del juego en formato BGR si se encuentra la ventana,
                               None en caso contrario.
    """
    windows = gw.getWindowsWithTitle(window_name)  # Obtiene todas las ventanas con el título especificado.
    if not windows:
        return None  # Si no se encuentra ninguna ventana con ese título, retorna None.
    game_window = windows[0]  # Selecciona la primera ventana encontrada (asumiendo que es la correcta).
    left, top, width, height = game_window.left, game_window.top, game_window.width, game_window.height  # Obtiene las dimensiones de la ventana.
    with mss.mss() as sct:  # Utiliza 'mss' para una captura de pantalla eficiente.
        monitor = {"top": top, "left": left, "width": width, "height": height}  # Define el área del monitor a capturar.
        img = sct.grab(monitor)  # Realiza la captura de pantalla del área definida.
        frame = np.array(img)[:, :, :3]  # Convierte la imagen capturada a un array NumPy y extrae los canales RGB.
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convierte el formato de color de RGB a BGR (el formato que OpenCV utiliza por defecto).



# ------------------- CARGA DE MODELOS Y SCALER ------------------- #
# Ruta modelo del jefe
ruta_model_boss = resource_path(os.path.join(
    "..", "Entrenamiento", "training_boos_junto_modelo_graficas", "content", "runs", "detect", "train2", "weights", "best.pt"
))
# Modelo YOLO entrenado para detectar al boss y los rayos
model_boss = YOLO(ruta_model_boss)
# Ruta modelo del personaje principal
ruta_model_player = resource_path(os.path.join(
    "..", "Entrenamiento", "training_character_junto_modelos_grafica", "content", "runs", "detect", "train", "weights", "best.pt"
))
# Modelo YOLO entrenado para detectar al personaje principal
model_player = YOLO(ruta_model_player)
# Ruta al modelo keras
ruta_modelo_rayo = resource_path(os.path.join("..", "Entrenamiento", "Codigo_keras_detectar_rayo_izq_derch", "mejor_modelo_direccion_rayo.keras"))
# Modelo Keras entrenado para predecir si esquivar hacia izquierda o derecha
modelo_rayo = tf.keras.models.load_model(ruta_modelo_rayo)
# Ruta al scaler
ruta_scaler = resource_path(os.path.join("..", "Entrenamiento", "Codigo_keras_detectar_rayo_izq_derch", "scaler_rayo.pkl"))
# Scaler previamente guardado para normalizar vectores de entrada al modelo Keras
scaler = joblib.load(ruta_scaler)


# ------------------- CLASES DE LOS MODELOS ------------------- #
# Define los nombres de las clases que el modelo de detección del jefe puede identificar.
boss_class_names = ["cristal_boss", "boss_ray", "ray_from_above", "base"]
# Define el nombre de la clase del personaje principal.
player_class_name = "character"


# ------------------- BUCLE PRINCIPAL DE EJECUCIÓN ------------------- #
while True:
    """
    Bucle principal del script que se ejecuta continuamente para:
    1. Capturar la pantalla del juego.
    2. Detectar al jefe y al personaje principal en el frame capturado utilizando modelos YOLO en hilos separados para mejorar el rendimiento.
    3. Ejecutar la lógica de evasión de ataques basada en las coordenadas detectadas.
    4. Permitir la salida del bucle al presionar la tecla 'q'.
    """
    frame = capture_game_window("Hollow Knight")  # Captura un frame de la ventana del juego "Hollow Knight".
    if frame is None:
        continue  # Si no se pudo capturar el frame (por ejemplo, si la ventana no se encuentra), salta a la siguiente iteración.

    # Utiliza un ThreadPoolExecutor para ejecutar las detecciones de objetos en paralelo.
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_boss = executor.submit(detect_objects, frame, model_boss, boss_class_names)  # Envía la detección del jefe a un hilo.
        future_player = executor.submit(detect_objects, frame, model_player, [player_class_name])  # Envía la detección del personaje principal a otro hilo.
        boss_coordinates = future_boss.result()  # Espera a que termine la detección del jefe y obtiene los resultados.
        player_coordinates = future_player.result().get(player_class_name, [])  # Espera a que termine la detección del personaje principal y obtiene las coordenadas (si se detecta).

    evade_attack(boss_coordinates, player_coordinates)  # Ejecuta la lógica para que el personaje principal evada los ataques del jefe.

    # Espera 1 milisegundo y verifica si se ha presionado la tecla 'q' para salir del bucle.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break  # Si se presiona 'q', sale del bucle principal.

cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV que puedan haberse creado (aunque en este script no se visualizan frames).