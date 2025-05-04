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
import speech_recognition as sr                   # Biblioteca para la conversión de voz a texto.
import threading                                  # Proporciona soporte para la creación y gestión de hilos (threads) para la ejecución concurrente.
from obtener_rutas import resource_path           # Función personalizada para obtener las rutas de los archivos


# ------------------- CARGA DE MODELOS Y SCALER ------------------- #
# Ruta modelo del jefe
ruta_model_boss = resource_path(os.path.join("Entrenamiento", "training_boos_junto_modelo_graficas", "content", "runs", "detect", "train2", "weights", "best.pt"
))
# Modelo YOLO entrenado para detectar al boss y los rayos
model_boss = YOLO(ruta_model_boss)
# Ruta modelo del personaje principal
ruta_model_player = resource_path(os.path.join("Entrenamiento", "training_character_junto_modelos_grafica", "content", "runs", "detect", "train", "weights", "best.pt"
))
# Modelo YOLO entrenado para detectar al personaje principal
model_player = YOLO(ruta_model_player)
# Ruta al modelo keras
ruta_modelo_rayo = resource_path(os.path.join("Entrenamiento", "Codigo_keras_detectar_rayo_izq_derch", "mejor_modelo_direccion_rayo.keras"))
# Modelo Keras entrenado para predecir si esquivar hacia izquierda o derecha
modelo_rayo = tf.keras.models.load_model(ruta_modelo_rayo)
# Ruta al scaler
ruta_scaler = resource_path(os.path.join("Entrenamiento", "Codigo_keras_detectar_rayo_izq_derch", "scaler_rayo.pkl"))
# Scaler previamente guardado para normalizar vectores de entrada al modelo Keras
scaler = joblib.load(ruta_scaler)

# Define los nombres de las clases que el modelo de detección del jefe puede identificar.
boss_class_names = ["cristal_boss", "boss_ray", "ray_from_above", "base"]
# Define el nombre de la clase del personaje principal.
player_class_name = "character"

# ------------------- VARIABLES COMPARTIDAS ENTRE HILOS ------------------- #
shared_frame = None             # Variable para almacenar el último frame capturado, compartido entre hilos.
shared_detections = {"boss": {}, "player": []} # Diccionario para almacenar las detecciones del boss y el personaje principal, compartido entre hilos.
lock = threading.Lock()         # Lock para controlar el acceso a las variables compartidas y evitar condiciones de carrera.

# ------------------- FUNCIÓN DE RECONOCIMIENTO DE VOZ ------------------- #
def esperar_frase_clave(frase_esperada="Inicia"):
    """
    Escucha continuamente hasta que el usuario dice la frase clave especificada.

    Args:
        frase_esperada (str, optional): La frase que se espera escuchar para activar la detección.
                                        Por defecto es "Inicia".
    """
    recognizer = sr.Recognizer() # Crea una instancia del reconocedor de voz.
    mic = sr.Microphone()       # Crea una instancia del micrófono para la entrada de audio.
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source) # Ajusta el reconocedor al ruido ambiental para mejorar la precisión.
            audio = recognizer.listen(source)          # Escucha la entrada de audio desde el micrófono.
        try:
            texto = recognizer.recognize_google(audio, language='es-ES') # Intenta convertir el audio a texto utilizando el servicio de Google en español.
            if frase_esperada.lower() in texto.lower(): # Comprueba si la frase esperada (en minúsculas) está presente en el texto reconocido (también en minúsculas).
                break # Sale del bucle si la frase clave es detectada.
        except (sr.UnknownValueError, sr.RequestError):
            pass # Si no se entiende el audio o hay un error con el servicio de reconocimiento, simplemente ignora y continúa escuchando.

# ------------------- FUNCIÓN DE DETECCIÓN DE OBJETOS ------------------- #
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
    coordinates = {class_name: [] for class_name in class_names}  # Inicializa un diccionario para almacenar las coordenadas por clase.
    for result in results:  # Itera sobre los resultados de la detección.
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):  # Itera sobre los bounding boxes y las etiquetas de clase detectadas.
            x1, y1, x2, y2 = map(int, box.tolist())  # Convierte las coordenadas del bounding box a enteros.
            class_id = int(cls)  # Obtiene el ID de la clase detectada.
            class_name = class_names[class_id]  # Obtiene el nombre de la clase a partir del ID.
            coordinates[class_name].append((x1, y1, x2, y2))  # Añade las coordenadas del objeto detectado a la lista correspondiente en el diccionario.
    return coordinates

# ------------------- FUNCIÓN DE PREDICCIÓN DE DIRECCIÓN DE RAYO ------------------- #
def predecir_direccion_rayo(x1_base, y1_base, x2_base, y2_base, x1_rayo, y1_rayo, x2_rayo, y2_rayo):
    """
    Predice la dirección (izquierda o derecha) en la que se debe esquivar un rayo del jefe
    utilizando un modelo Keras previamente entrenado.

    Args:
        x1_base (int): Coordenada x1 del bounding box de la base del rayo.
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
    features = np.array([[x1_base, y1_base, x2_base, y2_base, x1_rayo, y1_rayo, x2_rayo, y2_rayo]]) # Crea un array NumPy con las características de entrada.
    features_scaled = scaler.transform(features) # Normaliza las características utilizando el scaler cargado.
    prediction = modelo_rayo.predict(features_scaled) # Realiza la predicción utilizando el modelo Keras.
    return "izquierda" if np.argmax(prediction[0]) == 0 else "derecha" # Retorna "izquierda" si la probabilidad de la clase 0 es mayor, sino retorna "derecha".

# ------------------- FUNCIONES DE CONTROL DEL PERSONAJE PRINCIPAL ------------------- #
def move_player(direction):
    """
    Simula la pulsación de teclas para mover al personaje principal en una dirección específica.

    Args:
        direction (str): La dirección en la que se desea mover al personaje principal ("up", "left", "right", "Attack", "Dash").
    """
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
    """
    Inicia el movimiento del personaje principal en una dirección específica, deteniendo el movimiento anterior si es necesario.

    Args:
        direction (str): La dirección en la que se desea iniciar el movimiento ("left" o "right").
    """
    global moving_direction, is_moving
    if moving_direction != direction or not is_moving:
        stop_movement()
        pyautogui.keyDown(direction)
        moving_direction = direction
        is_moving = True

def stop_movement():
    """
    Detiene el movimiento del personaje principal si actualmente se está moviendo.
    """
    global moving_direction, is_moving
    if is_moving and moving_direction:
        pyautogui.keyUp(moving_direction)
        is_moving = False
        moving_direction = None

# ------------------- LÓGICA DE EVASIÓN DE ATAQUES ------------------- #
def evade_attack(boss_coords, player_coords):
    """
    Implementa la lógica para que el personaje principal evada los ataques del jefe basándose en las coordenadas detectadas.

    Args:
        boss_coords (dict): Un diccionario con las coordenadas de los objetos del jefe detectados.
        player_coords (list): Una lista con las coordenadas del personaje principal detectado.
    """
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

# ------------------- FUNCIÓN DE CAPTURA DE PANTALLA ------------------- #
def capture_game_window(window_name="Hollow Knight"):
    """
    Captura una porción específica de la pantalla correspondiente a la ventana del juego.

    Args:
        window_name (str, optional): El título de la ventana del juego que se desea capturar.
                                     Por defecto es "Hollow Knight".

    Returns:
        numpy.ndarray or None: Un frame de la ventana del juego en formato BGR si se encuentra la ventana,
                               None en caso contrario.
    """
    windows = gw.getWindowsWithTitle(window_name) # Obtiene una lista de ventanas con el título especificado.
    if not windows:
        return None # Si no se encuentra ninguna ventana con ese título, retorna None.
    game_window = windows[0] # Selecciona la primera ventana de la lista (asumiendo que es la ventana del juego).
    with mss.mss() as sct: # Utiliza la biblioteca 'mss' para una captura de pantalla rápida.
        monitor = {
            "top": game_window.top, "left": game_window.left,
            "width": game_window.width, "height": game_window.height
        } # Define el área del monitor a capturar basándose en las dimensiones de la ventana del juego.
        img = sct.grab(monitor) # Realiza la captura de pantalla del área definida.
        return np.array(img)[:, :, :3] # Convierte la imagen capturada a un array NumPy y extrae los canales de color RGB.

# ------------------- HILO DE VISUALIZACIÓN ------------------- #
def visual_thread():
    """
    Hilo dedicado a la visualización en tiempo real de las detecciones sobre la captura de pantalla.
    """
    global shared_frame, shared_detections
    while True:
        time.sleep(1/30) # Pausa para limitar la velocidad de fotogramas de la visualización a 30 FPS.
        lock.acquire()   # Adquiere el lock para acceder de forma segura a las variables compartidas.
        if shared_frame is None:
            lock.release() # Libera el lock si no hay un frame disponible.
            continue
        frame_copy = shared_frame.copy() # Crea una copia del frame compartido para evitar modificaciones mientras se visualiza.
        boss_coords = shared_detections["boss"] # Obtiene las coordenadas del jefe detectado.
        player_coords = shared_detections["player"] # Obtiene las coordenadas del personaje principal detectado.
        lock.release() # Libera el lock después de acceder a las variables compartidas.

        # Dibujar bounding box alrededor del personaje principal
        for (x1, y1, x2, y2) in player_coords:
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2) # Dibuja un rectángulo verde alrededor del personaje principal.
            cv2.putText(frame_copy, "character", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2) # Añade la etiqueta "character" sobre el bounding box.

        # Dibujar bounding boxes alrededor de los enemigos (jefe y sus ataques)
        for label, boxes in boss_coords.items():
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2) # Dibuja un rectángulo rojo alrededor de los objetos del jefe.
                cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # Añade la etiqueta de la clase del objeto del jefe sobre el bounding box.

        cv2.imshow("Detección en tiempo real", frame_copy) # Muestra el frame con las detecciones en una ventana llamada "Detección en tiempo real".
        if cv2.waitKey(1) & 0xFF == ord("q"): # Espera 1 milisegundo por una pulsación de tecla. Si se presiona 'q', sale del bucle.
            break
    cv2.destroyAllWindows() # Cierra todas las ventanas de OpenCV al finalizar el hilo.

# Lanzar el hilo de visualización como un demonio para que se cierre automáticamente con el programa principal.
threading.Thread(target=visual_thread, daemon=True).start()

# ------------------- INICIO DE LA DETECCIÓN POR COMANDO DE VOZ ------------------- #
esperar_frase_clave("Inicia") # Llama a la función para esperar a que el usuario diga "Inicia" antes de comenzar la detección.

# ------------------- BUCLE PRINCIPAL DE DETECCIÓN Y EVASIÓN ------------------- #
while True:
    """
    Bucle principal que se ejecuta continuamente para:
    1. Capturar la pantalla del juego.
    2. Realizar la detección de objetos (jefe y personaje principal) utilizando hilos para mejorar el rendimiento.
    3. Compartir los resultados de la detección para la visualización.
    4. Ejecutar la lógica de evasión de ataques basada en las detecciones.
    """
    frame = capture_game_window() # Captura un frame de la ventana del juego.
    if frame is None:
        continue # Si no se pudo capturar el frame, salta a la siguiente iteración del bucle.

    # Utiliza un ThreadPoolExecutor para ejecutar las detecciones de objetos en paralelo, mejorando el rendimiento.
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_boss = executor.submit(detect_objects, frame, model_boss, boss_class_names) # Envía la tarea de detección del jefe al pool de hilos.
        future_player = executor.submit(detect_objects, frame, model_player, [player_class_name]) # Envía la tarea de detección del personaje principal al pool de hilos.
        boss_coordinates = future_boss.result() # Espera a que la tarea de detección del jefe se complete y obtiene los resultados.
        player_coordinates = future_player.result().get(player_class_name, []) # Espera a que la tarea de detección del personaje principal se complete y obtiene los resultados (si el personaje principal es detectado).

    # Compartir los datos de la detección y el frame actual para el hilo de visualización.
    lock.acquire()
    shared_frame = frame
    shared_detections = {"boss": boss_coordinates, "player": player_coordinates}
    lock.release()

    # Ejecuta la lógica para que el personaje principal evada los ataques del jefe basándose en las coordenadas detectadas.
    evade_attack(boss_coordinates, player_coordinates)