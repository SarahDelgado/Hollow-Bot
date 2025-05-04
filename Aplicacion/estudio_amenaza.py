# ------------------- IMPORTACIONES ------------------- #
import os                               # Para lectura de rutas
import numpy as np                      # Para operaciones numéricas y arrays
import cv2                              # Para procesamiento de imágenes
from ultralytics import YOLO            # Para detección con modelos YOLO
import tensorflow as tf                 # Para cargar y usar el modelo Keras
import joblib                           # Para cargar el scaler (normalización)
import pyautogui                        # Para capturar la pantalla del juego
from obtener_rutas import resource_path # Función personalizada para obtener las rutas de los archivos

# ------------------- CARGA DE MODELOS Y SCALER ------------------- #
# Ruta modelo del jefe
ruta_model_boss = resource_path(os.path.join(
    "..", "Entrenamiento", "training_boos_junto_modelo_graficas", "content", "runs", "detect", "train2", "weights", "best.pt"
))
# Modelo YOLO entrenado para detectar al boss y los rayos
modelo_yolo_boss = YOLO(ruta_model_boss)
# Ruta modelo del personaje principal
ruta_model_player = resource_path(os.path.join(
    "..", "Entrenamiento", "training_character_junto_modelos_grafica", "content", "runs", "detect", "train", "weights", "best.pt"
))
# Modelo YOLO entrenado para detectar al personaje principal
modelo_yolo_personaje = YOLO(ruta_model_player)
# Ruta al modelo keras
ruta_modelo_rayo = resource_path(os.path.join("..", "Entrenamiento", "Codigo_keras_detectar_rayo_izq_derch", "mejor_modelo_direccion_rayo.keras"))
# Modelo Keras entrenado para predecir si esquivar hacia izquierda o derecha
modelo_keras = tf.keras.models.load_model(ruta_modelo_rayo)
# Ruta al scaler
ruta_scaler = resource_path(os.path.join("..", "Entrenamiento", "Codigo_keras_detectar_rayo_izq_derch", "scaler_rayo.pkl"))
# Scaler previamente guardado para normalizar vectores de entrada al modelo Keras
scaler = joblib.load(ruta_scaler)


def capturar_juego():
    """
     Captura una imagen de pantalla del juego en tiempo real.

     Utiliza pyautogui para realizar la captura y convierte la imagen
     al formato BGR compatible con OpenCV.

     Returns:
         numpy.ndarray: Imagen capturada en formato BGR (para uso con OpenCV).
     """
    screenshot = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return frame

def calcular_amenaza_desde_imagen():
    """
    Calcula el porcentaje de amenaza actual del jefe (boss) respecto al jugador,
    usando modelos de detección y predicción.

    Returns:
        float: Porcentaje de amenaza entre 0.0 y 100.0.
    """

    DIST_MAX = 300  # Distancia máxima usada para normalización

    def distancia_centros(caja1, caja2):
        """
        Calcula la distancia euclídea entre los centros de dos cajas delimitadoras.

        Args:
            caja1 (list or tuple): Coordenadas de la primera caja en formato [x, y, w, h],
                                   donde (x, y) es la esquina superior izquierda y (w, h) el ancho y alto.
            caja2 (list or tuple): Coordenadas de la segunda caja en el mismo formato [x, y, w, h].

        Returns:
            float: Distancia euclídea entre los centros de las dos cajas.
        """

        x1, y1, w1, h1 = caja1
        x2, y2, w2, h2 = caja2
        cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
        cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
        return np.linalg.norm([cx1 - cx2, cy1 - cy2])

    # ------------------- Proceso de detección ------------------- #
    imagen_frame = capturar_juego()  # Captura pantalla

    # Detección del personaje
    resultado_pj = modelo_yolo_personaje(imagen_frame)[0]
    personaje = None
    for det in resultado_pj.boxes.data:
        x1, y1, x2, y2, conf, cls = det.tolist()
        personaje = [x1, y1, x2 - x1, y2 - y1]  # Convertir a formato [x, y, w, h]
        break  # Solo tomamos la primera detección

    if personaje is None:
        return 0  # No hay personaje en pantalla

    # ------------------- Detección del boos y rayos ------------------- #
    resultado_boss = modelo_yolo_boss(imagen_frame)[0]

    boss = None
    rayos_vert = []
    rayos_horiz = []

    for det in resultado_boss.boxes.data:
        x1, y1, x2, y2, conf, cls = det.tolist()
        label = int(cls)
        caja = [x1, y1, x2 - x1, y2 - y1]

        if label == 0:  # Boss
            boss = caja
        elif label == 1:  # Rayo horizontal
            rayos_horiz.append(caja)
        elif label == 2:  # Rayo vertical
            rayos_vert.append(caja)

    if boss is None:
        boss = [0, 0, 0, 0]  # Si no hay boss detectado

    # ------------------- Predicción de dirección con el modelo Keras ------------------- #
    direccion_predicha = "izquierda"
    if rayos_vert:
        rayo = rayos_vert[0]  # Tomamos el primer rayo vertical detectado

        # Creamos el vector de entrada con coordenadas del personaje y el rayo
        vector = np.array([
            personaje[0], personaje[1],
            personaje[0] + personaje[2], personaje[1] + personaje[3],
            rayo[0], rayo[1],
            rayo[0] + rayo[2], rayo[1] + rayo[3],
        ]).reshape(1, -1)

        # Normalizamos y predecimos
        vector_norm = scaler.transform(vector)
        pred = modelo_keras.predict(vector_norm)[0]

        direccion_predicha = "izquierda" if pred[0] > 0.5 else "derecha"

    # ------------------- Cálculo de distancias normalizadas ------------------- #
    dist_boss = distancia_centros(personaje, boss)
    norm_dist_boss = max(0, 1 - dist_boss / DIST_MAX)

    dist_vert = min([distancia_centros(personaje, r) for r in rayos_vert], default=DIST_MAX)
    norm_dist_vert = max(0, 1 - dist_vert / DIST_MAX)

    dist_horiz = min([distancia_centros(personaje, r) for r in rayos_horiz], default=DIST_MAX)
    norm_dist_horiz = max(0, 1 - dist_horiz / DIST_MAX)

    # ------------------- Castigo por dirección incorrecta ------------------- #
    x_pj = personaje[0]
    castigo_direccion = 0
    for r in rayos_vert:
        if direccion_predicha == 'izquierda' and r[0] < x_pj:
            castigo_direccion = 1
        elif direccion_predicha == 'derecha' and r[0] > x_pj:
            castigo_direccion = 1

    # ------------------- Cálculo final de la amenaza ------------------- #
    amenaza = (
        0.4 * norm_dist_vert +
        0.3 * norm_dist_horiz +
        0.1 * norm_dist_boss +
        0.2 * castigo_direccion
    )
    return round(min(1.0, amenaza) * 100, 2)
