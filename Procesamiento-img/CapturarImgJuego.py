import mss
import numpy as np
import cv2
import time
import pygetwindow as gw
import os

# Carpeta para guardar capturas
output_folder = "dataset/frames"
os.makedirs(output_folder, exist_ok=True)  # Crear la carpeta si no existe

frame_count = 0
game_window_title = "Hollow Knight"  # Ajusta al nombre real de la ventana

# Configurar la tasa de FPS
FPS = 30
frame_time = 1 / FPS  # Tiempo entre capturas (en segundos)

with mss.mss() as sct:
    last_time = time.time()  # Tiempo inicial

    while True:
        # Obtener la ventana del juego
        win = gw.getWindowsWithTitle(game_window_title)
        if win:
            win = win[0]  # Tomamos la primera coincidencia
            monitor = {
                "top": win.top,
                "left": win.left,
                "width": win.width,
                "height": win.height
            }

            # Captura la pantalla de la ventana del juego
            img = sct.grab(monitor)
            img = np.array(img)[:, :, :3]

            # Guardar la imagen
            filename = f"{output_folder}/frame_{frame_count:06d}.png"  # Formato con ceros para orden correcto
            cv2.imwrite(filename, img)
            frame_count += 1

            print(f"Imagen guardada: {filename}")
        else:
            print("No se encontró la ventana del juego")

        # Controlar el tiempo de captura para mantener 30 FPS
        elapsed_time = time.time() - last_time
        sleep_time = max(0, frame_time - elapsed_time)  # Asegurar que no sea negativo
        time.sleep(sleep_time)
        last_time = time.time()

