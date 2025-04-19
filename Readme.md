# Hollow-Bot

## Descripción del Proyecto

Hollow-Bot es un proyecto de inteligencia artificial que tiene como objetivo desarrollar un bot capaz de derrotar a los jefes del Panteón en el videojuego Hollow Knight. El bot utiliza técnicas de aprendizaje automático para identificar los personajes del juego y así poder realizar las acciones del personaje del jugador en consecuencia. El programa consta de una interfaz de activación y utiliza la librería PyAutoGUI para realizar los inputs en el juego.


## Instalación y Uso

Para utilizar el bot, sigue estos pasos:

1.  **Descarga el archivo ejecutable (Hollow-Bot.exe):**
    * Ve a la carpeta `dist/` dentro de este repositorio.
    * Descarga el archivo `.exe` que contiene el bot compilado.

2.  **Ejecuta el bot:**
    * Haz doble clic en el archivo `.exe` descargado para ejecutar el bot.

3.  **Configura el entorno de juego:**
    * Asegúrate de que el juego Hollow Knight esté en ejecución.

4.  **Activa el bot:**
    * Presiona el botón "Iniciar bot".

**Nota:**
* El uso de bots en videojuegos puede estar en contra de los términos de servicio del juego. Úsalo bajo tu propia responsabilidad.


## Tecnologías Utilizadas

* **Lenguaje de programación:** Python 3.11

* **Librerías de IA:**
    * Keras 3.9.2
    * NumPy 2.1.1
    * Scikit-Learn 1.6.1
    * Scipy 1.15.2
    * TensorFlow 2.19.0
    * TensorBoard 2.19.0
    * Torch 2.6.0
    * Torchvision 0.21.0
    * Ultralytics 8.3.91 (YOLOv8)
    * Pandas 2.2.3    

* **Visualización de datos:**
    * Matplotlib 3.10.1
    * Seaborn 0.13.2

* **Automatización de la entrada:**
    * PyAutoGUI 0.9.54
      
* **Captura de imágenes y de la ventana del juego:**
    * PyGetWindow 0.0.9
      
* **Interfaz de usuario (GUI):**
    * Tkinter

* **Otras librerías importantes:**
    * OpenCV 4.11.0.86
    * PyYAML 6.0.2

* **Herramienta para la creación del instalador:**
    * Pyinstaller 6.12.0
 
* **Control de versiones y colaboración:**
    * GitHub

**Notas:**
* Asegúrate de que la versión de Python sea correcta.
* Instala las librerías del archivo `requirements.txt` con el comando `pip install -r requirements.txt` para asegurarte de tener todas las necesarias.
* Se recomienda utilizar un entorno virtual para instalar estas dependencias para evitar conflictos con otras instalaciones de Python.


## Estructura del Proyecto

La estructura principal del proyecto es la siguiente:

Hollow-Bot/
├── Aplicacion/           # Scripts de la interfaz y logica principal del bot
├── dataset_boss/         # Imágenes de entrenamiento y validación del jefe
├── dataset_character/    # Imágenes de entrenamiento y validación del personaje principal
├── dist/                 # Ejecutable del bot
├── Entrenamiento/        # Colabs con los entrenamientos de los modelos y los resultados de los mismos
├── img/                  # Imágenes para la interfaz de usuario
├── Procesamiento-img/    # Scripts para el procesamiento de las imagenes de entrenamiento
├── LICENSE               # Licencia de este proyecto
├── README.md             # Este archivo
└── requirements.txt      # Dependencias de Python


## Resultados

Los resultados del entrenamiento del modelo se detallan en la documentación del proyecto y se visualizan a través de diversas herramientas. Se incluyen métricas de rendimiento como precisión, recall y mAP, así como ejemplos visuales de las detecciones del modelo.


## Contribuciones

* Sarah Delgado Martín
* Alejandro Fernández Morales


## Licencia

Este proyecto se distribuye bajo la licencia establecida en el archivo LICENSE.

## Contacto

Para cualquier pregunta o sugerencia, por favor contacte con:
  * sdelmar514@ieszaidinvergeles.org
  * afermor @ieszaidinvergeles.org.
