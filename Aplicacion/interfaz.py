# ------------------- IMPORTACIONES ------------------- #
import tkinter as tk                     # Librería para interfaces gráficas
from PIL import Image, ImageTk           # Para manipular y mostrar imágenes
import os                                # Para manejar rutas de archivos
import threading                         # Permite ejecutar funciones en segundo plano (multihilo)
import subprocess                        # Para lanzar procesos externos (el bot)
import sys                               # Para acceder a la ruta del intérprete de Python
import time                              # Para pausas temporales
import estudio_amenaza                   # Módulo personalizado para calcular la amenaza del jefe
import reconocimiento_voz                # Módulo personalizado para detectar la voz del usuario
from obtener_rutas import resource_path  # Función personalizada para obtener las rutas de los archivos


class HollowBotGUI:
    """
        Clase que define la interfaz gráfica de Hollow-Bot.
        Permite iniciar, detener el bot, cambiar el modo de visualización
        y muestra el estado y el porcentaje de amenaza. También integra
        la funcionalidad de reconocimiento de voz para iniciar la aplicación.
        """
    def __init__(self, root):
        """
        Inicializa la ventana principal de la interfaz gráfica.

        Args:
            root (tk.Tk): La ventana raíz de Tkinter.
        """
        # Inicializa la ventana principal
        self.root = root
        self.root.title("Hollow-Bot")
        self.root.geometry("700x700")
        self.root.resizable(False, False)

        # Variables de estado
        self.bot_thread = None            # Hilo que ejecuta el bot
        self.reconocimiento_thread = None # Hilo para el reconocimiento de voz
        self.bot_process = None           # Proceso del bot
        self.running = False              # Si el bot está activo
        self.modo_oscuro = False          # Modo de visualización (claro u oscuro)
        self.juego_iniciado = False       # Flag para controlar si el juego se ha iniciado

        # ------------------- FONDO DE PANTALLA ------------------- #
        image_path = resource_path(os.path.join("img", "hollowbot_oscuro.jpg"))
        bg_image = Image.open(image_path).resize((700, 700))  # Redimensiona imagen
        self.bg_photo = ImageTk.PhotoImage(bg_image)
        # Coloca la imagen como fondo
        self.background_label = tk.Label(root, image=self.bg_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # ------------------- BOTONES PRINCIPALES ------------------- #
        self.start_button = tk.Button(
            root, text="Iniciar bot",
            font=("Helvetica", 12, "bold"),
            bg="#4CAF50", fg="white",  # Verde
            activebackground="#45a049", activeforeground="white",
            relief="raised", bd=4,
            command=self.iniciar_reconocimiento
        )
        self.stop_button = tk.Button(
            root, text="Detener bot",
            font=("Helvetica", 12, "bold"),
            bg="#f44336", fg="white",  # Rojo
            activebackground="#da190b", activeforeground="white",
            relief="raised", bd=4,
            command=self.detener_bot,
            state=tk.DISABLED
        )
        # Posiciona los botones en pantalla
        self.start_button.place(relx=0.3, rely=0.85, anchor=tk.CENTER)
        self.stop_button.place(relx=0.7, rely=0.85, anchor=tk.CENTER)

        # ------------------- LABEL DE MENSAJES DE VOZ ------------------- #
        self.voice_message_label = tk.Label(
            root, text="",
            font=("Segoe UI", 8, "bold"),
            bg="#ffffff", fg="#333333",
            padx=1, pady=1,
            relief="groove", bd=2
        )

        # ------------------- LABEL DE ESTADO ------------------- #
        self.status_label = tk.Label(
            root, text="Inactivo",
            font=("Segoe UI", 12, "bold"),
            bg="#ffffff", fg="#333333",
            padx=15, pady=8,
            relief="ridge", bd=2
        )
        # Posiciona la label en pantalla
        self.status_label.place(relx=0.5, rely=0.93, anchor=tk.CENTER)

        # ------------------- LABEL DE AMENAZA ------------------- #
        self.threat_label = tk.Label(
            root, text="Porcentaje de amenaza:\n0%",
            font=("Segoe UI", 12, "bold"),
            bg="#ffffff", fg="#333333",
            padx=5, pady=5,
            relief="groove", bd=2
        )
        # Posiciona la label en pantalla
        self.threat_label.place(relx=0.85, rely=0.95, anchor=tk.CENTER)

        # ------------------- BOTÓN DE CAMBIO DE MODO ------------------- #
        self.toggle_mode_button = tk.Button(
            root, text="☀Cambiar modo",
            font=("Helvetica", 12, "bold"),
            bg="#ffffff", fg="black",
            activebackground="#ffffff", activeforeground="white",
            relief="raised", bd=4,
            command=self.cambiar_modo
        )
        # Posiciona el botón en pantalla
        self.toggle_mode_button.place(relx=0.02, rely=0.02)


    def run_bot(self):
        """
        Ejecuta el bot en un hilo. Mientras está corriendo, actualiza el porcentaje
        de amenaza en la interfaz en tiempo real.
        """
        self.running = True
        self.juego_iniciado = True
        self.status_label.config(text="Ejecutando...", bg="#d1ffd6", fg="#006400")
        self.stop_button.config(state=tk.NORMAL) # Habilita el botón de detener

        # Lanza el proceso del bot
        bot_script_path = resource_path(os.path.join("Aplicacion", "logicajuego_sin_logs.py"))
        self.bot_process = subprocess.Popen([sys.executable, bot_script_path])

        # Actualiza continuamente el porcentaje de amenaza
        while self.running:
            amenaza = estudio_amenaza.calcular_amenaza_desde_imagen()
            self.actualizar_amenaza(amenaza)
            try:
                self.root.update()
            except tk.TclError:
                # La ventana se ha cerrado
                break
            time.sleep(0.001)

        # Espera que el proceso termine y actualiza estado
        if self.bot_process:
            self.bot_process.wait()
        self.status_label.config(text="Inactivo", bg="#ffffff", fg="#333333")
        self.running = False
        self.juego_iniciado = False


    def iniciar_bot(self):
        """
        Inicia el bot en un hilo separado si no se está ejecutando y el juego no se ha iniciado.
        """
        if not self.running and not self.juego_iniciado:
            self.bot_thread = threading.Thread(target=self.run_bot)
            self.bot_thread.start()


    def iniciar_reconocimiento_voz_en_hilo(self):
        if self.reconocimiento_thread is None or not self.reconocimiento_thread.is_alive():
            self.reconocimiento_thread = threading.Thread(target=reconocimiento_voz.esperar_frase_clave,
                                                          args=(self.root, self.voice_message_label, self.iniciar_bot))
            self.reconocimiento_thread.daemon = True
            self.reconocimiento_thread.start()
            print("Hilo de reconocimiento de voz iniciado.")
        else:
            print("El hilo de reconocimiento de voz ya está en ejecución.")


    def iniciar_reconocimiento(self):
        """
        Inicia el hilo para el reconocimiento de voz utilizando la función del módulo separado.
        """
        if self.reconocimiento_thread is None or not self.reconocimiento_thread.is_alive():
            self.reconocimiento_thread = self.iniciar_reconocimiento_voz_en_hilo()
            self.start_button.config(state=tk.DISABLED) # Deshabilita el botón de inicio
            self.stop_button.config(state=tk.NORMAL)   # Habilita el botón de detener
            self.status_label.config(text="Esperando activación por voz...", bg="#ffffff", fg="#333333")
            self.voice_message_label.place(relx=0.2, rely=0.98, anchor=tk.CENTER) # Coloca la label de reconocimiento de voz


    def detener_bot(self):
        """
        Detiene el bot si está corriendo o el reconocimiento de voz si está activo.
        Restablece las labels al estado inicial.
        """
        # Detener el bot si está en ejecución
        if self.running and self.bot_process:
            self.bot_process.terminate()
            self.running = False

        # Detener el hilo de reconocimiento de voz si está activo y el juego no se ha iniciado
        if self.reconocimiento_thread and self.reconocimiento_thread.is_alive() and not self.juego_iniciado:
            import ctypes
            thread_id = self.reconocimiento_thread.ident
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id),
                                                           ctypes.py_object(SystemExit))
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, None)
                print("Error al intentar terminar el hilo de reconocimiento.")
            self.reconocimiento_thread = None # Limpia la referencia al hilo

        # Restablecer las labels al estado inicial
        self.status_label.config(text="Inactivo", bg="#ffffff", fg="#333333")
        self.voice_message_label.config(text="")
        self.threat_label.config(
            text="Porcentaje de amenaza:\n0%",
            font=("Segoe UI", 12, "bold"),
            bg="#ffffff", fg="#333333",
            padx=5, pady=5,
            relief="groove", bd=2
        )
        self.start_button.config(state=tk.NORMAL) # Vuelve a habilitar el botón de inicio
        self.stop_button.config(state=tk.DISABLED) # Deshabilita el botón de detener
        self.voice_message_label.place_forget() # Oculta la label del reconocimiento de voz


    def actualizar_amenaza(self, porcentaje):
        """
        Actualiza el label del porcentaje de amenaza con colores según nivel.

        Args:
            porcentaje (float): Porcentaje de amenaza calculado, en el rango de 0 a 100.
        """
        porcentaje = max(0, min(100, porcentaje))  # Asegura que el porcentaje esté en [0, 100]
        self.threat_label.config(text=f"Porcentaje de amenaza:\n{porcentaje}%")

        # Cambia colores según nivel de amenaza
        if porcentaje < 30:
            self.threat_label.config(bg="#d4edda", fg="#155724")  # Verde
        elif porcentaje < 70:
            self.threat_label.config(bg="#fff3cd", fg="#856404")  # Amarillo
        else:
            self.threat_label.config(bg="#f8d7da", fg="#721c24")  # Rojo


    def cambiar_modo(self):
        """
        Cambia entre modo claro y modo oscuro cambiando el fondo y estilos.
        """
        self.modo_oscuro = not self.modo_oscuro

        # Selecciona imagen según modo
        nuevo_archivo = "hollowbot_claro.jpg" if self.modo_oscuro else "hollowbot_oscuro.jpg"
        image_path = resource_path(os.path.join("img", nuevo_archivo))
        nueva_imagen = Image.open(image_path).resize((700, 700))
        self.bg_photo = ImageTk.PhotoImage(nueva_imagen)
        self.background_label.config(image=self.bg_photo)

        # Cambiar el texto del botón 🌙 <-> ☀️
        nuevo_texto = "🌙Cambiar modo" if self.modo_oscuro else "☀Cambiar modo"
        self.toggle_mode_button.config(text=nuevo_texto)

        # Cambiar colores del botón de modo y label de estado
        if self.modo_oscuro:
            # Modo claro (fondo claro, elementos oscuros)
            self.toggle_mode_button.config(bg="#eeeeee", fg="#222222", activebackground="#dddddd")
            self.status_label.config(bg="#eeeeee", fg="#111111")
            self.voice_message_label.config(bg="#eeeeee", fg="#111111")
            self.threat_label.config(bg="#eeeeee", fg="#111111")
        else:
            # Modo oscuro (fondo oscuro, elementos claros)
            self.toggle_mode_button.config(bg="#ffffff", fg="black", activebackground="#ffffff")
            self.status_label.config(bg="#ffffff", fg="#333333")
            self.voice_message_label.config(bg="#ffffff", fg="#333333")
            self.threat_label.config(bg="#ffffff", fg="#333333")


if __name__ == "__main__":
    root = tk.Tk()
    app = HollowBotGUI(root)
    root.mainloop()