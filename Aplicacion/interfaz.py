# ------------------- IMPORTACIONES ------------------- #
import tkinter as tk  # Librería para interfaces gráficas
from PIL import Image, ImageTk  # Para manipular y mostrar imágenes
import threading  # Permite ejecutar funciones en segundo plano (multihilo)
import os  # Para manejar rutas de archivos
import subprocess  # Para lanzar procesos externos (el bot)
import sys  # Para acceder a la ruta del intérprete de Python
import time  # Para pausas temporales
import estudio_amenaza  # Módulo personalizado para calcular la amenaza del jefe


class HollowBotGUI:
    def __init__(self, root):
        # Inicializa la ventana principal
        self.root = root
        self.root.title("Hollow-Bot")
        self.root.geometry("700x700")
        self.root.resizable(False, False)

        # Variables de estado
        self.bot_thread = None            # Hilo que ejecuta el bot
        self.bot_process = None           # Proceso del bot
        self.running = False              # Si el bot está activo
        self.modo_oscuro = False          # Modo de visualización (claro u oscuro)
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # ------------------- FONDO DE PANTALLA ------------------- #
        image_path = os.path.join(self.base_dir, "..", "assets", "hollowbot_oscuro.jpg")
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
            command=self.iniciar_bot
        )
        self.stop_button = tk.Button(
            root, text="Detener bot",
            font=("Helvetica", 12, "bold"),
            bg="#f44336", fg="white",  # Rojo
            activebackground="#da190b", activeforeground="white",
            relief="raised", bd=4,
            command=self.detener_bot
        )
        # Posiciona los botones en pantalla
        self.start_button.place(relx=0.3, rely=0.85, anchor=tk.CENTER)
        self.stop_button.place(relx=0.7, rely=0.85, anchor=tk.CENTER)

        # ------------------- LABEL DE ESTADO ------------------- #
        self.status_label = tk.Label(
            root, text="Inactivo",
            font=("Segoe UI", 12, "bold"),
            bg="#ffffff", fg="#333333",
            padx=15, pady=8,
            relief="ridge", bd=2
        )
        # Posiciona la label en pantalla
        self.status_label.place(relx=0.5, rely=0.95, anchor=tk.CENTER)

        # ------------------- LABEL DE AMENAZA ------------------- #
        self.threat_label = tk.Label(
            root, text="Porcentaje de amenaza:\n0%",
            font=("Segoe UI", 12, "bold"),
            bg="#ffffff", fg="#b22222",
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
        self.status_label.config(text="Ejecutando...", bg="#d1ffd6", fg="#006400")

        # Lanza el proceso del bot
        self.bot_process = subprocess.Popen([sys.executable, "locajuegonewv3.py"])

        # Actualiza continuamente el porcentaje de amenaza
        while self.running:
            amenaza = estudio_amenaza.calcular_amenaza_desde_imagen()
            self.actualizar_amenaza(amenaza)
            self.root.update()
            time.sleep(0.1)

        # Espera que el proceso termine y actualiza estado
        self.bot_process.wait()
        self.status_label.config(text="Inactivo", bg="#ffffff", fg="#333333")
        self.running = False

    
    def iniciar_bot(self):
        """
        Inicia el bot en un hilo separado si no se está ejecutando.
        """
        if not self.running:
            self.bot_thread = threading.Thread(target=self.run_bot)
            self.bot_thread.start()

    
    def detener_bot(self):
        """
        Detiene el bot si está corriendo, termina el proceso y actualiza la interfaz.
        """
        if self.running and self.bot_process:
            self.bot_process.terminate()
            self.running = False
            self.status_label.config(text="Detenido", bg="#ffe6e6", fg="#8b0000")

            # Reinicia el label de amenaza visualmente
            self.threat_label = tk.Label(
                root,
                text="Porcentaje de amenaza:\n0%",
                font=("Segoe UI", 12, "bold"),
                bg="#ffffff", fg="#b22222",
                padx=5, pady=5,
                relief="groove", bd=2
            )

    
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
        image_path = os.path.join(self.base_dir, "..", "assets", nuevo_archivo)
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
        else:
            # Modo oscuro (fondo oscuro, elementos claros)
            self.toggle_mode_button.config(bg="#ffffff", fg="black", activebackground="#ffffff")
            self.status_label.config(bg="#ffffff", fg="#333333")


if __name__ == "__main__":
    root = tk.Tk()
    app = HollowBotGUI(root)
    root.mainloop()
