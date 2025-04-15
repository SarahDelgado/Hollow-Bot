import tkinter as tk
from PIL import Image, ImageTk
import threading
import os
import subprocess
import sys
import time
import estudio_amenaza

class HollowBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hollow-Bot")
        self.root.geometry("700x700")
        self.root.resizable(False, False)
        self.bot_thread = None
        self.bot_process = None
        self.running = False

        # Fondo con imagen
        base_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(base_dir, "..", "assets", "hollowbot_oscuro.jpg")
        bg_image = Image.open(image_path)
        bg_image = bg_image.resize((700, 700))
        self.bg_photo = ImageTk.PhotoImage(bg_image)

        # Imagen de fondo
        self.background_label = tk.Label(root, image=self.bg_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Botones alineados horizontalmente
        self.start_button = tk.Button(
            root,
            text="Iniciar bot",
            font=("Helvetica", 12, "bold"),
            bg="#4CAF50",  # Verde
            fg="white",
            activebackground="#45a049",
            activeforeground="white",
            relief="raised",
            bd=4,
            command=self.iniciar_bot
        )

        self.stop_button = tk.Button(
            root,
            text="Detener bot",
            font=("Helvetica", 12, "bold"),
            bg="#f44336",  # Rojo
            fg="white",
            activebackground="#da190b",
            activeforeground="white",
            relief="raised",
            bd=4,
            command=self.detener_bot
        )
        self.start_button.place(relx=0.3, rely=0.85, anchor=tk.CENTER)
        self.stop_button.place(relx=0.7, rely=0.85, anchor=tk.CENTER)

        # Label de estado
        self.status_label = tk.Label(
            root,
            text="Inactivo",
            font=("Segoe UI", 12, "bold"),
            bg="#ffffff",
            fg="#333333",
            padx=15,
            pady=8,
            relief="ridge",
            bd=2
        )
        self.status_label.place(relx=0.5, rely=0.95, anchor=tk.CENTER)

        # Label de porcentaje
        self.threat_label = tk.Label(
            root,
            text="Porcentaje de amenaza:\n0%",
            font=("Segoe UI", 12, "bold"),
            bg="#ffffff",
            fg="#b22222",
            padx=5,
            pady=5,
            relief="groove",
            bd=2
        )
        self.threat_label.place(relx=0.85, rely=0.95, anchor=tk.CENTER)

    def run_bot(self):
        self.running = True
        self.status_label.config(text="Ejecutando...", bg="#d1ffd6", fg="#006400")

        # Lanza el proceso del bot
        self.bot_process = subprocess.Popen([sys.executable, "locajuegonewv3.py"])

        while self.running:
            amenaza = estudio_amenaza.calcular_amenaza_desde_imagen()
            self.actualizar_amenaza(amenaza)
            self.root.update()
            time.sleep(0.5)  # puedes ajustar la frecuencia de análisis aquí

        # Espera que termine
        self.bot_process.wait()
        self.status_label.config(text="Inactivo", bg="#ffffff", fg="#333333")
        self.running = False

    def iniciar_bot(self):
        if not self.running:
            self.bot_thread = threading.Thread(target=self.run_bot)
            self.bot_thread.start()

    def detener_bot(self):
        if self.running and self.bot_process:
            self.bot_process.terminate()  # Detiene el proceso del bot
            self.running = False
            self.status_label.config(text="Detenido", bg="#ffe6e6", fg="#8b0000")
            self.threat_label = tk.Label(
                root,
                text="Porcentaje de amenaza:\n0%",
                font=("Segoe UI", 12, "bold"),
                bg="#ffffff",
                fg="#b22222",
                padx=5,
                pady=5,
                relief="groove",
                bd=2
            )

    def actualizar_amenaza(self, porcentaje):
        porcentaje = max(0, min(100, porcentaje))  # Asegura que esté entre 0 y 100
        self.threat_label.config(text=f"Porcentaje de amenaza:\n{porcentaje}%")

        # Colores visuales según amenaza
        if porcentaje < 30:
            self.threat_label.config(bg="#d4edda", fg="#155724")  # Verde
        elif porcentaje < 70:
            self.threat_label.config(bg="#fff3cd", fg="#856404")  # Amarillo
        else:
            self.threat_label.config(bg="#f8d7da", fg="#721c24")  # Rojo

if __name__ == "__main__":
    root = tk.Tk()
    app = HollowBotGUI(root)
    root.mainloop()