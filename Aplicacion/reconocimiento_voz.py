import speech_recognition as sr
import tkinter as tk
import time

def esperar_frase_clave(root, voice_message_label, iniciar_bot_callback, frase_esperada="Inicia"):
    """
    Espera a que el usuario diga la frase clave y actualiza el label de voz de forma segura para Tkinter.
    Maneja el caso en que el micrófono no está conectado.

    Args:
        root (tk.Tk): La ventana raíz de Tkinter.
        voice_message_label (tk.Label): El label para mostrar mensajes de voz.
        iniciar_bot_callback (callable): La función para iniciar el bot.
        frase_esperada (str): La frase que se espera escuchar (por defecto "Inicia").
    """
    recognizer = sr.Recognizer()

    while True:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)

                def actualizar_label(texto):
                    voice_message_label.config(text=texto)

                root.after(0, lambda: actualizar_label(f"🎤 Esperando que digas: '{frase_esperada}'..."))
                audio = recognizer.listen(source, timeout=5)
                texto = recognizer.recognize_google(audio, language='es-ES')
                root.after(0, lambda: actualizar_label(f"🗣️ Dijiste: {texto}"))

                if frase_esperada.lower() in texto.lower():
                    root.after(0, lambda: actualizar_label("✅ Frase reconocida. Iniciando aplicación..."))
                    root.after(100, iniciar_bot_callback)
                    break
                else:
                    root.after(0, lambda: actualizar_label("❌ Esa no es la frase. Intenta de nuevo.\n"))

        except sr.UnknownValueError:
            root.after(0, lambda: actualizar_label("❗ No se entendió lo que dijiste. Intenta otra vez.\n"))
        except sr.RequestError as e:
            root.after(0, lambda: actualizar_label(f"❗ Error con el servicio de reconocimiento: {e}"))
            root.after(2000, lambda: actualizar_label("⚠️ Error con el servicio de reconocimiento. Intenta de nuevo."))
            time.sleep(2)
        except OSError as e:
            if "No Default Audio Input Device Available" in str(e):
                root.after(0, lambda: actualizar_label("🔇 No se detecta micrófono. Conéctalo e intenta de nuevo..."))
                time.sleep(5)
            else:
                root.after(0, lambda: actualizar_label(f"🔇 Error de audio: {e}"))
                break
        except sr.WaitTimeoutError:
            pass
        except Exception as e:
            root.after(0, lambda: actualizar_label(f"❗ Error inesperado: {e}"))
            break