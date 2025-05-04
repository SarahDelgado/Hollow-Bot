import speech_recognition as sr
import tkinter as tk  # Necesario para acceder a los elementos de la GUI
import time  # Para pausas si el micrófono no está disponible

def esperar_frase_clave(root, voice_message_label, iniciar_bot_callback, frase_esperada="Inicia"):
    """
    Espera a que el usuario diga la frase clave y actualiza el label de voz.
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
            with sr.Microphone() as source:  # Envuelve la creación del Microphone
                recognizer.adjust_for_ambient_noise(source)
                voice_message_label.config(text=f"🎤 Esperando que digas: '{frase_esperada}'...")
                root.update()
                audio = recognizer.listen(source, timeout=5)  # Escucha por hasta 5 segundos
                texto = recognizer.recognize_google(audio, language='es-ES')
                voice_message_label.config(text=f"🗣️ Dijiste: {texto}")
                root.update()

                if frase_esperada.lower() in texto.lower():
                    voice_message_label.config(text="✅ Frase reconocida. Iniciando aplicación...")
                    root.update()
                    root.after(100, iniciar_bot_callback)  # Llama a la función de inicio del bot
                    break  # Salir del bucle
                else:
                    voice_message_label.config(text="❌ Esa no es la frase. Intenta de nuevo.\n")
                    root.update()

        except sr.UnknownValueError:
            voice_message_label.config(text="❗ No se entendió lo que dijiste. Intenta otra vez.\n")
            root.update()
        except sr.RequestError as e:
            voice_message_label.config(text=f"❗ Error con el servicio de reconocimiento: {e}")
            root.update()
            voice_message_label.config(text="⚠️ Error con el servicio de reconocimiento. Intenta de nuevo.")
            root.update()
            time.sleep(2) # Espera un poco antes de reintentar
        except OSError as e:
            if "No Default Audio Input Device Available" in str(e):
                voice_message_label.config(text="🔇 No se detecta micrófono. Conéctalo e intenta de nuevo...")
                root.update()
                time.sleep(5) # Espera a que el usuario conecte el micrófono
            else:
                voice_message_label.config(text=f"🔇 No se detecta micrófono. Conéctalo e intenta de nuevo...")
                root.update()
                break # Sale del bucle si es otro error de audio grave
        except sr.WaitTimeoutError:
            pass # No se escuchó nada en el timeout, vuelve a intentar
        except Exception as e:
            voice_message_label.config(text=f"❗ Error inesperado: {e}")
            root.update()
            break # Sale del bucle por error inesperado