import os
import subprocess

# De donde a donde se crean los videos
VIDEO_DIR = "./06-11-2025"
OUTPUT_FILE = "jueves-06-11-opt.avi"  
LIST_FILE = "videos.txt"

# Configuracion de optimizacion 
FPS_OBJETIVO = 12
ANCHO = 1200  
ALTO = int(ANCHO * 3 / 4)  


def main():
    # Obtengo todos los cortes de video 
    archivos = sorted(
        [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(".avi")]
    )

    if not archivos:
        print("No se encontraron archivos AVI.")
        return

    # Crear lista para concatenar
    with open(LIST_FILE, "w") as f:
        for nombre in archivos:
            ruta = os.path.join(VIDEO_DIR, nombre)
            f.write(f"file '{os.path.abspath(ruta)}'\n")

    print(f"Se encontraron {len(archivos)} videos. Procesando...")


   
    cmd = [
        "ffmpeg", # nombre de la herramienta
        "-f", "concat", # formato de concatenacion
        "-safe", "0", # seguridad para rutas
        "-i", LIST_FILE, # archivo de entrada
        "-r", str(FPS_OBJETIVO), # fps objetivo
        "-vf", f"scale={ANCHO}:{ALTO}",                     
        "-c:v", "mpeg4", # codec de video              
        "-qscale:v", "5", # calidad de video                
        "-y", 
        OUTPUT_FILE
    ]

    subprocess.run(cmd) # ejecuto el comando

    print(f"\n✅ Video optimizado guardado como {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
