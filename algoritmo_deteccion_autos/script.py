import cv2
import numpy as np
import time
import datetime
import os  # <--- NUEVO: Para verificar archivos
import csv
from ultralytics import YOLO
from norfair import Detection, Tracker
from utils import leer_timestamp, euclidean_distance

# ======================
# CONFIGURACIÓN
# ======================
VIDEO_PATH = "/home/adrian/Escritorio/Universidad/tesis/videos/videoIT/03-11-2025/lunes-03-11.avi"
START_TIME = "2025-11-03T08:24:53"
CSV_NAME = "resultados_videoIT_03-11.csv"
DIRECCION = 4
DIA_SEMANA = 1 # 0=Domingo, 1=Lunes, ..., 6=Sábado

GREEN_DURATION = 33    # segundos
RED_DURATION = 118     # segundos totales de rojo
RED_BUFFER = 3         # Segundos de antelación
                       # Salto = 118 - 3 = 115 segundos.

TARGET_CLASSES = {"car", "truck"}

# ======================
# GESTIÓN DEL CSV (Header)
# ======================
# Verificamos si el archivo existe. Si NO existe, lo creamos y ponemos encabezados.
if not os.path.exists(CSV_NAME):
    with open(CSV_NAME, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Total_Vehiculos", "Tiempo_Medio_s", "Ocupacion_%", "Hora_Inicio", "Hora_Fin", "Dia_Semana", "Direccion"])
    print(f"📁 Archivo {CSV_NAME} creado con encabezados.")
else:
    print(f"📁 Archivo {CSV_NAME} detectado. Se añadirán nuevas filas.")

# Configuración ROI
PX1, PY1, PX2, PY2 = 0, 730, 1200, 890
ROI = np.s_[PY1:PY2, PX1:PX2]

# Línea inclinada (RELATIVA A LA ROI)
lineal_start = (300, 59)
lineal_end   = (0, 100)

# ======================
# MODELO Y TRACKER
# ======================
print("Cargando modelo YOLO...")
model = YOLO("yolov8n.pt")
tracker = Tracker(
    distance_function=euclidean_distance,
    distance_threshold=30
)

cap = cv2.VideoCapture(VIDEO_PATH)
FPS = cap.get(cv2.CAP_PROP_FPS)
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ======================
# ESTADO GLOBAL Y MÉTRICAS
# ======================
estado = "ESPERANDO"
inicio_verde_time = None
ciclo = 0

historial = {}
contados = set()

tiempos_cruce = []
tiempo_ocupado = 0.0
frame_anterior_con_auto = False
ultimo_tiempo_frame = None

# Variables para estimación de tiempo
start_process_time = time.time()

# ======================
# FUNCIONES AUXILIARES
# ======================

def guardar_resultados_csv(total_vehiculos, tiempo_medio, ocupacion, hora_inicio, hora_fin, dia_semana, direccion):
    # Usamos mode='a' (append) para agregar al final sin borrar lo anterior
    with open(CSV_NAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            total_vehiculos, 
            f"{tiempo_medio:.2f}", 
            f"{ocupacion:.2f}", 
            hora_inicio, 
            hora_fin, 
            dia_semana, 
            direccion
        ])
    print("💾 Datos guardados en CSV.")

def get_frame_time(cap):
    frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
    return leer_timestamp(frame_idx / FPS, START_TIME)

def yolo_to_norfair(results):
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            detections.append(
                Detection(points=np.array([cx, cy]), scores=np.array([conf]))
            )
    return detections

def side_of_line(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

def cruzo_linea_segmento(prev_pt, curr_pt, a, b, margen=5):
    s1 = side_of_line(prev_pt, a, b)
    s2 = side_of_line(curr_pt, a, b)
    if s1 == 0 or s2 == 0: return False
    return (s1 > margen and s2 < -margen) or (s1 < -margen and s2 > margen)

def imprimir_y_guardar(ciclo, total, tiempos, ocupacion, t_inicio, t_fin):
    # Calcular tiempo medio
    if len(tiempos) > 1:
        intervalos = [(tiempos[i] - tiempos[i - 1]).total_seconds() for i in range(1, len(tiempos))]
        tiempo_medio = sum(intervalos) / len(intervalos)
    else:
        tiempo_medio = 0.0
    
    # Imprimir en consola
    print(f"\n✅ [RESULTADOS CICLO {ciclo}]")
    print(f"   Vehículos: {total} | Ocupación: {ocupacion:.2f}% | Intervalo Medio: {tiempo_medio:.2f}s")
    
    # Preparar datos para CSV
    # Asumimos que t_inicio y t_fin son objetos datetime
    hora_ini_str = t_inicio.strftime("%H:%M:%S")
    hora_fin_str = t_fin.strftime("%H:%M:%S")
  

    # Guardar en CSV
    guardar_resultados_csv(
        total_vehiculos=total,
        tiempo_medio=tiempo_medio,
        ocupacion=ocupacion,
        hora_inicio=hora_ini_str,
        hora_fin=hora_fin_str,
        dia_semana=DIA_SEMANA,
        direccion=DIRECCION
    )

def mostrar_progreso(frame_actual):
    elapsed = time.time() - start_process_time
    progreso = frame_actual / TOTAL_FRAMES
    
    if progreso > 0:
        total_estimado = elapsed / progreso
        restante = total_estimado - elapsed
        eta_str = str(datetime.timedelta(seconds=int(restante)))
    else:
        eta_str = "Calculando..."

    progreso_pct = progreso * 100
    print(f"\r⏳ Progreso: {progreso_pct:.2f}% | ETA: {eta_str} | Estado: {estado} | Ciclos: {ciclo}", end="")

# ======================
# LOOP PRINCIPAL (HEADLESS)
# ======================
print(f"Iniciando procesamiento rápido. Video: {TOTAL_FRAMES} frames.")

while True:
    ret, frame_full = cap.read()
    if not ret:
        break

    current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
    frame_time = get_frame_time(cap)
    
    # Recorte ROI
    frame = frame_full[ROI]

    # YOLO & Tracker (Sin verbose para no llenar consola)
    results = model(frame, verbose=False)
    detections = yolo_to_norfair(results)
    tracked_objects = tracker.update(detections=detections)

    hubo_cruce = False
    hay_auto_en_frame = False

    for obj in tracked_objects:
        cx, cy = obj.estimate[0]
        track_id = obj.id
        hay_auto_en_frame = True

        if track_id not in historial:
            historial[track_id] = (cx, cy)
            continue

        prev_pt = historial[track_id]
        curr_pt = (cx, cy)

        # Detectar cruce
        if cruzo_linea_segmento(prev_pt, curr_pt, lineal_start, lineal_end):
            hubo_cruce = True
            if estado == "VERDE" and track_id not in contados:
                contados.add(track_id)
                tiempos_cruce.append(frame_time)

        historial[track_id] = (cx, cy)

    # Lógica Ocupación
    if estado == "VERDE" and ultimo_tiempo_frame is not None:
        delta = (frame_time - ultimo_tiempo_frame).total_seconds()
        if delta < 1.0 and frame_anterior_con_auto:
            tiempo_ocupado += delta

    frame_anterior_con_auto = hay_auto_en_frame
    ultimo_tiempo_frame = frame_time

    # ======================
    # LÓGICA DE ESTADOS
    # ======================
    if estado == "ESPERANDO" and hubo_cruce:
        estado = "VERDE"
        inicio_verde_time = frame_time
        ciclo += 1
        contados.clear()
        tiempos_cruce.clear()
        tiempo_ocupado = 0.0
        print("\n🟢 Semáforo VERDE - Iniciando conteo...")

    elif estado == "VERDE":
        if (frame_time - inicio_verde_time).total_seconds() >= GREEN_DURATION:
            ocupacion = (tiempo_ocupado / GREEN_DURATION) * 100
            
            # --- MODIFICADO: Llamamos a la función que imprime Y guarda ---
            imprimir_y_guardar(
                ciclo=ciclo,
                total=len(contados),
                tiempos=tiempos_cruce,
                ocupacion=ocupacion,
                t_inicio=inicio_verde_time,
                t_fin=frame_time
            )

            # Salto temporal
            segundos_a_saltar = max(0, RED_DURATION - RED_BUFFER)
            frames_saltar = int(segundos_a_saltar * FPS)
            
            print(f"🔴 Fin VERDE. Saltando {segundos_a_saltar}s de video...")
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frames_saltar)

            estado = "ESPERANDO"
            historial.clear()
            ultimo_tiempo_frame = None
            continue

    # Mostrar progreso cada 30 frames para no saturar consola
    if int(current_frame_idx) % 30 == 0:
        mostrar_progreso(current_frame_idx)

print("\n\n✅ Procesamiento finalizado correctamente.")
cap.release()