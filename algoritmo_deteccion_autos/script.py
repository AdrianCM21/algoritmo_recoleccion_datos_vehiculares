import cv2
import numpy as np
import time
import datetime
import os
import csv
from ultralytics import YOLO
from norfair import Detection, Tracker
from utils import leer_timestamp, euclidean_distance


VIDEO_PATH = "../videos/videoJP1/06-11-2025/jueves-06-11-opt.avi"

START_TIME = "2025-11-06T08:27:16"

CSV_NAME = "resultados_videoIT_03-11.csv"
DIRECCION = 1
DIA_SEMANA = 4

# Tiempos del semáforo
GREEN_DURATION = 33    
RED_DURATION = 118     
RED_BUFFER = 7        

PX1, PY1, PX2, PY2 = 100, 500, 850, 600

lineal_start = (174, 11)
lineal_end   = (0, 67)

ROI = np.s_[PY1:PY2, PX1:PX2]

TARGET_CLASSES = {"car", "truck", "bus"}

if not os.path.exists(CSV_NAME):
    with open(CSV_NAME, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Total_Vehiculos", "Tiempo_Medio_s", "Ocupacion_Espacial_%", "Hora_Inicio", "Hora_Fin", "Dia_Semana", "Direccion"])

ANCHO_ROI = PX2 - PX1
ALTO_ROI = PY2 - PY1
AREA_TOTAL_ROI = ANCHO_ROI * ALTO_ROI
print(f"ℹ️ Área Total del ROI: {AREA_TOTAL_ROI} píxeles cuadrados.")

# ======================
# 3. FUNCIONES
# ======================
def get_frame_time(cap, fps):
    frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
    return leer_timestamp(frame_idx / fps, START_TIME)

def producto_cruz(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def cruzo_linea_robusto(prev_pt, curr_pt, line_start, line_end):
    """
    Verifica intersección física y filtra por dirección.
    """
    p1, p2 = prev_pt, curr_pt
    q1, q2 = line_start, line_end

    d1 = producto_cruz(q1, q2, p1)
    d2 = producto_cruz(q1, q2, p2)
    d3 = producto_cruz(p1, p2, q1)
    d4 = producto_cruz(p1, p2, q2)

    hay_interseccion = ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))

    if hay_interseccion:
        if d1 < 0: 
            return True
            
    return False

def guardar_resultados_csv(total, t_medio, ocupacion, h_ini, h_fin, dia, dir_v):
    with open(CSV_NAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([total, f"{t_medio:.2f}", f"{ocupacion:.2f}", h_ini, h_fin, dia, dir_v])
    print("💾 Datos guardados en CSV.")

def yolo_to_norfair(results):
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] not in TARGET_CLASSES: continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append(Detection(points=np.array([cx, cy]), scores=np.array([conf])))
    return detections

def mostrar_progreso(current_frame, total_frames, start_time, estado, ciclo):
    elapsed = time.time() - start_time
    progreso = current_frame / total_frames
    if progreso > 0:
        eta = str(datetime.timedelta(seconds=int((elapsed / progreso) - elapsed)))
    else:
        eta = "..."
    print(f"\r⏳ Progreso: {progreso*100:.2f}% | ETA: {eta} | Estado: {estado} | Ciclos: {ciclo}", end="")

# ======================
# 4. INICIALIZACIÓN
# ======================
print("Cargando modelo YOLO...")
model = YOLO("yolov8s.pt")
tracker = Tracker(distance_function=euclidean_distance, distance_threshold=30)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error al abrir video")
    exit()

FPS = cap.get(cv2.CAP_PROP_FPS)
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Variables de Estado Global
estado = "ESPERANDO"
inicio_verde_time = None
ciclo = 0

historial = {}
contados = set()
tiempos_cruce = []

# VARIABLES NUEVAS PARA OCUPACIÓN ESPACIAL
acumulador_porcentaje_ocupacion = 0.0
contador_frames_verde = 0

start_process_time = time.time()

# ======================
# 5. LOOP PRINCIPAL
# ======================
print(f"🚀 Iniciando procesamiento. Video: {TOTAL_FRAMES} frames.")

while True:
    ret, frame_full = cap.read()
    if not ret: break

    current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
    frame_time = get_frame_time(cap, FPS)
    
    # Recorte
    frame = frame_full[ROI]

    # Inferencia
    results = model(frame, verbose=False)
    
    # --- CÁLCULO DE OCUPACIÓN ESPACIAL (FRAME ACTUAL) ---
    area_ocupada_frame_actual = 0
    
    # Iteramos sobre las cajas de YOLO directamente para obtener dimensiones
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] not in TARGET_CLASSES: continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w = x2 - x1
            h = y2 - y1
            area_ocupada_frame_actual += (w * h)
    
    # Si estamos en verde, acumulamos el % de este frame
    if estado == "VERDE":
        pct_frame = (area_ocupada_frame_actual / AREA_TOTAL_ROI) * 100
        if pct_frame > 100: pct_frame = 100.0 # Capar al 100%
        
        acumulador_porcentaje_ocupacion += pct_frame
        contador_frames_verde += 1

    # Tracking y Conteo
    detections = yolo_to_norfair(results)
    tracked_objects = tracker.update(detections=detections)

    hubo_cruce = False

    for obj in tracked_objects:
        cx, cy = obj.estimate[0]
        track_id = obj.id

        if track_id not in historial:
            historial[track_id] = (cx, cy)
            continue

        prev_pt = historial[track_id]
        curr_pt = (cx, cy)

        # Usamos la función robusta
        if cruzo_linea_robusto(prev_pt, curr_pt, lineal_start, lineal_end):
            hubo_cruce = True
            if estado == "VERDE" and track_id not in contados:
                contados.add(track_id)
                tiempos_cruce.append(frame_time)

        historial[track_id] = (cx, cy)

    # ======================
    # LÓGICA DE ESTADOS (SEMÁFORO)
    # ======================
    if estado == "ESPERANDO" and hubo_cruce:
        estado = "VERDE"
        inicio_verde_time = frame_time
        ciclo += 1
        contados.clear()
        tiempos_cruce.clear()
        
        # Reseteamos métricas de ocupación para el nuevo ciclo
        acumulador_porcentaje_ocupacion = 0.0
        contador_frames_verde = 0
        
        print(f"\n🟢 [Ciclo {ciclo}] Iniciando VERDE en {frame_time}")

    elif estado == "VERDE":
        if (frame_time - inicio_verde_time).total_seconds() >= GREEN_DURATION:
            
            # --- CÁLCULO FINAL DE OCUPACIÓN DEL CICLO ---
            if contador_frames_verde > 0:
                ocupacion_final_promedio = acumulador_porcentaje_ocupacion / contador_frames_verde
            else:
                ocupacion_final_promedio = 0.0
            
            # Cálculo Tiempo Medio entre vehículos
            t_medio = 0.0
            if len(tiempos_cruce) > 1:
                intervalos = [(tiempos_cruce[i] - tiempos_cruce[i - 1]).total_seconds() for i in range(1, len(tiempos_cruce))]
                t_medio = sum(intervalos) / len(intervalos)

            # Imprimir y Guardar
            print(f"✅ Fin VERDE. Vehículos: {len(contados)} | Ocupación Espacial: {ocupacion_final_promedio:.2f}%")
            
            guardar_resultados_csv(
                len(contados), 
                t_medio, 
                ocupacion_final_promedio,
                inicio_verde_time.strftime("%H:%M:%S"),
                frame_time.strftime("%H:%M:%S"),
                DIA_SEMANA,
                DIRECCION
            )

            # Salto temporal
            segundos_a_saltar = max(0, RED_DURATION - RED_BUFFER)
            frames_saltar = int(segundos_a_saltar * FPS)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frames_saltar)

            estado = "ESPERANDO"
            historial.clear()
            
            # Resetear variables de ocupación
            acumulador_porcentaje_ocupacion = 0.0
            contador_frames_verde = 0
            continue

    # UI Consola
    if int(current_frame_idx) % 30 == 0:
        mostrar_progreso(current_frame_idx, TOTAL_FRAMES, start_process_time, estado, ciclo)

cap.release()
print("\n\n✅ Procesamiento finalizado.")