import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker
from utils import euclidean_distance 

# ======================
# 1. CONFIGURACIÓN
# ======================
VIDEO_PATH = "/home/adrian/Escritorio/Universidad/tesis/videos/videoJP2/12-11-2025/miercoles-12-11-opt2.avi"

PX1, PY1, PX2, PY2 = 200, 300, 1100, 400

lineal_start = (691, 23)
lineal_end   =(892, 108)

TARGET_CLASSES = {"car", "truck"}

# ======================
# 2. FUNCIONES
# ======================
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")

def producto_cruz(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def cruzo_linea_robusto(prev_pt, curr_pt, line_start, line_end):
    """
    Verifica intersección física real y filtra por dirección.
    """
    p1, p2 = prev_pt, curr_pt
    q1, q2 = line_start, line_end

    d1 = producto_cruz(q1, q2, p1)
    d2 = producto_cruz(q1, q2, p2)
    d3 = producto_cruz(p1, p2, q1)
    d4 = producto_cruz(p1, p2, q2)

    # 1. VERIFICACIÓN DE INTERSECCIÓN REAL
    hay_interseccion = ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))

    if hay_interseccion:
        # ==========================================================
        # 2. FILTRO DE DIRECCIÓN (Derecha a Izquierda)
        # ==========================================================
        
        # IMPORTANTE: Cambia este símbolo según lo que viste en tu DEBUG.
        # Si tus autos correctos tienen d1 POSITIVO, usa:  if d1 > 0:
        # Si tus autos correctos tienen d1 NEGATIVO, usa:  if d1 < 0:
        
        if d1 > 0:  # <--- PRUEBA CAMBIANDO ESTO A '>' SI NO CUENTA NADA
            print(f"✅ Cruce VÁLIDO (Dir. Correcta). Valor d1: {d1}")
            return True
        else:
            print(f"⛔ Cruce IGNORADO (Dir. Incorrecta). Valor d1: {d1}")
            return False

    return False
def yolo_to_norfair(results, model):
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

# ======================
# 3. EJECUCIÓN
# ======================
print("--- MODO CALIBRACIÓN (SIN FILTRO DE DIRECCIÓN) ---")

model = YOLO("yolov8m.pt")
tracker = Tracker(distance_function=euclidean_distance, distance_threshold=30)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ ERROR: No se pudo abrir el video en: {VIDEO_PATH}")
    exit()

ret, frame_full = cap.read()
if not ret: exit()

h_video, w_video = frame_full.shape[:2]
ROI = np.s_[PY1:PY2, PX1:PX2]
roi_test = frame_full[ROI]
uso_roi = True

if roi_test.size == 0:
    print(f"⚠️ ROI INVÁLIDO. Usando pantalla completa.")
    uso_roi = False
else:
    print("✅ ROI válido.")
FPS = cap.get(cv2.CAP_PROP_FPS)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0*FPS)  # Reiniciar al segundo 0

window_name = "Calibracion"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, click_event)

historial = {}
vehiculos_contados = set()
contador_total = 0

while True:
    ret, frame_full = cap.read()
    if not ret: break

    if uso_roi:
        frame = frame_full[ROI].copy()
    else:
        frame = frame_full.copy()

    results = model(frame, verbose=False)
    detections = yolo_to_norfair(results, model)
    tracked_objects = tracker.update(detections=detections)

    for obj in tracked_objects:
        cx, cy = obj.estimate[0]
        track_id = obj.id

        if track_id not in historial:
            historial[track_id] = (cx, cy)
            continue

        prev_pt = historial[track_id]
        curr_pt = (cx, cy)

        color_punto = (0, 255, 0)
        
        # LLAMADA A LA FUNCIÓN CORREGIDA
        if cruzo_linea_robusto(prev_pt, curr_pt, lineal_start, lineal_end):
            if track_id not in vehiculos_contados:
                vehiculos_contados.add(track_id)
                contador_total += 1
                print(f"🚗 Contado ID: {track_id} | Total: {contador_total}")
            
        if track_id in vehiculos_contados:
            color_punto = (0, 0, 255)

        historial[track_id] = (cx, cy)

        cv2.circle(frame, (int(cx), int(cy)), 4, color_punto, -1)
        cv2.putText(frame, str(track_id), (int(cx), int(cy)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_punto, 1)

    cv2.line(frame, lineal_start, lineal_end, (255, 0, 0), 2)
    cv2.putText(frame, f"Total: {contador_total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow(window_name, frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()