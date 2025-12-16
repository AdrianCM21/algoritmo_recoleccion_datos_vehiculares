import pytesseract
import cv2
import numpy as np
import re
import subprocess
from datetime import datetime,timedelta

def leer_timestamp(secont, start_time):

    # YYYY-MM-DDTHH:MM:SS
    time = datetime.fromisoformat(start_time)

    tiempo = time + timedelta(seconds=secont)    
    return tiempo

def euclidean_distance(detection, tracked_object):
    """
    Retorna la distancia euclidiana entre la detección y la estimación del objeto trackeado
    """
    
    return np.linalg.norm(detection.points - tracked_object.estimate[0])


def obtener_hora_inicio(video):
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format_tags=creation_time",
        "-of", "default=nw=1",
        video
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if "creation_time=" not in result.stdout:
        return None

    fecha = result.stdout.strip().split("=")[1]
    return datetime.fromisoformat(fecha.replace("Z", "+00:00"))

 
