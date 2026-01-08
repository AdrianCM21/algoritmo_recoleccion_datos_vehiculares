import pandas as pd

# =========================
# 1. CONFIGURACIÓN
# =========================

archivos_csv = [
    "jueves-06-11.csv",
    "lunes-03-11.csv",
    "martes-11-11.csv",
    "miercoles-12-11.csv",
    "viernes-14-11.csv",
]

columnas_seleccionadas = [
    "Total_Vehiculos",
    "Tiempo_Medio_s",
    "Ocupacion_Espacial_%",
    "Hora_Inicio",
    "Hora_Fin",
    "Dia_Semana",
    "Direccion",
]

# =========================
# 2. LECTURA Y PROCESAMIENTO
# =========================

lista = []

for archivo in archivos_csv:
    df = pd.read_csv("completo/" + archivo)

    # Validar columnas
    columnas_faltantes = set(columnas_seleccionadas) - set(df.columns)
    if columnas_faltantes:
        raise ValueError(
            f"El archivo {archivo} no tiene las columnas: {columnas_faltantes}"
        )

    # Seleccionar solo las columnas necesarias
    df = df[columnas_seleccionadas]

    # Filtrar por dirección (forma correcta)
    lista.append(df)

# =========================
# 3. CONCATENACIÓN FINAL
# =========================

df_dia_1 = pd.concat(lista, ignore_index=True)

df_dia_1.to_csv("completo.csv", index=False)