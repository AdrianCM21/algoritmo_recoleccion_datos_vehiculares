# Análisis final

## Etapa 1: Encontramos el tiempo óptimo con PSO

Para determinar el tiempo de luz verde óptimo, se utiliza un algoritmo de optimización metaheurística PSO.

### configuracion del PSO

- Particulas = 20
- Interacciones = 50

### Lógica

El algoritmo evalúa la calidad de un tiempo de luz verde mediante una Función de Costo. El "puntaje" de cada solución candidata se calcula sumando penalizaciones basadas en tres factores clave:

1.  **Penalización por Déficit ( No pasan todos los vehiculos ):**
    Si el tiempo de verde propuesto es insuficiente para la cantidad de vehículos detectados, se aplica una penalización cuadrática.
    > *Lógica: Si Vehículos > Capacidad → Costo Muy Alto*

2.  **Penalización por Exceso:**
    Si el tiempo de verde es excesivo y la calle queda vacía, se aplica una penalización moderada. Se considera ineficiente.
    > *Lógica: Si Capacidad > Vehículos → Costo Medio*

3.  **Referencia Histórica:**
    Se busca que el tiempo no se desvíe drásticamente de los estándares históricos del clúster (T_ref) para mantener la estabilidad del sistema.

### Restricciones Dinámicas por Clúster

Se utilizan rangos de búsqueda adaptados específicamente al tipo de tráfico detectado en base a los clusters:

| Cluster | Tipo de Tráfico | Rango de Búsqueda |
| :--- | :--- | :--- |
| **0** | Tráfico Pesado | 35s - 90s |
| **1** | Tráfico Medio | 20s - 60s |
| **2** | Tráfico Ligero | 10s - 40s |

---

### Ejemplo Práctico de Optimización

Escenario de Tráfico Pesado un lunes por la mañana con los siguientes datos promedio:
* **Vehículos:** 18
* **Ocupación:** 60% 

El enjambre prueba diferentes tiempos y evalúa su costo:

1.  **Prueba con 20 segundos:**
    * El sistema detecta un Déficit elevado que no pasan los 18 autos.
    * Se aplica el factor de urgencia cuando ocupación > 50% y tiempo < 25s.
    * Resultado: Costo altísimo. El enjambre descarta esta opción rápidamente.

2.  **Prueba con 85 segundos:**
    * Todos los autos pasan, pero sobra mucho tiempo.
    * Resultado: Costo medio. Es viable, pero ineficiente.

3.  **Convergencia en 58 segundos:**
    * La capacidad cubre la demanda casi exactamente.
    * No hay penalización por urgencia.
    * El tiempo está dentro del rango permitido para el Cluster 0.
    * Resultado: Costo Mínimo. Este valor se guarda como el `tiempo_optimo`.

## Etapa 2: Adición de características

En esta fase se realizó un proceso en donde se añadieron nuevas columnas al dataset en base a los datos ya existentes, esto para ayudar al modelo a aprender mejor.

Siendo que Random Forest es poderoso, pero necesita información relevante.

Se añadieron variables de alto valor predictivo.

A continuación se listan las variables añadidas y ordenadas por el tipo de variable.

### 1. Variable Temporal

Para ayudar al modelo a entender los patrones cíclicos de la ciudad, extraemos información tiempo:

- **Hora en Minutos:** Convertimos el formato `HH:MM:SS` a un valor lineal (0 a 1440), esto para ayudar al algoritmo.

### 2. Variables de Memoria

Para que el algoritmo tenga memoria o contexto del ciclo anterior aprovechando la naturaleza ordenada y dependiente de los registros, se implementaron variables de retraso y promedios para capturar esta inercia:

- **Lags:** Registramos el número de vehículos y la ocupación del ciclo anterior.
- **Media últimos 3 ciclos:** Promediamos los últimos 3 ciclos para suavizar variaciones atípicas y detectar la carga real.
- **Tendencia:** Calculamos la diferencia de vehículos entre el ciclo actual y el anterior para saber si el tráfico está subiendo o bajando.

### 3. Métricas de Eficiencia y Capacidad

Introducimos conceptos de ingeniería de tráfico para evaluar el rendimiento del semáforo:

- **Capacidad Teórica:** Define cuántos vehículos *podrían* pasar en el tiempo de verde asignado.  
> **Capacidad** = Tiempo de Verde / Tiempo Medio entre autos

- **Saturación Actual:** Es la relación entre los autos detectados y los que el semáforo realmente puede manejar.  
> **Saturación** = Vehículos Observados / Capacidad Teórica
---

### Resumen de Nuevas Características

| Categoría     | Variables Generadas                                   | Propósito                              |
|---------------|-------------------------------------------------------|----------------------------------------|
| **Tiempo**    | `Hora_Minutos`                                        | Ubicación temporal del evento.         |
| **Histórico** | `Total_Vehiculos_lag1`, `Media_Movil_3ciclos`        | Capturar la inercia del tráfico.       |
| **Dinámica**  | `Tendencia_Vehiculos`                                 | Detectar si la fila está creciendo.    |
| **Rendimiento** | `Capacidad_Teórica`, `Saturación_Actual`           | Medir la eficiencia del sistema.       |

---

### Resultado del Proceso

Al finalizar esta etapa, obtenemos un dataset enriquecido donde cada fila contiene el contexto histórico y técnico necesario para que el modelo aprenda a predecir el **Tiempo Óptimo** con mayor precisión.

---

## Etapa 3: Preparación y División de Datos

En esta fase, transformamos el dataset enriquecido en estructuras que el modelo de Machine Learning pueda procesar, asegurando una evaluación realista mediante una división temporal.

### Selección de variables a pasar al modelo

En esta etapa se suman las nuevas variables más las variables básicas y se pasan al modelo los siguientes datos que creemos que son de suma importancia:

- Vehículos  
- Ocupación  
- Tiempos medios  
- Hora del día  
- Día de la semana  
- Dirección  
- Lags históricos  
- Medias móviles  
- Tendencias de tráfico  
- Niveles de saturación

---

## Fase 4: Optimización configuracion con PSO

En esta fase, utilizaremos PSO para encontrar la configuración óptima y de mejor rendimiento para el Random Forest, utilizando la técnica de medición MSE alcanzado para validar el camino correcto.

### Parámetros que estamos optimizando

**El PSO busca el valor ideal dentro de estos rangos**

**Configuración:** 
- Particulas = 18 
- Iteraciones = 25 
- Parametros_PSO 
   - 'c1' = 0.4  
   - 'c2'= 0.8
   - 'w' = 0.6

- Función objetivo: minimizar MSE en validación cruzada

**Estos son los resultados obtenidos:**

| Hiperparámetro | Resultado | Descripción |
| :--- | :--- | :--- |
| `n_estimators` | 456 | Cantidad de árboles que componen el bosque. |
| `max_depth` | 7 | Profundidad máxima de cada árbol (controla la complejidad del modelo). |
| `min_samples_split` | 5 | Número mínimo de muestras necesarias para dividir un nodo interno. |
| `min_samples_leaf` | 3 | Número mínimo de muestras que debe contener una hoja terminal. |
| `max_features` | 1.0 | Proporción de variables consideradas en cada división del árbol. |
| `random_state` | 42 | Asegura que los resultados del entrenamiento sean reproducibles. |
| `n_jobs` | -1 | Utiliza todos los núcleos del procesador disponibles para acelerar el entrenamiento. |

---

## FASE 5: Modelo Random Forest

En esta fase implementamos nuestro modelo predictivo ya con la configuración obtenida con el PSO. Utilizamos el algoritmo Random Forest para establecer un punto de comparación inicial antes de aplicar otras optimizaciones.

### 2. Estrategia de División Temporal

Para validar el modelo, adoptamos una estrategia de Validación de Origen en Expansión (Walk-Forward acumulativo).

Este método simula el ciclo de vida real de un sistema de tráfico inteligente: el modelo entrena con los datos históricos disponibles hasta hoy para predecir el mañana, acumulando conocimiento progresivamente.

#### Metodología del Proceso
El conjunto de datos se ordena cronológicamente (Lunes hasta viernes). En cada iteración, la ventana de entrenamiento crece, anclada en el inicio, mientras que la ventana de prueba siempre se sitúa inmediatamente después en el tiempo.

1.  **Iteración 1:** El modelo aprende con el inicio de la semana (Lunes) y se evalúa con el día siguiente (Martes).
2.  **Iteración 2:** El modelo re-entrena con lo que ya sabía más los nuevos datos (Lunes + Martes) y se evalúa con el siguiente (Miércoles).
3.  **Iteración N:** El proceso se repite hasta cubrir toda la semana.

---

### Esquema de las Iteraciones de Validación

A diferencia de tener un único set estático, evaluamos el rendimiento promedio a través de múltiples escenarios temporales:

| Iteración | Conjunto de Entrenamiento (Historia Acumulada) | Conjunto de Prueba (Futuro Inmediato) | Objetivo de la Evaluación |
| :--- | :--- | :--- | :--- |
| **1** | **20%** (Lunes completo) | **20%** (Martes) | Evaluar aprendizaje inicial con un solo día base. |
| **2** | **40%** (Lunes + Martes) | **20%** (Miércoles) | Medir la mejora al duplicar la experiencia del modelo. |
| **3** | **60%** (Lun + Mar + Mié) | **20%** (Jueves) | Validar la consistencia a mitad de la semana. |
| **4** | **80%** (Lun + Mar + Mié + Jue) | **20%** (Viernes) | **Prueba Final:** Predicción del comportamiento del último día hábil. |

---

### Configuración del Modelo

Iniciamos con la configuración obtenida anteriormente (se puede detallar en la tabla anterior).

### Evaluación de Métricas

Para saber qué tan bueno es nuestro modelo base, comparamos sus predicciones contra los valores reales del conjunto de validación usando tres métricas clave:

| Técnica | Descripción Breve | Unidad de Medida |
| :--- | :--- | :--- |
| **MAE** | Promedio simple de los errores. Indica cuánto se equivoca el modelo en promedio sin exagerar los fallos. | Segundos ($s$) |
| **MSE**  | Promedio de los errores al cuadrado. Penaliza severamente los errores grandes (picos) para detectar anomalías graves. | Segundos cuadrados ($s^2$) |
| **RMSE** | Raíz cuadrada del MSE. Mantiene la penalización a los errores grandes pero devuelve el valor a una escala de tiempo legible. | Segundos ($s$) |
| **R²**  | Mide la "calidad" del ajuste. Indica qué porcentaje de la variabilidad del tráfico es explicado por el modelo (0 a 1). | Adimensional (Sin unidad) |

#### Resultados de las métricas de evaluación

```
MAE Promedio:  1.149
MSE Promedio:  7.704
RMSE Promedio: 2.762
R² Promedio:   0.8603
```

- **MAE:** 1.149 segundos  

En promedio, el modelo se equivoca por **1.149 segundos**, es excelente, es un error mínimo.

- **MSE:** 7.704

Esta en rango esperado.

- **RMSE:** 2.76 segundos  

El RMSE es más alto que el MAE, lo que indica que hay errores más grandes ocasionales; esto indica que existen cambios bruscos ocacionales, esperado conciderando el comportamiento humano y confirma alta precisión.

- **R²:** 0.86 

El modelo explica el **86% de la variabilidad** en los datos, es muy bueno para sistemas que involucran comportamiento humano.


## Resultados Obtenidos

Tras el entrenamiento y validación del modelo, se procedió a extraer los valores representativos para la configuración operativa de los semáforos.

### Propuesta de Nueva Programación Semafórica

| Dirección | Tiempo Fijo Actual (s) | Nuevo Tiempo Optimizado (s) |
| :--- | :---: | :---: |
| **Dirección 1** | 30 | **26** |
| **Dirección 2** | 30 | **26** |
| **Dirección 3** | 30 | **30** |
| **Dirección 4** | 30 | **28** |

---
