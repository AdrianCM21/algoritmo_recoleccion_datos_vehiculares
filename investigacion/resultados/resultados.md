## Análisis del dataset completo

### Distribución del dataset original

- Las filas representan ciclos semafóricos, los cuales constan de 33 segundos de luz verde más luz amarilla, período durante el cual los vehículos pueden circular.
- Cada fila contiene:
  - **Total_vehiculo**: Número total de vehículos en ese ciclo.
  - **Tiempo_Medio_s**: Tiempo promedio en el que se detectan los vehículos. Por ejemplo, si se detecta un vehículo cada 1 segundo, significa que los vehículos pasan muy próximos entre sí.
  - **Ocupacion_Espacial**: Porcentaje de ocupación de la vía.
  - **Hora_Inicio**: Hora en la que inicia el ciclo (luz verde).
  - **Hora_Fin**: Hora en la que finaliza el ciclo (cuando la luz se pone en rojo).
  - **Dia_Semana**: Día de la semana en formato numérico (lunes = 1).
  - **Direccion**: Carril evaluado.
- Los datos están sincronizados. Por ejemplo, si en el día 1, en la dirección 1, el ciclo semafórico termina a las 08:10:00, en la dirección 2 el ciclo comenzará a las 08:10:05, respetando los ciclos definidos en los semáforos reales. Esto se logró gracias a la constante sincronización del tiempo en los equipos de recolección.

# Datos por dirección

1. **Dirección 1**: Japón → camino al circuito  
2. **Dirección 2**: Japón → camino al centro  
3. **Dirección 3**: Japón → camino a la rotonda  
4. **Dirección 4**: Iturbe → camino a la playa


### Distribucion del dataset ya etiquetado

1. Total de los datos registrados ya etiquetados.
2. Total de registros por dia de la semana.
3. Total de registros por direccion.

![alt text](image/distribucion_dataset.png)

#### Comportamiento del tráfico.

Se observa que el tráfico en gran parte del periodo estudiado fue un trafico fluido, que se vuelve pesado en horarios picos.

---

### Análisis de centroide de cada grupo

![image.png](./image/centroides.png)

**Cluster 0:**  
- **Vehículos:** 0.95 - alto  
- **Tiempo medio entre autos:** - 0.28 - bajo  
- **Ocupación:** 0.97 - alto  
- **Cantidad:** 1.470  

Se puede observar que en este cluster el flujo vehicular es muy alto, los vehículos circulan muy cercanos entre sí debido al reducido tiempo medio entre vehículos, y la ocupación del carril es casi completa. Este cluster corresponde a tráfico pesado, donde se ocupa gran parte del carril y los vehículos avanzan de forma densa.

tráfico: alto - tiempo medio vehículos: muy bajo - ocupación del carril: completa.

**Cluster 1:**  
- **Vehículos:** -0.57 - moderado/bajo  
- **Tiempo medio entre autos:** -0.12 - moderado/bajo  
- **Ocupación:** -0.58 - moderado/bajo  
- **Cantidad:** 2.113  

En este cluster se observa un tráfico moderado, con menos vehículos que el Cluster 0. El tiempo medio entre vehículos es ligeramente bajo, lo que indica un flujo continuo pero sin saturación. La ocupación del carril es parcial, lo que permite cierto espacio entre vehículos y una circulación más fluida.

tráfico: moderado - tiempo medio vehículos: bajo - ocupación del carril: parcial

**Cluster 2:**  
- **Vehículos:** -0.93 - bajo  
- **Tiempo medio entre autos:** 3.06 - alto  
- **Ocupación:** -0.91 - bajo  
- **Cantidad:** 218  

Este cluster representa tráfico ligero. La cantidad de vehículos es baja, el tiempo medio entre ellos es alto, y la ocupación del carril es mínima. Se corresponde a periodos de baja densidad de tráfico, donde los vehículos circulan con amplio espacio.

tráfico: ligero - tiempo medio vehículos: alto - ocupación del carril: mínima


**Distribución de los cluster** 

![image.png](./image/distribucion_dataset_completo.png)

---

### Distribución de los datos por día y dirección

Los resultados de clustering, arrojan etiquetas según el estado del tráfico por día. Tomando en cuenta las direcciones se interpreta de la siguiente manera:

![image.png](image/distribucion_dataset_dia_direccion.png)

#### Análisis

- **Día lunes 03-11-2025:**
  - **Direcciones 1 y 2:** Se observa que el tráfico predominante es el etiquetado como fluido.
  - **Direcciones 3 y 4:** El tránsito es más pesado.

- **Día martes 11-11-2025:**
  - **Dirección 2:** Se observa un leve aumento del tránsito pesado con respecto a lo obtenido en el día 1.
  - **Direcciones 3 y 4:** Se mantiene una proporción similar al día 1, donde predomina el tránsito pesado.

- **Día miércoles 12-11-2025:**
  - **Direcciones 1 y 2:** Se mantiene una proporción similar a los días 1 y 2, lo que sugiere la existencia de un patrón.
  - **Dirección 3:** Se nota un aumento del tráfico fluido.
  - **Dirección 4:** Se mantiene similar a los días 1, 2 y 3.

- **Día jueves 06-11-2025:**
  - **Direcciones 1 y 2:** Se observa un aumento considerable del tráfico pesado con respecto a los días 1, 2 y 3.
  - **Direcciones 3 y 4:** Mantienen valores similares a los días anteriores.

- **Día viernes 14-11-2025:**
  - **Direcciones 1 y 2:** Se mantiene la proporción observada en los días 1, 2 y 3.
  - **Direcciones 3 y 4:** Presentan un cambio significativo en comparación con los días anteriores, donde se observa por primera vez que el tráfico fluido supera al tráfico pesado.


**Conclusiones**
Se asume que las vías que están conectadas a lugares de trabajo son aquellas que presentan mayor tráfico.
La concentración de puestos de trabajo en la zona del circuito y la calle Iturbe conecta con la calle de la costanera que tiene a su vez tiene acceso a las calles principales del centro de Encarnación, incrementando notablemente la carga de vehículos que circulan en dirección 3 y 4.


#### Distribución cluster por día y dirección

**Dirección**

Se muestra la distribución de los clusters en las cuatro direcciones y sus centroides.

**Clusters de las cuatro direcciones**

![img cluster](image/distribucion_cluster_4dir.png)

**Centroides de las cuatro direcciones**

![img centroide](image/centroide_4dir.png)

**Análisis según las direcciones**

Se puede observar que el patrón se mantiene en las cuatro direcciones, representando variaciones mínimas entre ellas.


**Dia**

**Cluster de los cinco días**

![img cluster](image/distribucion_cluster_5dias.png)

Análisis:

**Centroides de los cinco días**

![img centroide](image/centroide_5dias.png)

---

### Distribución de los datos por día, hora y dirección

Se muestra una imagen correspondiente al día y en los pixeles se puede ver las direcciones, el tamaño de los pixeles es de 10 minutos.

**1. Dia lunes 03-11-2025**

![image.png](image/dataset_dia_direccion_hora_03-11.png)

Análisis
- **Dirección 1:** Recuperación rápida; inicia saturada pero desde las 08:30 se denota el tráfico fluido casi sin interrupciones.

- **Dirección 2:** Inestable; alterna constantemente entre saturado y fluido, con un bloque crítico de congestión a media mañana.

- **Direcciones 3 y 4:** Críticas; el tráfico pesado es dominante durante toda la jornada, con ventanas de fluidez casi inexistentes.

**2. Dia martes 11-11-2025**

![image.png](image/dataset_dia_direccion_hora_11-11.png)

Análisis
- **Dirección 1:** Alta estabilidad; salvo el inicio de jornada, mantiene un flujo continuo ( verde ) superior al del lunes.

- **Dirección 2:** Muy intermitente; intercala saturación y fluidez.

- **Direcciones 3 y 4:** Se mantiene constantemete saturado, intensificándose el color rojo entre las 12:30 y 14:00.

**3. Dia miercoles 12-11-2025**

![image.png](image/dataset_dia_direccion_hora_12-11.png)
Análisis
- **Dirección 1:** Óptimo desempeño; es el día con mayor proporción de tráfico fluido para esta dirección.

- **Dirección 2:** Los intervalos fluidos son más extensos que en días anteriores.

- **Direcciones 3 y 4:** Significativos respecto al lunes y martes.

**4. Dia jueves 06-11-2025**

![image.png](image/dataset_dia_direccion_hora_06-11.png)
Análisis
- **Dirección 1:** Se presentan extensos bloques de saturación dispersos durante toda la mañana y mediodía.

- **Dirección 2:** Congestión severa; empeora respecto al miércoles, con predominancia de rojo.

- **Direcciones 3 y 4:** Mantienen la tendencia de saturación total observada en el resto de la semana.

**5. Dia viernes 14-11-2025**

![image.png](image/dataset_dia_direccion_hora_14-11.png)
Análisis
- **Direcciones 1 y 2:** Comportamiento estándar; mantienen un flujo mayoritariamente fluido, similar al promedio de lunes-miércoles.

- **Direcciones 3 y 4:** Inversión de patrón; se observa un cambio drástico positivo. A diferencia de los otros días, aparecen extensos bloques de tráfico fluido (verde).
