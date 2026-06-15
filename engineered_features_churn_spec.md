# Especificación Técnica: Ingeniería de Características (Engineered Features) para la Predicción de Churn

Este documento detalla las 8 características diseñadas (*engineered features*) estratégicas para el modelo predictivo de abandono de usuarios (*churn*). Estas variables transforman los datos transaccionales y de comportamiento crudos en señales matemáticas claras que un modelo de Machine Learning o un sistema de alertas pueda procesar para anticipar la cancelación del servicio.

---

## 📊 Categoría 1: Engagement y Uso del Sistema

### 1. Ratio de Desaceleración de Uso (Velocity Ratio)
* **Descripción:** Mide la tendencia a corto plazo del nivel de actividad del usuario en comparación con su propio promedio histórico reciente.
* **Fórmula:** $$\text{Velocity Ratio} = \frac{\text{Acciones ejecutadas en los últimos 7 días}}{\left(\frac{\text{Acciones ejecutadas en los últimos 30 días}}{4}\right)}$$
* **Tipo de Dato:** `FLOAT` (numérico continuo).
* **Interpretación:** * Un valor cercano o superior a `1.0` indica estabilidad o incremento en el uso.
  * Un valor inferior a `0.5` indica que el usuario ha reducido su actividad a la mitad en la última semana, lo que representa una alerta roja de desinterés progresivo.

### 2. Profundidad de Adopción (Feature Breadth)
* **Descripción:** Cuantifica la diversificación del usuario dentro de la plataforma. Evalúa qué tan integrado está el software en los diferentes flujos de trabajo de la cuenta.
* **Fórmula:**
  $$\text{Feature Breadth} = \frac{\text{Módulos o funciones distintas usadas en el mes}}{\text{Total de módulos esenciales disponibles}}$$
* **Tipo de Dato:** `FLOAT` (rango entre `0.0` y `1.0`).
* **Interpretación:** Los usuarios con baja adopción (ej. `0.1` o `0.2`) usan la herramienta para una sola tarea aislada. Son altamente propensos al churn porque son fáciles de sustituir por soluciones competidoras. Los usuarios con alta adopción (`>0.7`) tienen procesos operativos dependientes de la plataforma.

### 3. Intensidad de la Acción Principal (Core Action Intensity)
* **Descripción:** Aísla el "ruido" de la navegación superficial (como hacer login y mirar el dashboard) y mide específicamente la frecuencia con la que el usuario extrae el valor real o el *core* del producto.
* **Fórmula:**
  $$\text{Core Action Intensity} = \frac{\text{Número de veces que ejecutó la \"acción de valor\"}}{\text{Días activos en el mes}}$$
* **Tipo de Dato:** `FLOAT` (frecuencia diaria).
* **Interpretación:** La "acción de valor" varía según la aplicación (ej. exportar un reporte crítico, procesar una transacción, disparar un webhook de automatización). Si este ratio cae de forma sostenida, significa que el cliente está pagando por una herramienta que ya no le genera valor operativo.

---

## ⚠️ Categoría 2: Fricción y Experiencia Técnica

### 4. Densidad de Errores Recientes (Error Rate Severity)
* **Descripción:** Captura la frustración técnica acumulada por el usuario en su ventana de actividad más reciente.
* **Fórmula:**
  $$\text{Error Rate Severity} = \frac{\text{Errores de sistema/excepciones encontradas en los últimos 14 días}}{\text{Total de sesiones iniciadas en los últimos 14 días}}$$
* **Tipo de Dato:** `FLOAT`.
* **Interpretación:** Mide el impacto de la inestabilidad técnica. Un incremento repentino en este ratio correlaciona directamente con el churn por frustración (*bad user experience*), permitiendo al equipo de soporte intervenir proactivamente antes de la cancelación.

### 5. Tiempo de Resolución de Soporte (Support Friction Time)
* **Descripción:** Mide el impacto percibido de los problemas no resueltos o de resolución lenta, indexado contra el rendimiento estándar de la empresa.
* **Fórmula:**
  $$\text{Support Friction Time} = \frac{\text{Horas promedio requeridas para cerrar los tickets del usuario}}{\text{Promedio histórico general de resolución de la empresa}}$$
* **Tipo de Dato:** `FLOAT`.
* **Interpretación:** * Un valor de `1.0` significa que el usuario recibe una atención dentro de la media.
  * Un valor de `2.5` significa que sus problemas tardan un 150% más de tiempo en resolverse que los del resto de clientes, aumentando radicalmente la probabilidad de abandono por abandono de servicio.

---

## 💰 Categoría 3: Comportamiento Financiero y Ciclo de Vida

### 6. Variación de Gasto (Downgrade Momentum)
* **Descripción:** Detecta el *churn parcial*. Identifica cuando un cliente reduce su facturación, comportamiento que suele preceder al cierre definitivo de la cuenta.
* **Fórmula:**
  $$\text{Downgrade Momentum} = \text{Gasto del mes actual} - \text{Gasto promedio de los últimos 3 meses}$$
* **Tipo de Dato:** `DECIMAL` / `FLOAT` (monetario).
* **Interpretación:** * Valores negativos (ej. `-250.00`) indican que el usuario está eliminando licencias, reduciendo capacidades contratadas o bajando de plan tarifario. Es una señal inequívoca de contracción y riesgo.

### 7. Proximidad a Renovación Crítica (Renewal Danger Zone)
* **Descripción:** Transforma una fecha estática en un indicador numérico de cuenta regresiva que permite al modelo asociar la inactividad con la oportunidad de salida contractual.
* **Fórmula:**
  $$\text{Renewal Danger Zone} = \text{Fecha de fin de contrato / renovación} - \text{Fecha actual (en días)}$$
* **Tipo de Dato:** `INTEGER` (días restantes).
* **Interpretación:** Un usuario con bajo engagement no puede hacer churn fácilmente si tiene un contrato anual vigente a 200 días de distancia. Sin embargo, si esta métrica es `< 30` días y los ratios de uso son bajos, la probabilidad de no-renovación es inminente.

### 8. Días de Silencio Post-Onboarding (Ghosting Rate)
* **Descripción:** Evalúa el éxito o fracaso de la etapa de adopción temprana (*onboarding*). Mide el tiempo de inactividad inicial después del registro.
* **Fórmula:**
  $$\text{Ghosting Rate} = \text{Días transcurridos desde el registro} - \text{Días hasta la primera acción de valor}$$
* **Nota:** Si la primera acción de valor aún no ha ocurrido, se utiliza: `Días transcurridos desde el registro` de forma directa.
* **Tipo de Dato:** `INTEGER` (días).
* **Interpretación:** Si un usuario se registra o instala el sistema y este valor supera los 7 o 10 días sin registrar actividad real, el proceso de onboarding ha fallado. El usuario se ha convertido en un "fantasma" (*ghosted user*) y su probabilidad de retención a largo plazo es estadísticamente nula.

---

## 🛠️ Notas de Implementación para el Desarrollo

1. **Almacenamiento e Historial:** Para poder calcular estas variables de forma eficiente, la base de datos relacional (por ejemplo, mediante vistas organizadas o tablas agregadas en **MySQL**) debe registrar la actividad con marcas de tiempo (`TIMESTAMP`) claras para cada evento, sesión, error y cambio de facturación.
2. **Pipeline de Cómputo (ETL):** Estas características no necesitan ser calculadas en tiempo real con cada clic del usuario. Se recomienda implementar un script o procedimiento almacenado programado (cron job / evento diario) que compute estas 8 métricas una vez al día para cada usuario activo y almacene el resultado en una tabla histórica de características (`user_features_snapshot`).
3. **Preparación para Machine Learning:** Antes de alimentar el modelo predictivo (como una regresión logística, Random Forest o XGBoost), recuerda tratar los valores nulos (por ejemplo, definir qué pasa con el *Velocity Ratio* si el usuario tuvo 0 acciones el mes pasado) y normalizar las escalas si el algoritmo lo requiere.
