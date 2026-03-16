# M4 ML notes

## 1. ML Workflow:

1. Business Understanding: comprensión del negocio y estrategia a seguir.
2. Data Understanding: distribución columnas y relación predictoras (X) con target (y)
3. Data Preparation: traducir todas las columnas a numérico: impute and transform (OHE,FE, Scaling...)
4. Modelling: escoger algoritmo y optimización de parámetros.
5. Evaluación: métricas y memorización.

### 1.1. Data Understanding y preparation:
1. Columnas a eliminar: columnas con >95% nulos y alta cardinalidad (IDs, etc).
2. Filas a eliminar: nulos en target, fila con muchos nulos y en columnas con <0.5% nulos.
3. Imputación:
4. Transformación:

## 2. Tipos de datos:

tabulares (ML Clásico), datos complejos (Deep Learning: imágenes, textos, sonido) o no hay datos (RF)


## 2. Tipos de tareas

### 2.1. Aprendizaje supervisado

Etiqueta en los datos que define exactamente la tarea. (i.e. Malware detectado, Vuelo retrasado...)

Esquema: Dataset --> Máquina --> Tarea

#### 2.1.1. Clasificación:

Boolean: True/False


#### 2.1.2. Regresión:

Float: valor contínuo numérico o series temporales

---


### 2.2. Aprendizaje no supervisado:

No hay etiqueta

#### 2.2.1. Agrupamiento:

Encontrar parecidos entre los datos. Por ejemplo: clientes parecidos.


#### 2.2.2. Asociación:

Identificar secuencias. Por ejemplo: patrones de consumo (e-comerce).

#### 2.2.3. Reducción de dimensionalidad:



### 2.3. Reinforcement Learning (RF)

No hay datos disponibles

# Hasta Clasi1:58