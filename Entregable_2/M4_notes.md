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
3. Análisis de variables:

    3.1. Fecha: plot vs target.

    3.2. Categóricas: PIVOTTABLE (groupby por etiqueta) con el len, sum y media de target. df.pivot_table(index = 'var_analizar', values = target, aggfinc = [len, sum, np.mean])

    3.3. Numéricas: en principio no hay preprocesado. Lo suyo es mirar las más importantes y las menos. Boxplot, correlación pej.

4. Imputación:

    Depende del algoritmo tendrá una estrategia u otra. DT y RF manejan bien categorias 'missing', linear regression no tanto.

5. Transformación:

    5.1.  

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


## 3. Tipos de algoritmos:

### 3.1. Métodos de reglas:

Reducción de entropía y creación de subsets parecidos

- Clasificación y Regresión: DT, RF & Gradient Boosting DT

- Agrupamiento: Hierarchical clustering

### 3.2. Métodos de vecindad:

Minimización de distancia entre puntos. Cálculos de parecidos.

- Clasificación y Regresión: K-NN (K- Nearest Neighbours)
- Agrupamiento: K Means/Modes (busca medias/modas de puntos parecidos) y DBSCAN

### 3.2. Métodos geométricos:

Ajuste de un plano de N-dimensiones. (En n=2, ajuste de línea a puntos)

- Clasificación: Logistic Regression & SVM (Support Vector Machine)
- Regresión: Linear Regression (series temporales), ARIMA o Prophet
- Agrupamiento: nada.


### 3.3. Métodos conexionistas:

Redes neuronales (clasi & regression):

- Clasificación: textos, Large Language Models (LLMs). {}






## M4. JFK dataset

Si dataset desbalanceado, cuanto menos datos más problemático es. Jugar con class_weights



     Hasta Clasi1:58