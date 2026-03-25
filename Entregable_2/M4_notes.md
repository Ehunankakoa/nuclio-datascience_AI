# M4 ML notes

## 1. ML Workflow:

1. Business Understanding: comprensión del negocio y estrategia a seguir.
2. Data Understanding: distribución columnas y relación predictoras (X) con target (y)
3. Data Preparation: traducir todas las columnas a numérico: impute and transform (OHE,FE, Scaling...)
4. Modelling: escoger algoritmo y optimización de parámetros.
5. Evaluación: métricas y memorización.

### 1.1. Data preparation:

1. Split: validation (ver Estrategias de Validación), test y train.
2. Columnas a eliminar: columnas con >95% nulos o alta cardinalidad (IDs, etc).
3. Filas a eliminar: nulos en target, fila con muchos nulos y en columnas con <0.5% nulos.
4. Análisis de variables:

    4.1. Fecha: plot vs target. Extraer números.

    4.2. Categóricas: PIVOTTABLE (groupby por etiqueta) con el len, sum y media de target. df.pivot_table(index = 'var_analizar', values = target, aggfinc = [len, sum, np.mean])

    4.3. Numéricas: en principio no hay preprocesado. Lo suyo es mirar las más importantes y las menos. Boxplot, correlación pej.

5. Imputación:

    Depende del algoritmo tendrá una estrategia u otra. DT y RF manejan bien categorias 'missing', linear regression no tanto.

    - Si boolean: tener en cuenta moda subgrupo, si null se parece a True --> '2', si False --> '-1', depende del alg. (si hay muchos nulos: imputar moda implica ruido y no diferenciar nulos// añadir categoría 'otros' pasa a OHE, seguimos teniendo toda la info)

6. Transformación (categóricas):

    6.1.  Previo, ajuste de tipos:
    - Extraer numeros de fechas (día, mes, año, semana, día_semana, etc). 
    - Numéricas con comportamiento categórico (Ids, número de versión, etc)

    6.2. Es boolean? (2 variables) -> sí: True/False o 0/1

    6.3. >2 variables: tiene orden? -> Intrínseco: respuestas encuesta, versiones -> LABEL ENCONDING (revisar correcto encoding)

    6.4. no hay orden, tengo tiempo. Revisar fuentes externas: ciudad -> PIB/cap / aeropuerto: #pasajeros anuales, con esta transformación le doy contexto numérico a la máquina. Pero exige mucho tiempo.

    6.5. Si no hay tiempo. <100 etiquetas --> OHE
    
    6.6. > 100 etiquetas -> reducción de etiquetas, agrupación de minoritarias (RareGrouper)
    
    6.7. > 100 etiquetas aun reduciendo -> Frequency Encoding. La máquina aprende la frecuencia con la que aparece la etiqueta en el dataset.

## 2. Tipos de datos:

tabulares (ML Clásico), datos complejos (Deep Learning: imágenes, textos, sonido) o no hay datos (RF)


## 2. Tipos de tareas

### 2.1. Aprendizaje supervisado

Etiqueta en los datos que define exactamente la tarea. (i.e. Malware detectado, Vuelo retrasado...)

Esquema: Dataset --> Máquina --> Tarea

#### 2.1.1. Clasificación:

- Boolean: True/False
- Data Understanding: pivot table (cat) & boxplot (num)

#### 2.1.2. Regresión:

- Float: valor contínuo numérico o series temporales
- Data Understanding: boxplot (cat) & corr (num)

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

Reducción de entropía y creación de subsets parecidos. Trabajan bien con técnicas de introducción de valor distinto ('missing') o fuera de rango (en numéricas).

---
Clasificación y Regresión: DT, RF & Gradient Boosting DT

#### 3.1.1. Decision Tree (10k-100k filas, poca tendencia a overfitting):

Funciona bien en tareas de clasificación y regresión (no en time-series)

Parámetros para el ajuste ('DecisionTreeClassifier?' para más info):

- min_sample_leaf (otro es min_sample_split): número mínimo de samples para cada hoja. Si es muy pequeño el resultado no es significativo. Normlamente min = 1000.
- max_depth: profundidad del árbol (poco underfitting, mucho overfitting)
- Threshold: a partir de qué porcentaje de la predicción se predice que el target será True.

Para verificarlo graphviz

#### 3.1.2. Random Forest (100k-500k):

Funciona bien en tareas de clasificación y regresión (no en time-series)

**Parámetro extra respecto DT: n_estimators, # copias del dataset**

Consiste en la combinación de varios modelos intentando predecir la misma tarea. Tipos de combinación:

- Hard voting: avg(pred) (1,1,0)= 66%
- Average voting (+ usada): avg(prob) (0.9, 0.7, 0.2) = 60%
- Stacking (+resursos): utilizar las prob. de cada modelo base como predictor de un nuevo modelo a entrenar. No mucho beneficio respecto av voting.

Para que el modelo funcione:

- Precisos: modelo base es bueno
- Variedad: los modelos base tienen predicciones distintas. Se puede conseguir con:
    - Diferentes algoritmos
    - Modificar el dataset original (+usado): bagging (RF) o Boosting (GB)

1. Bagging: réplicas del dataset original con la técnica bootstrap. En cada réplica pueden repetirse o faltar filas del dataset, de esta forma se genera variedad.


#### 3.1.3. Gradient boost (>500k filas):

Parámetro extra respecto DT: n_estimators, # copias del dataset

2. Boosting: proceso iterativo, tarda más, # iteraciones = dataset distintos.
Tras un primer modelo DT, pondera los errores ("duplica" las filas que el modelo anterior se equivoca), el nuevo dataset se emplea para entrenar el siguiente modelo.
La prediccion final se obtiene del avg voting de todos los modelos.

Nota: alta tendencia a overfitting, ponderacion de errores sólo en train, puede encontrar errores especificos del dataset.


---
Agrupamiento: Hierarchical clustering

### 3.2. Métodos de vecindad:

Minimización de distancia entre puntos. Cálculos de parecidos.

- Clasificación y Regresión: K-NN (K- Nearest Neighbours)
- Agrupamiento: K Means/Modes (busca medias/modas de puntos parecidos) y DBSCAN

**Sensible a outliers**

### 3.3. Métodos geométricos:

Ajuste de un plano de N-dimensiones. (En n=2, ajuste de línea a puntos)

- Clasificación: Logistic Regression & SVM (Support Vector Machine)
- Regresión: Linear Regression (series temporales), ARIMA o Prophet
- Agrupamiento: nada.

**Sensible a outliers**

### 3.4. Métodos conexionistas:

Redes neuronales (clasi & regression):

- Clasificación: textos, Large Language Models (LLMs). {}

## 4. Clasificación: Métodos de rendimiento/evaluación (Clasi4)

Comparación del model.score con train vs test -> over o underfitting? Si diferencia de score es <1%, es un fit correcto.

Si dataset con fechas, revisar el success dependiendo de la fecha.

### 4.1. Accuracy (clasificación)

- % aciertos: y <> ÿ (target y predicción son distintos)
- Muy sencilla -> si un modelo tiene 99%, no se puede concluir que sea bueno. Depende de la gravedad de los diferentes tipos de error. Además, puede estar overfitted. Hay que comparar con modelo dummy = media target (puede ser que haya muy poco ejemplo de target y 99% son target = 0)


### 4.2. Confusion Matrix, F1 score:

- Confusion matrix: True Positive (TP), False Positive (FP), False Negative (FN) & True Negative (TN). Ayuda a visualizar los valores obtenidos y los tipos de fallos.

- F1 score: realiza una media armónica de Precision y Recall. Beneficia cuando son parecidas. Premia la reducción de falsos negativos. Robusto a datasets desbalanceados.

        R = TP/(TP+FN)
        P = TP/(TP+FP)

        F1 = 2/[(1/P)+(1/R)]


- F2 score: hermano del F1 score, mediante su parámetros beta se decide la importancia del recall.

### 4.3. AUC ROC

Area Under the Curve, robusto a desbalanceo. Curva formada por TPR vs FPR, la curva se forma con puntos con diferentes threshold.

- TPR: True Positive Rate. Cantidad de aciertos, recall.

        TPR (Recall)= TP/(TP+FN) // Better if close to 1

- FPR: False positive rate

        FPR = FP/(FP + TN)       // Better if close to 0

AUC e [0.5 ,1] (better close to 1), if <0.5, probabilities might have to be changed (detects errors better)

AUC > 0.75 --> buen modelo


### 4.4. Feature Importance

Es importante revisar las feature importance para:

- Detectar features eliminables para en el momento de volver a entrenar, se haga con menos ruido y menos overfitting (menos reglas específicas).
- Revisar comportamiento del modelo: tal vez sólo usa 5 variables (underfit de manual)
- Revisar features muy correlacionadas (podría sumarse su importance) 

    pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending = False).head (15)

## 5. Regresión: Métodos de rendimiento/evaluación

Comparación del model.score con train vs test -> over o underfitting? Si diferencia de score es <1%, es un fit correcto.

Si dataset con fechas, revisar el success dependiendo de la fecha.

### 5.1. Métricas

- MSE (Mean Square Error)
- RMSE (Root Mean Square Error)
- MAE (Mean absolute Error)
- MAPE (Mean Absolute Percentage Error)


### 5.2. Outliers (Reg1)

Se deben corregir para que no afecten a los métodos de vecindad o geométricos.

Formas de corregir:

1. Eliminación: se reconocen como errores random (persona con 80 hijos) y su afectación es menor al 1% de los datos.
2. Transformación:

    2.1. Logarítmica: contrae los valores extremos al grueso de la distribución. Si transformación tambn en target nos aseguramos relación lineal. (Sólo vale para valores positivos). 
    La distribución normal nos permite emplear boxplot para eliminar outliers y usar Regresión o métodos de Vecindad con más tranquilidad.
    Permite comparar diferentes variables.    

    2.2. Scaling (non-supervised methods): normalización de los datos para e [0,1]

    2.3. Imputación: mismas técnicas de imputación que nulos.

Detección de outliers **en Distribuciones Normales**:

- Distribución normal: nos muestra la media (µ) y la desviación tipo (σ, dispersión de los datos respecto a la media).
    - µ +- σ : 68% de los datos.
    - µ +- 2σ : 95% de los datos.
    - µ +- 3σ : prácticamente todos los datos. <ins>Los que quedan son outliers</ins>.

- Rango Intercuartílico (IQR): rango entre el percentil 25 (Q1) y 75 (Q3).
    - Límite superior = Q3 + 1.5 IQR
    - Límite inferior = Q1 - 1.5 IQR


### 5.3. Correlaciones:

Importante: correlación **<ins>no significa causalidad </ins>**

- Lineal (Pearson): variables continuas tienen correlación lineal e [-1,1]. Si >0 relación directa, si es <0 inversamente relacionada.
- Relación no lineal (Spearman): las series temporales son un ejemplo. La variable cambia con el tiempo.

- TPR: True Positive Rate. Cantidad de aciertos, recall.

        TPR (Recall)= TP/(TP+FN) // Better if close to 1

- FPR: False positive rate

        FPR = FP/(FP + TN)       // Better if close to 0

AUC e [0.5 ,1] (better close to 1), if <0.5, probabilities might have to be changed (detects errors better)

AUC > 0.75 --> buen modelo


### 5.4. Feature Importance

Es importante revisar las feature importance para:

- Detectar features eliminables para en el momento de volver a entrenar, se haga con menos ruido y menos overfitting (menos reglas específicas).
- Revisar comportamiento del modelo: tal vez sólo usa 5 variables (underfit de manual)
- Revisar features muy correlacionadas (podría sumarse su importance) 

    pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending = False).head (15)


## 6. Estratégicas de validación

Bias-Variance trade off. 
- Overfitting: data memorization, low generalization capacity. Too much vairance.
- Underfitting: too simple, too much bias. Decision thresholds not correctly adjusted.

Para elegir el validation set se debe tener en cuenta la distribución del target:

- Clasificación: intentar mantener la distribución del target en cada split.
- Estacionalidad: si el target tiene ciclos repetitivos, el validation set es el ciclo anterior (Julio -> julio anterior)
- Tendencia: dirección constante (precio vivienda). validation set es el subset más reciente.
- Si mix: mejor datos más recientes.

### 6.1. Random Holdout

Dataset split into Validation Set (final evaluation), Test Set (model development) and Training Set. El tamaño de cada partición debe ajustarse al tamaño del dataset y ser suficientemente grandes para obtener métricas de evaluación estadísticamente significativas.

### 6.2. K-Fold Cross Validation

Repetición de proceso de modelización (Train + Test) k veces. El dataset ya se ha dividido en Validation y modelling split/k. De esta manera se obtiene una métrica de rendimiento promedio, modelizando k veces, empleando cada split de modeling como test en cada vez.

### 6.3. Bootstrap (muy pocos datos)

En caso de muy pocos datos para realizar la validación, se realizan n repeticiones de modelización, mezclando splits dentro del dataset.


## M4. JFK dataset

Si dataset desbalanceado, cuanto menos datos más problemático es. Jugar con class_weights

## M4. Alejandro Tinto

- Lograr certificado AWS. Aprender a emplear los recursos de aws.
- FireDucks reduce mucho el tiempo de pandas en datasets grandes.

     Hasta descanso Regre1