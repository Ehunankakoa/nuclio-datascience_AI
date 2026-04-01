Visualization 1: https://vimeo.com/reviews/45bc3b2f-2846-468e-8d7b-224e6b9cbc73/videos/1165495457

Visualization 2: https://vimeo.com/reviews/d67f3f9e-f3ad-4b20-bbe5-f05ace129fc3/videos/1166142307

Visualization 3: https://vimeo.com/reviews/04803976-5b5a-4570-995d-252b05f80987/videos/1166490881

# Buenas prácticas de visualización de datos

## Gráficos: mostrar sólo información necesaria.

- 2D, no 3D. Se debe evitar el ruido dado que la info debe ser la justa y necesaria para entender y retener la información.

- Fuente adecuada: color legible y tamaño adecuado. Se puede enfatizar, destacar un dato, se puede mostrar un cambio de color.

- Diferencias representadas de forma precisa. Dependiendo del mensaje que queremos dar, emplear un rango u otro. No es deseable, pero dependiendo de contexto, útil.

- No conectar variables discretas con gráficos de conexión de puntos (propio de time series).


### Malas prácticas:

- Representación no clara por emplear imagen (billetes en gráfico barras) y muchos números --> simplificar.
- Mucha información en el gráfico --> usar pies de gráfico para más explicación.
- Empleo de 3D cuando no es necesario.
- Títulos de ejes en orientación incorrecta.
- Leyendas: más difícil de leer el gráfico --> Etiquetas
- Mucha información en la misma columna y poco comparable --> agrupación de categorías


### Tipos de gráficos a tener en cuenta:

1. Finance Sankeys: https://www.sankeyart.com/
2. Pie Charts: https://dondevanmisimpuestos.es/politicas#view=functional&year=2023
    - Usar como partes sobre un todo, mejor si todo suma 100%.
    - Ordenados de mayor a menor.
    - Limitar a 5 categorías (añadir 'otros' si es necesario)
    - Si queremos representar datos, mejor con barras.
    - Etiquetas, no leyendas.


# Seaborn

## Introducción a Seaborn

sns.set_theme(style=”darkgrid”)

sns.load_dataset(”tips”) #si no encuentra el dataset descargado lo busca y lo descarga

## Relaciones Estadísticas

sns.relplot(x=”total_bill”, y=”tips”, data=tips); #scaterplot basico

sns.relplot(x=”total_bill”, y=”tips”, hue = “smoker”, style = “time”, data=tips);

- (El punto y coma sirve para que devuelva la gráfica y no la gráfica como objeto)

- **hue** pinta de colores diferentes esa vairable, si es numérica la categoriza de más claro a más oscuro (se puede emplear el comando **palette** para ajustar el color: verde= “ch:r=-.5,1=.75”), también hay: pastel, muted, bright, deep, colorblind y dark

- **style** cambia la forma de los puntos

- **size** hace que los puntos cambien de tamaño según el dato, si se añade el parámetro ‘sizes’ se puede especificar el tamaño. sizes = (15,200)

- **col = “time”** hace que se generen dos gráficos, uno por cada categoría de time (Lunch y Dinner

- cuando hay variables con varias categorías se puede emplear **col_wrap = 5**, esto agrupa gráficos teniendo 5 por fila

**height** da la altura del gráfico, **aspect = .75** (base = 75% de la altura), **linewidth=**grosor line, data = fmri[fmri[”region”]==”frontal”]); datos sólo para region = frontal

## Relplots

df = pd.Dataframe(dict(time=np.arange(500), value=np.random.randn(500).cumsum())) 

- time tendrá un array de 0 a 499 de uno en uno.
- value tendrá 500 números media en 0 y con distribución normal
- cumsum hace la suma acumulativa

sns.relplot(x=”time”, y=”value”, kind=”line”, data=df);

Gráfico mal hecho —> se debe asegurar que el eje x sólo tiene 1 valor de y

df = pd.Dataframe(np.random.randn(500,2).cumsum(axis=0), columns=[’x’,’y’])

sns.relplot(x=’x’,y=’y’, sort=False, kind=”line”, data=df)