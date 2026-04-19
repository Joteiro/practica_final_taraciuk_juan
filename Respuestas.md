# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
### A) Resumen estructural

El dataset seleccionado contiene información histórica sobre astronautas y sus misiones espaciales.

Número de filas: 1277
Número de columnas: 24
Tamaño en memoria: 0.9252147674560547 MB

El dataset presenta una combinación de variables:

Numéricas continuas: year_of_birth, year_of_selection, year_of_mission, hours_mission, total_hrs_sum, eva_hrs_mission, total_eva_hrs
Categóricas: sex, nationality, military_civilian, occupation, mission_title, entre otras

En cuanto a la calidad de los datos, se identificaron valores nulos en las siguientes columnas:

| Variable                  | null_count | null_percentage |
|--------------------------|-----------:|----------------:|
| id                       | 0          | 0.000000        |
| number                   | 0          | 0.000000        |
| nationwide_number        | 0          | 0.000000        |
| name                     | 0          | 0.000000        |
| original_name            | 5          | 0.391543        |
| sex                      | 0          | 0.000000        |
| year_of_birth            | 0          | 0.000000        |
| nationality              | 0          | 0.000000        |
| military_civilian        | 0          | 0.000000        |
| selection                | 5          | 0.391543        |
| year_of_selection        | 0          | 0.000000        |
| mission_number           | 0          | 0.000000        |
| total_number_of_missions | 0          | 0.000000        |
| occupation               | 0          | 0.000000        |
| year_of_mission          | 0          | 0.000000        |
| mission_title            | 1          | 0.078309        |
| ascend_shuttle           | 1          | 0.078309        |
| in_orbit                 | 0          | 0.000000        |
| descend_shuttle          | 1          | 0.078309        |
| hours_mission            | 0          | 0.000000        |
| total_hrs_sum            | 0          | 0.000000        |
| field21                  | 0          | 0.000000        |
| eva_hrs_mission          | 0          | 0.000000        |
| total_eva_hrs            | 0          | 0.000000        |

Se observa que el dataset presenta un nivel muy bajo de valores nulos (<1% en todas las variables).

Dado su bajo impacto, se decide:
- Mantener el resto sin imputación para evitar introducir sesgo

### B) Estadísticos descriptivos de variables numéricas

Antes del análisis, se realizó una depuración de variables para eliminar aquellas sin valor analítico o potencialmente problemáticas:

- id, number, nationwide_number: identificadores únicos sin valor predictivo.
name, original_name: variables de alta cardinalidad y sin utilidad para modelado.
- field21: variable poco documentada (“Instances of EVA by mission”) y redundante respecto a otras variables como eva_hrs_mission. Se elimina para evitar ambigüedad y posible multicolinealidad.

Esta limpieza permite mejorar la interpretabilidad del análisis y evitar sesgos en etapas posteriores de modelado.

También se crearon nuevas variables derivadas para mejorar la interpretabilidad del análisis:

- age: calculada como la diferencia entre year_of_mission y year_of_birth, representa la edad del astronauta en el momento de la misión.
- years_since_selection: diferencia entre year_of_mission y year_of_selection, que indica la experiencia acumulada desde la selección hasta la misión.

Estas transformaciones permiten trabajar con variables más intuitivas y potencialmente más relevantes desde el punto de vista explicativo.

**Variable objetivo: hours_mission**
- Media vs mediana:
La media (1050.88) es considerablemente superior a la mediana (261), lo que indica una distribución fuertemente sesgada a la derecha. Esto sugiere la presencia de misiones con duraciones excepcionalmente largas que elevan el promedio.
- Rango intercuartílico (IQR):
El IQR es de 191.97, lo que indica que el 50% central de las observaciones se concentra en un rango relativamente acotado en comparación con el valor máximo (10505), reforzando la existencia de valores extremos.
- Asimetría (skewness):
El valor de 2.092 confirma una fuerte asimetría positiva, con una cola larga hacia valores altos.
- Curtosis:
La curtosis (4.316) indica una distribución leptocúrtica, con alta concentración de valores alrededor de la media y presencia de outliers significativos.

En conjunto, estos resultados muestran que hours_mission no sigue una distribución normal y presenta valores extremos que deberán ser considerados en el análisis posterior.

### C) Distribuciones

Las variables year_of_selection, year_of_mission y total_number_of_missions no presentan valores atípicos según el criterio IQR, lo que sugiere distribuciones relativamente estables y sin valores extremos.

Esto indica que estas variables no requieren tratamiento específico en términos de outliers y pueden ser utilizadas directamente en el análisis posterior.

En el gráfico de boxplots es interesante observar que hay países que tienen más valores extremos que otros (es lógico suponer que sus astronautas pasaron más tiempo en misiones que otros), como así hay ocupaciones que también tienen más horas de misión que otras: comandante, Mission Specialist (MSP) e ingeniero de vuelo lógicamente superan a turista espacial.

### D) Variables categóricas

En lo que respecta al género, hay una clara predominancia de hombres (88.8%) por sobre mujeres, lo cual es consistente con la historia de la exploración espacial.
En otras variables como nationality o occupation, la distribución es más dispersa, aunque sigue habiendo ciertas categorías predominantes: EEUU domina el ranking de países con el 66.9% y MSP el de ocupaciones con el 39%. La proporción entre militares y civiles también es más pareja, sin embargo los primeros representan el 60.2%.

### E) Correlaciones

La variable objetivo "hours_mission" está muy correlacionada con "total_hrs_sum", ya que ésta un agregado de la primera (casi la misma información). De todas formas, no la descartamos por multicolinealidad porque su |r| es de 0.7 (menor al umbral de 0.9).

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

El dataset proviene de la iniciativa pública TidyTuesday (repositorio de datos abiertos en GitHub), concretamente del conjunto de datos sobre astronautas publicado en julio de 2020:

https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-07-14/astronauts.csv

La variable objetivo seleccionada es `hours_mission`, que representa la duración (en horas) de cada misión espacial.

Tiene sentido aplicar un modelo de regresión sobre esta variable ya que:

- Es una variable numérica continua.
- Presenta una alta variabilidad entre observaciones.
- Es razonable suponer que puede estar influida por otras variables del dataset, como la edad (`age`), la experiencia (`years_since_selection`) o el número de misiones realizadas.

Por lo tanto, se trata de una variable adecuada para estudiar relaciones cuantitativas y realizar predicciones mediante técnicas de regresión.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

Las variables numéricas del dataset presentan en su mayoría distribuciones asimétricas, especialmente aquellas relacionadas con la duración y acumulación de actividad espacial.

En particular, variables como hours_mission, total_hrs_sum, eva_hrs_mission y total_eva_hrs muestran distribuciones fuertemente sesgadas a la derecha, lo cual indica la presencia de valores extremos altos en un pequeño número de observaciones. Esto es consistente con la naturaleza del fenómeno, donde algunas misiones presentan duraciones significativamente mayores que el resto.

En contraste, variables temporales como year_of_mission y year_of_selection presentan distribuciones más simétricas y estables, sin evidencia de outliers según el criterio IQR.

Respecto a los outliers detectados mediante el método IQR (1.5×IQR), se observa que:

- hours_mission presenta un 21.77% de outliers
- eva_hrs_mission presenta un 15.27% de outliers
- total_hrs_sum presenta un 7.91% de outliers
- total_eva_hrs presenta un 3.52% de outliers

El resto de variables presenta proporciones marginales o nulas.

Dado el contexto del dataset, estos outliers no se consideran errores de medición, sino parte estructural de la variabilidad del dominio (misiones de distinta duración y complejidad). Por este motivo, no se ha aplicado eliminación de outliers, ya que podría eliminar información relevante del fenómeno analizado.

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

Las tres variables numéricas con mayor correlación en valor absoluto con la variable objetivo hours_mission son las siguientes:

1. total_hrs_sum → r = 0.7029
2. year_of_mission → r = 0.3771
3. eva_hrs_mission → r = 0.3814

Estas correlaciones indican que la duración de la misión está fuertemente asociada con la experiencia acumulada del astronauta (total_hrs_sum), lo cual es coherente con la intuición del problema.

Asimismo, se observa una relación moderada con variables temporales como year_of_mission, lo que sugiere un efecto de evolución histórica en la duración de las misiones espaciales (misiones más recientes tienden a tener perfiles distintos).

Finalmente, la variable eva_hrs_mission también presenta una correlación moderada, lo que indica que las actividades extravehiculares están parcialmente asociadas a misiones de mayor duración.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

Sí, hay valores nulos en el dataset pero representan menos del 1%. Como ni siquiera son variables numéricas, o categóricas que se vayan a usar, se mantienen como están. 

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---
Las variables categóricas fueron transformadas mediante One-Hot Encoding, generando variables binarias para cada categoría. Esta técnica fue elegida debido a que las variables categóricas del dataset no presentan un orden natural, por lo que métodos como Label Encoding podrían introducir relaciones artificiales entre categorías.

El uso de One-Hot Encoding permite que el modelo trate cada categoría de manera independiente, mejorando tanto la capacidad predictiva como la interpretabilidad de los coeficientes en la regresión lineal.

Se aplicó StandardScaler para escalar las variables numéricas, transformándolas a una distribución con media 0 y desviación estándar 1. Esta técnica fue elegida en lugar de MinMaxScaler debido a la presencia de outliers en varias variables, ya que el escalado estándar es más robusto frente a valores extremos y permite preservar mejor la estructura de los datos.

El modelo de regresión lineal fue entrenado tras un proceso de limpieza y selección de variables en el cual se eliminaron aquellas que no aportaban valor o que podían introducir redundancia con el target, en particular total_hrs_sum, ya que representa una agregación directamente relacionada con la duración total de las misiones.

El rol en la misión es el principal determinante de horas. Las misiones más recientes tienden a tener mayor duración, lo que refleja la evolución tecnológica y operativa de los programas espaciales. Asimismo, una mayor experiencia acumulada desde la selección también se asocia con misiones más largas. Existen diferencias estructurales entre programas espaciales de distintos países, lo que refleja factores organizacionales e históricos más que características individuales. También se observan diferencias estructurales entre programas espaciales de distintos países, lo que refleja factores organizacionales e históricos más que características individuales.

Si miramos el gráfico de residuos, observamos que el modelo genera algunas predicciones negativas para la variable objetivo, lo cual no es consistente con la naturaleza del problema, ya que dicha variable solo puede tomar valores positivos.
Este comportamiento se debe a que la regresión lineal no impone restricciones sobre el rango de salida, pudiendo generar valores en todo el dominio de los números reales. Esto evidencia una limitación del modelo en contextos donde la variable objetivo tiene restricciones naturales. Además vemos un patrón claramente no aleatorio, con una estructura lineal en los errores para valores bajos de predicción, lo que indica que el modelo presenta sesgos sistemáticos. Además, se observa heterocedasticidad, ya que la varianza de los residuos aumenta con el valor predicho. Esto sugiere que la relación entre variables no está completamente capturada por un modelo lineal simple.
Una posible solución sería aplicar una transformación sobre la variable objetivo (por ejemplo, logarítmica) o utilizar modelos que respeten estas restricciones, como hicimos con la regresión logística.

Notamos que la transformación del problema (de regresión a clasificación) puede mejorar el rendimiento predictivo, pero a costa de perder información granular sobre la variable objetivo.

> Bonus (Interpretación de predicciones Artemis II)

Aprovechando el hype por la reciente misión espacial, quisimos ver cómo nuestros modelos creados estimaban una misión con esas características. Encontramos que las predicciones muestran una alta variabilidad entre astronautas, con valores que oscilan entre aproximadamente 100 y 2500 horas, mientras que la misión duró unas 240 horas (10 días). Esta dispersión refleja que el modelo está fuertemente influenciado por variables como la ocupación y el contexto histórico de la misión, lo que provoca sobreestimaciones en perfiles asociados a roles de mayor jerarquía en el dataset histórico. Esto evidencia una limitación del modelo al ser aplicado fuera del dominio original de datos (out-of-distribution problem).
El modelo de clasificación asigna mayoritariamente la categoría “medio” (3 de 4 astronautas), que sería el equivalente a una misión de entre 1 y 2 semanas, lo que indica una mayor estabilidad frente a la variabilidad del problema continuo.

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

- MAE: 798.854
- RMSE: 1167.064
- R²: 0.556

El modelo presenta un coeficiente de determinación (R² = 0.556), lo que indica que explica aproximadamente un 56% de la variabilidad de la variable objetivo (hours_mission). Este valor puede considerarse moderado, especialmente teniendo en cuenta la naturaleza heterogénea del fenómeno analizado. El MAE de 798.85 indica que, en promedio, el modelo se desvía en aproximadamente 799 horas respecto al valor real, mientras que el RMSE de 1167.06 sugiere la existencia de errores grandes en algunos casos, penalizando más fuertemente las predicciones extremas. Esto es consistente con la alta dispersión y presencia de outliers en la variable objetivo, lo que impacta especialmente en el RMSE al penalizar más fuertemente los errores grandes.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
El modelo no “aprende” en el sentido abstracto, sino que resuelve un problema matemático concreto mediante álgebra lineal. La estimación de los coeficientes se obtiene a través de una solución cerrada que minimiza el error cuadrático, lo que permite entender con claridad qué está ocurriendo internamente.

La regresión lineal múltiple es una herramienta que permite estimar relaciones entre variables en presencia de incertidumbre. Su objetivo no es predecir perfectamente los valores observados, sino aproximar la estructura subyacente de los datos, reconociendo que siempre existirá una componente de variabilidad que no puede ser explicada por el modelo.
---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

La fórmula β = (XᵀX)⁻¹ Xᵀy determina la combinación de coeficientes β que hace que mis predicciones se parezcan lo más posible a y. Es análogo a lo que hacíamos en la regresión lineal simple cuando minimizábamos la suma de los errores al cuadrado, pero con más de una variable independiente.
Es necesario añadir una columna de unos porque permite aprender el intercepto. Si no la agregamos, estaríamos forzando que el modelo pase por el origen (0,0).

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       | 4.864995       |
| β₁        | 2.0       | 2.063618       |
| β₂        | -1.0      | -1.117038      |
| β₃        | 0.5       | 0.438517       |

Los cuatro coeficientes ajustados por mi función son similares a los de referencia, pero no exactamente iguales debido al ruido (por eso tenemos un R2 de ≈ 0.69 y no de 1). Si lo quitaramos nos darían los mismos resultados.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

He obtenido un MAE de 1.166462, un RMSE de 1.461243 y un R² de 0.689672. Para el MAE y el RMSE se encuentran dentro del margen de referencia, pero el R² se encuentra por afuera: ± 0.1. 

---

## Ejercicio 4 — Series Temporales
---
El modelo aditivo funciona: La descomposición logra separar con éxito tendencia, estacionalidad y ruido, lo que indica que los componentes se suman de forma independiente y que la amplitud de la estacionalidad no varía con el nivel de la serie (si lo hiciera, sería más apropiado un modelo multiplicativo).
La serie es predecible en su estructura: La tendencia lineal y la estacionalidad anual son componentes deterministas y estables, lo que significa que un modelo relativamente sencillo (por ejemplo, una regresión con términos de Fourier o un SARIMA) sería capaz de capturar la mayor parte del comportamiento futuro de la serie.

El residuo es "buen ruido": El hecho de que el residuo sea gaussiano, estacionario y sin autocorrelación significa que el modelo de descomposición no ha dejado ningún patrón sin explicar. No hay información oculta que un modelo más complejo pudiera aprovechar. Esto es exactamente lo que se busca en cualquier análisis de series temporales: que lo que quede sin explicar sea puro azar.
La limitación principal es que la descomposición clásica con seasonal_decompose no separa explícitamente el ciclo de largo plazo (~4 años) de la tendencia, fusionando ambos en la componente de tendencia.

El PACF del residuo tiene exactamente el mismo aspecto que el ACF: solo el lag 0 es significativo y todo lo demás está dentro de las bandas de confianza. Esto confirma que el residuo no tiene ninguna estructura autorregresiva, es decir, no hay ningún lag pasado que ayude a predecir el valor actual. Es la firma de un ruido blanco puro, lo cual es exactamente lo que se esperaba.

En resumen: la serie es un ejemplo casi ideal de comportamiento estocástico bien estructurado, donde la señal y el ruido están claramente separados. El ACF del residuo funciona como una prueba de que la descomposición fue exitosa: si todos los lags están dentro de las bandas de confianza, significa que ni la tendencia ni la estacionalidad dejaron huella en lo que queda.

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

Sí, la serie presenta una tendencia lineal creciente. En la gráfica de descomposición se aprecia claramente una recta ascendente que va desde aproximadamente 67 en enero de 2018 hasta 155 en diciembre de 2023, lo que supone un incremento de unas 88 unidades en 6 años (~14,7 unidades/año). Esto se corresponde exactamente con el parámetro de generación: tendencia = 0.05 * t + 50, que acumula 0,05 unidades por día.

La conexión con el ACF es indirecta pero importante: si la tendencia no hubiera sido bien extraída, el residuo seguiría conteniendo una deriva, y el ACF del residuo mostraría autocorrelaciones significativas que decaen muy lentamente hacia cero (el patrón típico de una serie no estacionaria). Sin embargo, el ACF del residuo cae prácticamente a cero desde el lag 1, lo que confirma que la tendencia fue eliminada correctamente. El test ADF lo ratifica con p-value = 0.00.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

Sí, existe una estacionalidad clara con periodo de 365 días (anual), visible tanto en la serie original como en el componente estacional de la descomposición. El patrón es de tipo senoidal, con picos en torno a +10/+11 unidades y valles de hasta −21 unidades respecto al nivel de tendencia, lo que da una amplitud total de aproximadamente 32 unidades. Esto se corresponde con la combinación de seno y coseno del código (15*sin + 6*cos).

Si hubiera calculado el ACF sobre la serie original sin descomponer, se verían picos significativos en los lags 365, 730, 1095... (múltiplos del periodo estacional), que son la firma característica de la estacionalidad anual. En cambio, el ACF del residuo que sí se ve en la imagen no muestra ningún pico en esos lags, lo que demuestra que la estacionalidad fue completamente absorbida por la descomposición y no quedó ningún patrón estacional sin explicar en el residuo.

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

Sí. Además de la tendencia lineal, la serie incorpora un ciclo de largo plazo de periodo ~4 años (≈1461 días) con amplitud de 8 unidades. En la gráfica de la serie original se pueden intuir ondulaciones suaves que modulan la tendencia (por ejemplo, la ralentización aparente hacia 2020 y la aceleración posterior). 

La diferencia clave con la tendencia es que el ciclo es oscilatorio —sube y baja de forma periódica— mientras que la tendencia es monotónicamente creciente. La descomposición clásica con period=365 no los separa explícitamente: el ciclo queda absorbido parcialmente en la tendencia extraída por la media móvil, razón por la que la componente de tendencia no es perfectamente rectilínea.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

Sí, el residuo se ajusta muy bien a un ruido blanco gaussiano ideal. Los estadísticos lo confirman:

Media ≈ 0.13 (cercano a cero, sin sesgo sistemático).
Desviación típica ≈ 3.22, próxima al σ=3.5 con el que se generó el ruido (la ligera diferencia se debe a que parte de la varianza queda en la tendencia extraída).
Asimetría ≈ −0.05 y curtosis ≈ −0.06, ambas muy cercanas a 0, indicando una distribución prácticamente simétrica y mesocúrtica.
Test de Jarque-Bera: estadístico = 1.10, p-value = 0.58 >> 0.05 → no se rechaza la hipótesis de normalidad.
Test ADF: p-value = 0.00 → el residuo es estacionario (sin raíz unitaria).
Los gráficos ACF y PACF muestran todas las autocorrelaciones dentro de las bandas de confianza para lags > 0, lo que confirma ausencia de estructura temporal en el residuo.

Por lo tanto, el residuo cumple todas las propiedades del ruido ideal: media nula, distribución normal, varianza constante y ausencia de autocorrelación.

---

*Fin del documento de respuestas*
