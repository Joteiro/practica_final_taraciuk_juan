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

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset proviene de la iniciativa pública TidyTuesday (repositorio de datos abiertos en GitHub), concretamente del conjunto de datos sobre astronautas publicado en julio de 2020:

https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-07-14/astronauts.csv

La variable objetivo seleccionada es `hours_mission`, que representa la duración (en horas) de cada misión espacial.

Tiene sentido aplicar un modelo de regresión sobre esta variable ya que:

- Es una variable numérica continua.
- Presenta una alta variabilidad entre observaciones.
- Es razonable suponer que puede estar influida por otras variables del dataset, como la edad (`age`), la experiencia (`years_since_selection`) o el número de misiones realizadas.

Por lo tanto, se trata de una variable adecuada para estudiar relaciones cuantitativas y realizar predicciones mediante técnicas de regresión.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> _Escribe aquí tu respuesta_

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> _Escribe aquí tu respuesta_

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> Sí, hay valores nulos en el dataset pero representan menos del 1%. Como ni siquiera son variables numéricas, o categóricas que se vayan a usar, se mantienen como están. 

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---
Añade aqui tu descripción y analisis:

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> _Escribe aquí tu respuesta_


---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
Añade aqui tu descripción y analisis:

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> La fórmula β = (XᵀX)⁻¹ Xᵀy determina la combinación de coeficientes β que hace que mis predicciones se parezcan lo más posible a y. Es análogo a lo que hacíamos en la regresión lineal simple cuando minimizábamos la suma de los errores al cuadrado, pero con más de una variable independiente.
Es necesario añadir una columna de unos porque permite aprender el intercepto. Si no la agregamos, estaríamos forzando que el modelo pase por el origen (0,0).

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       | 4.864995       |
| β₁        | 2.0       | 2.063618       |
| β₂        | -1.0      | -1.117038      |
| β₃        | 0.5       | 0.438517       |

> Los cuatro coeficientes ajustados por mi función son similares a los de referencia, pero no exactamente iguales debido al ruido (por eso tenemos un R2 de ≈ 0.69 y no de 1). Si lo quitaramos nos darían los mismos resultados.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> He obtenido un MAE de 1.166462, un RMSE de 1.461243 y un R² de 0.689672. Para el MAE y el RMSE se encuentran dentro del margen de referencia, pero el R² se encuentra por afuera: ± 0.1. 

---

## Ejercicio 4 — Series Temporales
---
Añade aqui tu descripción y analisis:

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> _Escribe aquí tu respuesta_

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> _Escribe aquí tu respuesta_

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> _Escribe aquí tu respuesta_

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> _Escribe aquí tu respuesta_

---

*Fin del documento de respuestas*
