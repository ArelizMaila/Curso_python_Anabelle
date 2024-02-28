#Trabajo final
"""
Areliz Maila 
  Variable clave : espacio_lavado
  Filtro : sexo == "Hombre"

"""

"""Ejercicio 1: Exploración de Datos
"""
########### LIBRERÍA NECESARIA #############

# Importamos numpy para realizar operaciones numéricas eficientes.
import numpy as np
# Pandas nos permitirá trabajar con conjuntos de datos estructurados.
import pandas as pd
# Desde sklearn.model_selection importaremos funciones para dividir conjuntos de datos y realizar validación cruzada.
from sklearn.model_selection import train_test_split, KFold
# Utilizaremos sklearn.preprocessing para preprocesar nuestros datos antes de entrenar modelos de aprendizaje automático.
from sklearn.preprocessing import StandardScaler
# sklearn.metrics nos proporcionará métricas para evaluar el rendimiento de nuestros modelos.
from sklearn.metrics import accuracy_score
# statsmodels.api nos permitirá realizar análisis estadísticos más detallados y estimación de modelos.
import statsmodels.api as sm
# Por último, matplotlib.pyplot nos ayudará a visualizar nuestros datos y resultados.
import matplotlib.pyplot as plt

#### Cargar el archivo CSV para trabajar 
datos = pd.read_csv("sample_endi_model_10p (1).txt", sep=";") # sep=";" especifica que el separador de campos en el archivo CSV es “;”

### Variable y filtro a trabajar
"""Variable : espacio_lavado (Si la vivienda dispone de espacio de lavado de manos)
   Filtro : sexo == "Hombre"
"""

########### LIMPIEZA DE DATOS ###########
datos = datos[~datos["dcronica"].isna()] #Eliminamos los valores N/A
variables = ['n_hijos', 'region', 'sexo', 'condicion_empleo','espacio_lavado']
datos.groupby("espacio_lavado").size()
#for para recorrer cada variable en nuestra lista variables y eliminar las filas con valores nulos en esa variable
for i in variables:
    datos = datos[~datos[i].isna()] 

""" 
RESPUESTA ------------------Ejercicio 1: Exploración de Datos---------------
1) Calcular el numero de niños Hombres que cuentan con un espacio de lavado de manos
"""
# Conteo de niños Hombres que cuentan con un espacio de lavado de manos (variable asignada)
data = datos[(datos["sexo"] == "Hombre") & (datos["espacio_lavado"] == 1.0)]
niños_espacio_lavado = data["espacio_lavado"].value_counts()

print("El numero de niños Hombres que cuentan con un espacio de lavado de manos de los 2006 niños y niñas es:", niños_espacio_lavado)


""" Ejercicio 2: Modelo Logit
"""

############## TRANSFORMACIÓN DE VARIABLES ################
#Definir las variables categóricas y numéricas que utilizaremos en nuestro análisis
variables_categoricas = ['region', 'sexo', 'condicion_empleo', 'espacio_lavado']
variables_numericas = ['n_hijos']

#Crear un transformador para estandarizar las variables numéricas y una copia de nuestros datos para no modificar el conjunto original
transformador = StandardScaler()
datos_escalados = datos.copy()
#Estandarizar las variables numéricas utilizando el transformador
datos_escalados[variables_numericas] = transformador.fit_transform(datos_escalados[variables_numericas])
#Convertir las variables categóricas en variables dummy utilizando one-hot encoding
datos_dummies = pd.get_dummies(datos_escalados, columns=variables_categoricas, drop_first=True)
datos_dummies.info()
#Seleccionar las variables predictoras (X) y la variable objetivo (y) para nuestro modelo
X = datos_dummies[['n_hijos', 'sexo_Mujer', 
                   'condicion_empleo_Empleada', 'condicion_empleo_Inactiva', 'condicion_empleo_Menor a 15 años', 'espacio_lavado_1.0']]
y = datos_dummies["dcronica"]

#Definir los pesos asociados a cada observación para considerar el diseño muestral
weights = datos_dummies['fexp_nino']

########### SEPARACI[ON DE LAS MUESTRAS EN ENTRAMIENTO (train) Y PRUEBA (test)] #######
# dividir los datos en conjuntos de entrenamiento y prueba para poder evaluar el rendimiento de nuestro modelo. Utilizaremos un 80% de los datos para entrenamiento y un 20% para pruebas
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)
#Asegurar que todas las variables sean numericas
# Convertimos todas las variables a tipo numérico
X_train = X_train.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Convertimos las variables a tipo entero
variables = X_train.columns
for i in variables:
    X_train[i] = X_train[i].astype(int)
    X_test[i] = X_test[i].astype(int)

y_train = y_train.astype(int)

print(X_train.isna().sum())
X_train.dtypes

########### AJUSTE DEL MODELO ###############
modelo = sm.Logit(y_train, X_train)
result = modelo.fit()
print(result.summary())

#Mejorar la visualización los datos en un nuevo Dataframe
# Extraemos los coeficientes y los almacenamos en un DataFrame
coeficientes = result.params
df_coeficientes = pd.DataFrame(coeficientes).reset_index()
df_coeficientes.columns = ['Variable', 'Coeficiente']

# Creamos una tabla pivote para una mejor visualización
df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
df_pivot.reset_index(drop=True, inplace=True)

#Ya podemos realizar predicciones en el conjunto de prueba o evaluar su rendimiento
# Realizamos predicciones en el conjunto de prueba
predictions = result.predict(X_test)
# Convertimos las probabilidades en clases binarias
predictions_class = (predictions > 0.5).astype(int)
# Comparamos las predicciones con los valores reales
predictions_class == y_test
print("La precisión promedio del modelo testeando con datos test es", np.mean(predictions_class))
# Realizamos predicciones en el conjunto de entrenamiento
predictions_train = result.predict(X_train)
# Convertimos las probabilidades en clases binarias
predictions_train_class = (predictions_train > 0.5).astype(int)
# Comparamos las predicciones con los valores reales
predictions_train_class == y_train
print("La precisión promedio del modelo testeando con datos train es", np.mean(predictions_train_class))

"""
RESPUESTA ----------Ejercicio 2: Modelo Logit------------------------
1) ¿Cuál es el valor del parámetro asociado a la variable clave
si ejecutamos el modelo solo con el conjunto de entrenamiento 
y predecimos con el mismo conjunto de entrenamiento? 
¿Es significativo?

El coeficiente estimado para la variable asignada "espacio_lavado" para niños hombres es de -1.4147,
lo que indica que hay una relación logaritmica negativa entre la variable asignada y la variable de respuesta (desnutrición crónica)
lo que quiere decir que la disminución de la variable asignada se asocia con el aumento de la varibale de desnutrición. 

El P>|z| asociado a este coeficiente es  de 0.000, menor que 0.05,lo que significa que hay una estadisticamente significativa entre 
la desnutrición infantil y "espacio_lavado", se sugiere que la relación entre estas dos variables no es aleatoria y es probable que exista una 
asociación entre ellas en la población de interés, sin embargo no implica necesariamente causalidad
"""


"""  Ejercicio 3: Evaluación del Modelo con Datos Filtrados
"""
############### VALIDACIÓN CRUZADA #############
#Realizar una validación cruzada con 100 pliegues para evaluar el rendimiento de nuestro modelo de regresión logística en el conjunto de entrenamiento
# 100 folds:
kf = KFold(n_splits=100)
accuracy_scores = []
df_params = pd.DataFrame()

for train_index, test_index in kf.split(X_train):

    # aleatorizamos los folds en las partes necesarias:
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    weights_train_fold, weights_test_fold = weights_train.iloc[train_index], weights_train.iloc[test_index]
    
    # Ajustamos un modelo de regresión logística en el pliegue de entrenamiento
    log_reg = sm.Logit(y_train_fold, X_train_fold)
    result_reg = log_reg.fit()
    
    # Extraemos los coeficientes y los organizamos en un DataFrame
    coeficientes = result_reg.params
    df_coeficientes = pd.DataFrame(coeficientes).reset_index()
    df_coeficientes.columns = ['Variable', 'Coeficiente']
    df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
    df_pivot.reset_index(drop=True, inplace=True)
    
    # Realizamos predicciones en el pliegue de prueba
    predictions = result_reg.predict(X_test_fold)
    predictions = (predictions >= 0.5).astype(int)
    
    # Calculamos la precisión del modelo en el pliegue de prueba
    accuracy = accuracy_score(y_test_fold, predictions)
    accuracy_scores.append(accuracy)
    
    # Concatenamos los coeficientes estimados en cada pliegue en un DataFrame
    df_params = pd.concat([df_params, df_pivot], ignore_index=True)

print(f"Precisión promedio de validación cruzada: {np.mean(accuracy_scores)}")

]

########## VALIDACIÓN CRUZADA: EL COMPORTAMIENTO DEL PARAMETRO ASOCIADO A "espacio_lavado" #########
plt.hist(df_params["espacio_lavado_1.0"], bins=30, edgecolor='black')

# Añadir una línea vertical en la media de los coeficientes
plt.axvline(np.mean(df_params["espacio_lavado_1.0"]), color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la media de los coeficientes
plt.text(np.mean(df_params["espacio_lavado_1.0"])-0.1, plt.ylim()[1]-0.1, f'Media de los coeficientes: {np.mean(df_params["espacio_lavado_1.0"]):.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Beta (Espacio de lavado)')
plt.xlabel('Valor del parámetro')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()

"""
RESPUESTA -------------- Ejercicio 3 ------------------
1) ¿Qué sucede con la precisión promedio del modelo cuando se utiliza el conjunto de datos filtrado? 
- Precisión promedio de validación cruzada para: 
Modelo sin flitrar (de la guía) = 0.731372549019608
Modelo filtrado (Incluyendo la variable y poblacion objetivo) = 0.7769607843137254

Al incluir la variable "espacio_lavado" la precisión promedio aumenta

2) ¿Qué sucede con la distribución de los coeficientes beta en comparación con el ejercicio anterior?

- Media de coeficientes para: 
Modelo sin flitrar (Obtenido de la guía) = 0.11
Modelo filtrado (Incluyendo la variable y poblacion objetivo) = 0.07

Al incluir la variable "espacio_lavado" la media del coeficiente disminuye. 






"""

