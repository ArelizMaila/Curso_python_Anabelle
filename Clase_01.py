#Declaro mi variable
mi_variable = "Hola mundo"
print(mi_variable)

#lista de numeros, siempre entre corchetes
mi_lista = mi_lista = [1, 2, 3, 4, 5]
print(mi_lista)

#Diccionarios
mi_diccionario = {"clave_1" : "valor1","clave2" :"valor2"}
print(mi_diccionario)

###################################
#Vectores 

vector_enteros = [10]*5
print(vector_enteros)

vector_flotantes = [3.14]*5
print(vector_flotantes)

##################################
#Diccionarios
#es una estructura de datos que permite almacenar pares clave-valor.
#basicamente es darle un nombre o categoria a una columna de datos
mi_diccionario = {"entero" : vector_enteros,"flotante" : vector_flotantes ,"complejo" : vector_flotantes}
print(mi_diccionario)
print(mi_diccionario["entero"])  # Imprime: valor1


##################################
#Cadenas
cadena_simple = "hola mundo"
cadena_doble = ["Holi","gatito"]
print(cadena_doble)


##################################
#Dataframe
#Python cuenta desde el cero (indice cero)

import pandas as pd #se le da el nombre de pd a la libreria pandas 

# Crear un DataFrame : 

datos = {
    'Pez': ['Tilapia', 'Carachama', 'Carpa'], #Filas
    'Orden': ['Perciformes', 'Siluriformes', 'Cypriniformes'], #De aqui en adelante son las columnas
    'Familia': ['Cichlidae', 'Loricariidae', 'Cyprinidae'],
    'Especie': ['Oreochromis niloticus', 'Chaetostoma microps', 'Cyprinus carpio']
    }
df = pd.DataFrame(datos)

#mostar el dataframe
print(df)



#################
#lectura de una tabla Excel usando pandas
import pandas as pd
banco_tejido = pd.read_excel("data/Banco_tejido_peces.xlsx")
print(banco_tejido)