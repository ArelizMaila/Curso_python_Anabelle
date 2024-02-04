''' TAREA 4
Integrantes: 
- Areliz Maila 
- Pablo Sarango
'''

# Análisis de indicadores del Banco Mundial 
# Tema: Emisiones de CO2 per capita

### Importar herramientas 
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns

### Leer las hojas del archivo de excel 
df_countries = pd.read_excel("data/tables/API_EN.ATM.CO2E.KT_DS2_es_excel_v2_6301161.xls",sheet_name="Metadata - Countries")

df_index = pd.read_excel("data/tables/API_EN.ATM.CO2E.KT_DS2_es_excel_v2_6301161.xls",sheet_name="Data",skiprows=  3) 
# skiprows sirve para indicar cuantas filas al inicio del archivo se deben omitir durante la lectura 

''' PREGUNTA 1
¿Cuál es el valor promedio del indicador seleccionado entre los países de América Latina en el año 2020?
'''

### Filtrar el DataFrame para America latina 
america_latina = df_countries[df_countries["Region"] == "América Latina y el Caribe (excluido altos ingresos)"]

### Fusionar los Data_Frames para obtener las emisiones de CO2 de América latina
emisiones_am_lat = america_latina.merge(df_index,on=["Country Name", "Country Code"], how="left")

### Selecciono la columna del año 2020
emisiones_2020 =  emisiones_am_lat["2020"]

### Calcular el valor promedio
valor_promedio = emisiones_2020.mean()

### Imprimo el promedio
print(f"El valor promedio de las emisiones de CO2 en América Latina en el año 2020 es: {valor_promedio}")

''' PREGUNTA 2
¿Cómo ha evolucionado este indicador a lo largo del tiempo en América Latina?
'''

### Decir que la columna que se quedará estatica será la de país

### Seleccionar solo las columnas 'Country Code' y los años
columnas_seleccionadas = ['Country Code'] + [str(anio) for anio in range(1990, 2020)] #Desde 1990 hay datos, omitiré los demas años 

emisiones_am_lat_paises = emisiones_am_lat[columnas_seleccionadas]

### Decir que la columna estática será Country Code
valores = [col for col in emisiones_am_lat_paises.columns if col!="Country Code"]

### Hacer que los años y los valores sean columnas
am_lat_melted = pd.melt(emisiones_am_lat_paises, id_vars = ["Country Code"], value_vars = valores,value_name ="Emisiones de CO2 (kt)",var_name="Anio")

### Graficar la evolución a lo largo del tiempo
plt.figure(figsize=(16, 8))
sns.barplot(data=am_lat_melted, x='Anio', y='Emisiones de CO2 (kt)', hue='Country Code', linewidth=1)

### Ajustar la leyenda fuera del gráfico y a la derecha
plt.legend(title='Country Code', bbox_to_anchor=(1, 1), loc='upper left')

### Título y ejes
plt.title('Evolución de las emisiones de CO2 en América Latina por pais')
plt.xlabel('Año')
plt.ylabel('Emisiones de CO2 (kt)')
plt.xticks(rotation=90, ha='right')
plt.show()

''' PREGUNTA 3
¿Cómo es el mapa de correlación entre los últimos 5 años de datos disponibles para los países de América Latina?
'''

### Seleccionamos los ultimos 5 años 
ultimos_5_anios = emisiones_am_lat_paises.iloc[:, -5:]

### Creamos la matriz de correlacion
matriz_correlacion = ultimos_5_anios.corr()

### Usa seaborn para crear un mapa de calor
sns.heatmap(matriz_correlacion, annot=True, cmap="YlGnBu")
plt.title('Mapa de Correlación')
plt.show()
