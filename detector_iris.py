
import numpy as np
import pandas as pd

iris = pd.read_csv("Iris.csv")

#iris = iris.drop('Id',axis=1)


print('Información del dataset:')
print(iris.info())

print('Descripción del dataset:')
print(iris.describe())

print('Distribución de las especies de Iris:')
print(iris.groupby('species').size())

import matplotlib.pyplot as plt

fig, ax = plt.subplots()  # Crea la figura y los ejes para graficar

# Ahora usas este `ax` para los tres gráficos
iris[iris.species == 'Iris-setosa'].plot(kind='scatter', x='sepal_length', y='sepal_width', color='blue', label='Setosa', ax=ax)
iris[iris.species == 'Iris-versicolor'].plot(kind='scatter', x='sepal_length', y='sepal_width', color='green', label='Versicolor', ax=ax)
iris[iris.species == 'Iris-virginica'].plot(kind='scatter', x='sepal_length', y='sepal_width', color='red', label='Virginica', ax=ax)

# Añadir etiquetas y título
ax.set_xlabel('Sépalo - Longitud')
ax.set_ylabel('Sépalo - Ancho')
ax.set_title('Sépalo - Longitud vs Ancho')

# Mostrar el gráfico
plt.show()


fig = iris[iris.species == 'Iris-setosa'].plot(kind='scatter',
    x='petal_length', y='petal_width', color='blue', label='Setosa')
iris[iris.species == 'Iris-versicolor'].plot(kind='scatter',
    x='petal_length', y='petal_width', color='green', label='Versicolor', ax=fig)
iris[iris.species == 'Iris-virginica'].plot(kind='scatter',
    x='petal_length', y='petal_width', color='red', label='Virginica', ax=fig)

fig.set_xlabel('Pétalo - Longitud')
fig.set_ylabel('Pétalo - Ancho')
fig.set_title('Pétalo Longitud vs Ancho')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

X = np.array(iris.drop(['species'], axis=1))  # Corregido aquí
y = np.array(iris['species'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))

algoritmo = LogisticRegression()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisión Regresión Logística: {}'.format(algoritmo.score(X_train, y_train)))

algoritmo = SVC()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisión Máquinas de Vectores de Soporte: {}'.format(algoritmo.score(X_train, y_train)))

algoritmo = KNeighborsClassifier(n_neighbors=5)
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisión Vecinos más Cercanos: {}'.format(algoritmo.score(X_train, y_train)))

algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precisión Árboles de Decisión Clasificación: {}'.format(algoritmo.score(X_train, y_train)))

# Modelo solo con sépalo
sepalo = iris[['sepal_length','sepal_width','species']]
X_sepalo = np.array(sepalo.drop(['species'], axis=1))  # Corregido aquí
y_sepalo = np.array(sepalo['species'])

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sepalo, y_sepalo, test_size=0.2)
print('Son {} datos sépalo para entrenamiento y {} datos sépalo para prueba'.format(X_train_s.shape[0], X_test_s.shape[0]))

algoritmo = LogisticRegression()
algoritmo.fit(X_train_s, y_train_s)
Y_pred = algoritmo.predict(X_test_s)
print('Precisión Regresión Logística - Sépalo: {}'.format(algoritmo.score(X_train_s, y_train_s)))

algoritmo = SVC()
algoritmo.fit(X_train_s, y_train_s)
Y_pred = algoritmo.predict(X_test_s)
print('Precisión Máquinas de Vectores de Soporte - Sépalo: {}'.format(algoritmo.score(X_train_s, y_train_s)))

algoritmo = KNeighborsClassifier(n_neighbors=5)
algoritmo.fit(X_train_s, y_train_s)
Y_pred = algoritmo.predict(X_test_s)
print('Precisión Vecinos más Cercanos - Sépalo: {}'.format(algoritmo.score(X_train_s, y_train_s)))

algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train_s, y_train_s)
Y_pred = algoritmo.predict(X_test_s)
print('Precisión Árboles de Decisión Clasificación - Sépalo: {}'.format(algoritmo.score(X_train_s, y_train_s)))

# Modelo solo con pétalo
petalo = iris[['petal_length','petal_width','species']]
X_petalo = np.array(petalo.drop(['species'], axis=1))  # Corregido aquí
y_petalo = np.array(petalo['species'])

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_petalo, y_petalo, test_size=0.2)
print('Son {} datos pétalo para entrenamiento y {} datos pétalo para prueba'.format(X_train_p.shape[0], X_test_p.shape[0]))

algoritmo = LogisticRegression()
algoritmo.fit(X_train_p, y_train_p)
Y_pred = algoritmo.predict(X_test_p)
print('Precisión Regresión Logística - Pétalo: {}'.format(algoritmo.score(X_train_p, y_train_p)))

algoritmo = SVC()
algoritmo.fit(X_train_p, y_train_p)
Y_pred = algoritmo.predict(X_test_p)
print('Precisión Máquinas de Vectores de Soporte - Pétalo: {}'.format(algoritmo.score(X_train_p, y_train_p)))

algoritmo = KNeighborsClassifier(n_neighbors=5)
algoritmo.fit(X_train_p, y_train_p)
Y_pred = algoritmo.predict(X_test_p)
print('Precisión Vecinos más Cercanos - Pétalo: {}'.format(algoritmo.score(X_train_p, y_train_p)))

algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train_p, y_train_p)
Y_pred = algoritmo.predict(X_test_p)
print('Precisión Árboles de Decisión Clasificación - Pétalo: {}'.format(algoritmo.score(X_train_p, y_train_p)))
