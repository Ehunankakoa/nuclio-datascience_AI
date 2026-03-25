# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


diabetes = datasets.load_diabetes()

diabetes
X = diabetes.data[:, np.newaxis, 2]  # Usamos solo la tercera característica

y = diabetes.target

df = pd.DataFrame(data=X, columns=[diabetes.feature_names[2]])
df['target'] = y
# En este contexto Y significa la progresión de la diabetes,
# Un valor más alto significa que la progresión de la diabetes es más rápida.
# X significa el indice de masa corporal (IMC)

df.head(20)

df.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)


plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='black', label='Datos reales')
plt.plot(X_test, y_pred, color='blue', linewidth=2, label='Recta de regresión')
plt.xlabel(diabetes.feature_names[2], fontsize=12)
plt.ylabel('Progresión de la diabetes', fontsize=12)
plt.title('Regresión lineal', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True) 
plt.show()

# Mostrar coeficientes

print(f'Coeficiente (pendiente): {reg.coef_[0]:.2f}')
print(f'Intersección (ordenada al origen): {reg.intercept_:.2f}')

# entonces podemos representar el modelo de la siguiente manera 

print(f"y = {reg.intercept_:.2f} +- X1·{reg.coef_[0]:.2f}")

print()

mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio (MSE): {mse:.2f}')
rmse = np.sqrt(mse)
print(f'Raíz del error cuadrático medio (RMSE): {rmse:.2f}')
mae = mean_absolute_error(y_test, y_pred)
print(f'Error absoluto medio (MAE): {mae:.2f}')
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f'Error absoluto medio porcentual (MAPE): {mape:.2f}')
# En promedio el modelo tiene un error del 45% respecto a los errores


r2 = r2_score(y_test, y_pred)
print(f'Coeficiente de determinación (R²): {r2:.2f}')

# Solo el 23% de la variabilidad de Y es explicada por X

# %%
