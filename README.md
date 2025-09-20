# Heatmap de métodos de pago por país y tipo

Este repositorio contiene una aplicación en Streamlit para generar mapas de calor a partir de un archivo de transacciones en formato Excel o CSV. Los mapas de calor muestran, para cada código CAGE (país), el número de transacciones por **método de pago** y **estado**, diferenciando entre los tipos de transacción **DEPOSIT** y **WITHDRAWAL**.

## Cómo funciona

1. El usuario carga un archivo CSV, XLSX o XLS con las columnas relevantes (`Cage Code`, `Payment Method`, `Status`, `Create Time Minute`, `Type` y `Status Time Minute`).
2. La aplicación limpia y normaliza los datos (convierte fechas, detecta el separador y estandariza los textos).
3. Se generan mapas de calor por país y tipo. Cada celda muestra el número de transacciones y, si el estado es `IN_PROGRESS`, también la fecha y hora más antigua en ese estado.
4. El título de cada gráfico incluye el rango de fechas de las transacciones.

## Requisitos

- Python 3.7 o superior
- Dependencias listadas en `requirements.txt`:
