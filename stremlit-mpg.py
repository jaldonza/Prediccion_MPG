import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cargar los modelos entrenados
with open('random_forest_mpg_model.pkl', 'rb') as file:
    model_random_forest = pickle.load(file)

with open('lineal_regresion_mpg_model.pkl', 'rb') as file:
    model_regresion_lineal = pickle.load(file)

# Cargar el scaler que usaste en el entrenamiento (asegúrate de haberlo guardado previamente)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Cargar los datos (usa el dataset que tenías sin las columnas de marca)
data = pd.read_csv('auto-mpg-limpio-imputado.csv')
data_sin_brand = data.drop([col for col in data.columns if col.startswith('brand_')], axis=1)

# Configuración de la página
st.set_page_config(page_title="Predicción de Consumo de Combustible (MPG)", layout="wide")

# Sidebar para navegar entre las páginas
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a", ["Home", "Análisis Exploratorio", "Predicción"])

# Home
if page == "Home":
    st.title("Predicción de Consumo de Combustible (MPG)")
    st.image("https://cdn.shopify.com/s/files/1/2784/4966/files/Ferrari_Enzo_-_M5267--00001_2048x2048.jpg?v=1665397455", use_column_width=True)
    st.write("""
        Esta aplicación permite predecir el consumo de combustible (MPG) de un automóvil basado en características como la cilindrada, caballos de fuerza, peso, etc. 
        Puedes elegir entre un modelo de **Random Forest** o **Regresión Lineal** para hacer las predicciones.
    """)

# Análisis Exploratorio
elif page == "Análisis Exploratorio":
    st.title("Análisis Exploratorio de Datos")
    st.write("A continuación puedes visualizar gráficos interactivos de las variables más importantes del conjunto de datos utilizado para entrenar el modelo.")

    # Gráfico 1: Heatmap de correlación
    st.write("### Heatmap de correlación")
    corr_matrix = data_sin_brand.corr()
    fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Matriz de Correlación")
    st.plotly_chart(fig_heatmap)

    # Gráfico 2: Histogramas interactivos de todas las variables
    st.write("### Histogramas interactivos de las Variables")
    for var in ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']:
        fig_hist = px.histogram(data_sin_brand, x=var, nbins=20, title=f'Histograma de {var.capitalize()}', marginal="rug")
        st.plotly_chart(fig_hist)

    # Gráfico 3: Boxplots interactivos de todas las variables
    st.write("### Boxplots interactivos de las Variables")
    for var in ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']:
        fig_box = px.box(data_sin_brand, y=var, title=f'Boxplot de {var.capitalize()}')
        st.plotly_chart(fig_box)

    # Gráfico 4: Pairplot interactivo usando scatter_matrix
    st.write("### Pairplot interactivo entre las Variables")
    fig_pairplot = px.scatter_matrix(data_sin_brand, dimensions=['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'mpg'], title="Pairplot Interactivo")
    st.plotly_chart(fig_pairplot)

# Predicción
elif page == "Predicción":
    st.title("Predicción de Consumo de Combustible (MPG)")

    # Sidebar con entradas para las variables del modelo, usando sliders
    st.sidebar.subheader("Introduce los datos del automóvil")
    cylinders = st.sidebar.slider("Cylinders", min_value=3, max_value=12, step=1, value=6)
    displacement = st.sidebar.slider("Displacement (pulgadas cúbicas)", min_value=50, max_value=500, step=10, value=200)
    horsepower = st.sidebar.slider("Horsepower (caballos de fuerza)", min_value=40, max_value=300, step=5, value=100)
    weight = st.sidebar.slider("Weight (libras)", min_value=1500, max_value=6000, step=100, value=3000)
    acceleration = st.sidebar.slider("Acceleration (segundos)", min_value=8.0, max_value=30.0, step=0.1, value=15.0)
    model_year = st.sidebar.slider("Model Year", min_value=70, max_value=82, step=1, value=75)
    origin = st.sidebar.selectbox("Origin", options=[1, 2, 3], format_func=lambda x: {1: "USA", 2: "Europa", 3: "Asia"}[x])

    # Seleccionar el modelo
    model_choice = st.sidebar.radio("Selecciona el modelo", ("Random Forest", "Regresión Lineal"))

    # Botón para hacer la predicción
    if st.sidebar.button("Predecir MPG"):
        # Crear un DataFrame con los valores introducidos
        input_data = pd.DataFrame({
            'cylinders': [cylinders],
            'displacement': [displacement],
            'horsepower': [horsepower],
            'weight': [weight],
            'acceleration': [acceleration],
            'model year': [model_year],
            'origin': [origin]
        })

        # Escalar los datos usando el scaler entrenado
        input_data_scaled = scaler.transform(input_data)

        # Elegir el modelo y hacer la predicción
        if model_choice == "Random Forest":
            model = model_random_forest
        else:
            model = model_regresion_lineal

        predicted_mpg = model.predict(input_data_scaled)

        # Mostrar la predicción
        st.subheader(f"El consumo estimado de combustible es: {predicted_mpg[0]:.2f} MPG con {model_choice}")

        # Calcular las métricas de evaluación usando el conjunto de prueba original
        X_test_scaled = scaler.transform(data_sin_brand.drop('mpg', axis=1))
        y_test = data_sin_brand['mpg']
        y_pred = model.predict(X_test_scaled)

        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Mostrar las métricas de evaluación
        st.write(f"### {model_choice} - Mean Absolute Error (MAE): {mae:.4f}")
        st.write(f"### {model_choice} - Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"### {model_choice} - Root Mean Squared Error (RMSE): {rmse:.4f}")
        st.write(f"### {model_choice} - R-squared (R2): {r2:.4f}")

        # Lista de variables para el scatter plot
        variables = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']

        # Mostrar un scatter plot con la regresión lineal para cada variable
        st.write("### Gráficos de regresión lineal por variable")
        # Crear los gráficos para cada variable
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        axes = axes.flatten()

        for i, var in enumerate(variables):
            X = data_sin_brand[[var]].values
            y = data_sin_brand['mpg'].values

            # Ajustar el modelo de regresión lineal sobre los datos de prueba
            linear_regressor = LinearRegression()
            linear_regressor.fit(X, y)
            y_pred_line = linear_regressor.predict(X)

            # Crear el scatter plot y la línea de regresión
            axes[i].scatter(data_sin_brand[var], data_sin_brand['mpg'], color='blue', label='Datos reales')
            axes[i].plot(data_sin_brand[var], y_pred_line, color='red', label='Regresión lineal')
            axes[i].scatter(input_data[var], predicted_mpg, color='green', label='Predicción del usuario', s=100)
            axes[i].set_xlabel(var.capitalize())
            axes[i].set_ylabel('MPG')
            axes[i].legend()

        plt.tight_layout()

        # Mostrar los gráficos en Streamlit
        st.pyplot(fig)





