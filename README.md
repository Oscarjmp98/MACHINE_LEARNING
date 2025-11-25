# MACHINE_LEARNING
## Examen Final – Machine Learning

Aplicación Web - Machine Learning Aplicado

Esta aplicación web, desarrollada con Streamlit, tiene como objetivo facilitar la evaluación de distintos modelos de Machine Learning previamente entrenados. De manera simple y accesible, el usuario puede interactuar con modelos de Regresión Logística, KNN y K-Means, aplicados a dos casos reales: predicción de churn en telecomunicaciones y segmentación de clientes de tarjetas de crédito.

## Descripción General

La plataforma fue diseñada para que cualquier persona, incluso sin experiencia técnica, pueda:

- Evaluar el modelo de Regresión Logística para predecir la probabilidad de que un cliente abandone el servicio (churn).
- Probar el modelo KNN utilizando el mismo formulario de características del cliente.
- Aplicar el modelo K-Means para clasificar perfiles de usuarios de tarjetas de crédito según su comportamiento financiero.

De esta manera, la aplicación integra modelos supervisados y no supervisados en una sola interfaz, permitiendo comparaciones rápidas y análisis exploratorios.

## Cómo Instalar y Ejecutar la Aplicación

### Requisitos previos

Antes de iniciar, es necesario contar con:

- Python 3.8 o superior
- pip, el gestor de paquetes de Python

### Pasos para correr la aplicación

1. **Clonar el repositorio**

```bash
git clone https://github.com/Oscarjmp98/MACHINE_LEARNING.git
cd MACHINE_LEARNING
```

2. **Instalar las dependencias**

```bash
pip install -r requirements.txt
```

3. **Ejecutar la aplicación**

```bash
cd pages
streamlit run app.py
```

4. **Abrirla en el navegador**

Streamlit se abrirá automáticamente en: http://localhost:8501

Si no se abre, simplemente copia la URL mostrada en la terminal

## Estructura del Proyecto

```
MACHINE_LEARNING/
├── pages/
│   ├── app.py                     # Aplicación principal Streamlit
│   ├── assets/
│   │   └── estilo.css             # Estilos personalizados
│   └── modelos/
│       ├── modelo_logistica.pkl
│       ├── modelo_knn.pkl
│       ├── modelo_kmeans.pkl
│       ├── scaler.pkl
│       ├── scaler_kmeans.pkl
│       ├── feature_columns.json
│       ├── feature_columns_credit.json
│       └── cluster_profiles.json
├── requirements.txt
└── README.md
```

Esta estructura organiza claramente los modelos, configuraciones, estilos y la lógica de la aplicación.

## Funcionalidades Principales

### 1. Predicción de Churn (Clientes Telecom)

**Modelos utilizados:** Regresión Logística y KNN

El formulario recoge datos esenciales del cliente, como:

- **Numéricos:** antigüedad, cargos mensuales, cargos totales
- **Categóricos:** género, pareja, dependientes, servicios contratados, entre otros

**Resultados obtenidos:**

- **Regresión Logística:** probabilidad de churn y decisión final (Yes/No)
- **KNN:** clasificación del cliente (Yes/No)

### 2. Asignación de Cluster (Tarjetas de Crédito)

**Modelo utilizado:** K-Means

Se solicitan las variables financieras mínimas necesarias, como:

- BALANCE
- PURCHASES
- CASH_ADVANCE
- CREDIT_LIMIT
- PAYMENTS
- MINIMUM_PAYMENTS
- TENURE

**Resultados obtenidos:**

- Número del cluster asignado (0, 1 o 2)
- Descripción del perfil correspondiente a ese cluster, basada en datos reales de entrenamiento

## Dependencias del Proyecto

Todo lo necesario se encuentra en `requirements.txt`, incluyendo:

- streamlit
- pandas
- numpy
- scikit-learn
- joblib

Cada una de estas librerías cumple un rol clave: desde la interfaz web hasta la carga de modelos y el manejo de datos.

## Tecnologías Utilizadas

- **Python 3.8+**
- **Streamlit** – creación rápida de aplicaciones web
- **scikit-learn** – entrenamiento y ejecución de modelos ML
- **pandas** – manipulación de datos
- **numpy** – cálculos y operaciones numéricas
- **joblib** – carga eficiente de modelos entrenados

## Notas Importantes

- Los modelos deben estar previamente entrenados y guardados en formato .pkl
- Los archivos JSON de configuración son esenciales para reproducir correctamente las predicciones
- El scaler debe ser el mismo utilizado durante el entrenamiento para asegurar coherencia en los datos

## Autores

Equipo del curso Machine Learning Aplicado.

## Licencia

Proyecto desarrollado con fines académicos.
