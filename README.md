# Análisis de Migración en España - Proyecto de Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Análisis exhaustivo de patrones migratorios en España utilizando técnicas de aprendizaje automático y modelos de pronóstico de series temporales.

## Contexto Académico

**Institución:** Universidad Técnica de Riga (RTU)  
**Programa:** Maestría en Ciencias de la Computación  
**Curso:** Data Analytics with Machine Learning  
**Tipo de Proyecto:** Proyecto Final del Curso  
**Año Académico:** 2024-2025

Este proyecto representa la culminación del curso de análisis de datos y aprendizaje automático, demostrando competencia en análisis estadístico, modelado predictivo y pronóstico de series temporales.

## Descripción del Proyecto

Este proyecto de investigación analiza los patrones migratorios en España mediante un enfoque integral basado en datos, integrando múltiples conjuntos de datos internacionales y aplicando técnicas avanzadas de aprendizaje automático para predecir tasas de inmigración y comprender las dinámicas migratorias.

### Objetivos

1. **Integración de Datos**: Combinar conjuntos de datos de la ONU, Banco Mundial y gobierno español
2. **Análisis Exploratorio**: Identificar patrones y tendencias en datos migratorios
3. **Modelado Predictivo**: Desarrollar y comparar múltiples modelos de ML para predicción de tasas de inmigración
4. **Análisis de Series Temporales**: Aplicar técnicas de pronóstico temporal (VAR, LSTM)
5. **Análisis por Género**: Examinar patrones migratorios específicos por género

## Fuentes de Datos

### Conjuntos de Datos Principales

- **Naciones Unidas (ONU)**: Estadísticas globales de migración y tasas de alfabetización
- **Banco Mundial (BM)**: Indicadores económicos (inflación, alfabetización, PIB)
- **Gobierno Español (datos.gob.es)**: Registros detallados de migración para España
- **Cobertura Temporal**: Datos históricos multi-anuales
- **Alcance Geográfico**: Flujos migratorios internacionales hacia/desde España

### Características Analizadas

- Tasas de inmigración (general, por género)
- Tasas de alfabetización (adultos, jóvenes, por género)
- Indicadores económicos (inflación, PIB)
- Tendencias temporales
- País de origen/destino

## Metodología

### 1. Preprocesamiento de Datos

**Limpieza de Datos:**
- Imputación de valores faltantes mediante regresión lineal y media
- Detección y tratamiento de valores atípicos
- Normalización de cadenas y estandarización de códigos de países
- Conversión y validación de tipos de datos

**Ingeniería de Características:**
- Mapeo de códigos de países (pycountry, babel)
- Extracción de características específicas por género
- Creación de características temporales
- Codificación categórica

### 2. Análisis Exploratorio de Datos (EDA)

- Resúmenes estadísticos y distribuciones
- Análisis de correlación
- Visualización de tendencias temporales
- Identificación de patrones específicos por país
- Análisis de disparidad de género

### 3. Modelos de Machine Learning

#### Enfoques de ML Tradicional

**Regresión Lineal**
- Modelo base para predicción de tasa de inmigración
- Regresión OLS con statsmodels
- Modelos específicos por país y globales
- Rendimiento: R² variable dependiendo del país

**Random Forest Regressor**
- Método de ensamble para relaciones no lineales
- Análisis de importancia de características
- Ajuste de hiperparámetros
- Validación cruzada

**XGBoost**
- Gradient boosting para mayor precisión
- Manejo de valores faltantes
- Selección de características
- Técnicas de regularización

**Redes Neuronales (MLP)**
- Regresor de perceptrón multicapa
- Optimización de arquitectura
- Selección de función de activación
- Análisis de convergencia

#### Modelos de Series Temporales

**LSTM (Long Short-Term Memory)**
- Aprendizaje profundo para datos secuenciales
- Modelado de dependencias temporales
- Pronóstico multi-paso
- Implementación de early stopping

**VAR (Vector Autoregression)**
- Análisis multivariado de series temporales
- Pruebas de estacionariedad (Augmented Dickey-Fuller)
- Selección de orden de rezago
- Análisis de respuesta al impulso

### 4. Evaluación de Modelos

**Métricas:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Error Absoluto Mediano
- Score R² (Coeficiente de Determinación)
- Scores de validación cruzada

**Estrategia de Validación:**
- División train-test
- Validación cruzada K-fold
- Validación temporal para series temporales
- Validación específica por país

## Stack Tecnológico

### Tecnologías Core

```
Python 3.8+
├── Procesamiento de Datos
│   ├── pandas
│   ├── numpy
│   └── pycountry, babel
├── Machine Learning
│   ├── scikit-learn
│   ├── XGBoost
│   └── statsmodels
├── Deep Learning
│   ├── TensorFlow
│   └── Keras
├── Visualización
│   ├── matplotlib
│   └── seaborn
└── Análisis Estadístico
    └── statsmodels
```

### Librerías Clave

- **pandas**: Manipulación y análisis de datos
- **numpy**: Cálculos numéricos
- **scikit-learn**: Algoritmos de ML y preprocesamiento
- **TensorFlow/Keras**: Modelos de aprendizaje profundo (LSTM)
- **XGBoost**: Gradient boosting
- **statsmodels**: Modelos estadísticos (OLS, VAR, prueba ADF)
- **matplotlib/seaborn**: Visualización de datos
- **pycountry/babel**: Estandarización de códigos de países

## Estructura del Proyecto

```
spain-migration-analysis/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
└── Spain_Migration_Analisis.ipynb
```

## Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- Jupyter Notebook o Google Colab
- 4GB RAM mínimo (8GB recomendado)
- Conexión a Internet para acceso a conjuntos de datos

### Instalación Local

```bash
# Clonar repositorio
git clone https://github.com/Daniromero1410/spain-migration-analysis.git
cd spain-migration-analysis

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt

# Lanzar Jupyter Notebook
jupyter notebook Spain_Migration_Analisis.ipynb
```

### Google Colab (Recomendado)

El notebook está optimizado para Google Colab:

1. Subir `Spain_Migration_Analisis.ipynb` a Google Colab
2. Instalar dependencias adicionales (primera celda)
3. Montar Google Drive para acceso a conjuntos de datos
4. Ejecutar celdas secuencialmente

## Uso

### Flujo de Trabajo Básico

```python
# 1. Importar librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 2. Cargar datos
df = pd.read_csv("datos_migracion.csv")

# 3. Preprocesar
# (Ingeniería de características, imputación, codificación)

# 4. Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
modelo = RandomForestRegressor(n_estimators=100)
modelo.fit(X_train, y_train)

# 5. Evaluar
predicciones = modelo.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))
```

### Avanzado: Pronóstico de Series Temporales

```python
# LSTM para predicción secuencial
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

modelo = Sequential([
    LSTM(50, return_sequences=True, input_shape=(timesteps, features)),
    LSTM(50),
    Dense(1)
])

modelo.compile(optimizer='adam', loss='mse')
modelo.fit(X_train, y_train, epochs=100, validation_split=0.2)
```

## Hallazgos Clave

### Resumen de Rendimiento de Modelos

| Modelo | RMSE | R² | MAE | Notas |
|--------|------|----|----|-------|
| Regresión Lineal | Variable | -0.010 a 0.85 | - | Rendimiento dependiente del país |
| Random Forest | Mejorado | 0.65-0.90 | Menor | Mejor para patrones no lineales |
| XGBoost | Mejor (tradicional) | 0.70-0.92 | Más bajo | Maneja bien datos faltantes |
| Red Neuronal MLP | Bueno | 0.60-0.88 | - | Requiere ajuste cuidadoso |
| LSTM | Excelente (temporal) | 0.75-0.93 | - | Mejor para series temporales |
| VAR | Bueno (multivariado) | - | - | Pronóstico de múltiples variables |

### Conclusiones

1. **Patrones Específicos por País**: El rendimiento del modelo varía significativamente según el país de origen
2. **Diferencias de Género**: Diferencia de rendimiento mínima entre modelos específicos por género
3. **Dependencias Temporales**: Los modelos LSTM y VAR capturan patrones temporales efectivamente
4. **Importancia de Características**: Las tasas de alfabetización e indicadores económicos son predictores fuertes
5. **Impacto de Calidad de Datos**: Los países con datos completos muestran mejor precisión de predicción

## Contribuciones Académicas

### Resultados de Aprendizaje

- Preprocesamiento avanzado de datos e ingeniería de características
- Análisis comparativo de algoritmos de ML
- Técnicas de pronóstico de series temporales
- Estrategias de evaluación y validación de modelos
- Análisis estadístico y prueba de hipótesis
- Integración de datos de múltiples fuentes

### Innovaciones Metodológicas

- Pipeline de integración de datos multi-fuente
- Marco de análisis desagregado por género
- Optimización de modelos específicos por país
- Enfoque híbrido de ML tradicional y aprendizaje profundo

## Visualización de Resultados

El notebook incluye visualizaciones exhaustivas:

- Tendencias temporales (gráficos de líneas)
- Análisis de distribución (histogramas, box plots)
- Mapas de calor de correlación
- Comparaciones de rendimiento de modelos
- Gráficos de valores reales vs. predichos
- Análisis específicos por país

## Limitaciones y Trabajo Futuro

### Limitaciones Actuales

- La disponibilidad de datos varía según país y período temporal
- Los valores faltantes requirieron imputación
- Rendimiento del modelo dependiente del país
- Conjunto de características limitado (económico, educativo)

### Mejoras Futuras

- Incorporar características adicionales (estabilidad política, datos climáticos)
- Expandir cobertura temporal
- Implementar modelos de ensamble
- Desarrollar dashboard interactivo
- Extender análisis a otros países europeos
- Sistema de predicción en tiempo real

## Referencias

### Fuentes de Datos

- Departamento de Asuntos Económicos y Sociales de las Naciones Unidas (UN DESA)
- Datos Abiertos del Banco Mundial
- Portal de Datos Abiertos del Gobierno Español (datos.gob.es)

### Referencias Académicas

- Análisis de Series Temporales: Hamilton (1994), Tsay (2005)
- Machine Learning: Hastie et al. (2009), Bishop (2006)
- Estudios de Migración: Massey et al. (1993), Castles & Miller (2009)

## Autor

**Daniel Romero**  
Estudiante de Maestría en Ciencias de la Computación  
Universidad Técnica de Riga

**Contacto:**  
- Email: danielromero.software@gmail.com
- LinkedIn: [daniromerosoftware](https://www.linkedin.com/in/daniromerosoftware)
- GitHub: [Daniromero1410](https://github.com/Daniromero1410)

## Agradecimientos

- Facultad de Ciencias de la Computación y Tecnología de la Información de la Universidad Técnica de Riga
- Instructores del curso Data Analytics with Machine Learning
- Proveedores de datos: ONU, Banco Mundial, Gobierno Español
- Comunidad open-source por las herramientas y librerías

## Licencia

MIT License - Ver archivo LICENSE para detalles

## Citación

Si utiliza este trabajo en su investigación, por favor cite:

```bibtex
@misc{romero2024spain,
  author = {Romero, Daniel},
  title = {Análisis de Migración en España: Un Enfoque de Machine Learning},
  year = {2024},
  publisher = {GitHub},
  journal = {Repositorio GitHub},
  howpublished = {\url{https://github.com/Daniromero1410/spain-migration-analysis}}
}
```

---

**Estado del Proyecto:** Completado (2024-2025)  
**Institución Académica:** Universidad Técnica de Riga  
**Curso:** Data Analytics with Machine Learning  
**Tipo de Proyecto:** Proyecto Final del Curso
