# Clasificación de Objetos Astronómicos con MLPClassifier

## Descripción del Proyecto
Este proyecto implementa un sistema de clasificación automática de objetos astronómicos a partir de datos fotométricos del reto PLAsTiCC (Photometric LSST Astronomical Time-series Classification Challenge). Utiliza un MLPClassifier de scikit-learn para distinguir entre 14 categorías de eventos u objetos variables. El proceso incluye extracción de características estadísticas de las series temporales de luz y la incorporación de metadatos (redshift, distancia, extinción, entre otros).

## Estructura de Archivos
- `training_set.csv.zip`: Archivo comprimido con las curvas de luz de entrenamiento (light curves), separado por `object_id` y observaciones en varios filtros.
- `training_set_metadata.csv`: Metadatos asociados a cada `object_id`, incluyendo coordenadas, redshift, extinción, distancia, y la clase objetivo (`target`).
- Notebook: `Modelo.ipynb`: Contiene el código comentado para la extracción de características, entrenamiento y evaluación del modelo, y la generación de la submission para Kaggle.
- `README.md`: Este archivo con explicación del funcionamiento, librerías necesarias y fundamentos teóricos.
- `model.joblib` (opcional): Archivo de modelo serializado tras el entrenamiento, para carga y uso en predicciones posteriores.
- `submission.csv`: Formato de salida con probabilidades para cada clase por objeto, listo para envío a la competición.

## Requisitos y Librerías
- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib (opcional, para visualizaciones y análisis de resultados)
- zipfile (módulo estándar de Python para leer archivos comprimidos)

Se sugiere crear un entorno virtual e instalar las dependencias, por ejemplo:
```bash
python -m venv venv
source venv/bin/activate   # o `venv\Scripts\activate` en Windows
pip install pandas numpy scikit-learn joblib matplotlib
```

## Fundamentos y Metodología

1. **Carga de Datos**: 
   - Se lee `training_set_metadata.csv` para obtener metadatos por `object_id`.
   - Se procesa `training_set.csv.zip` en chunks (por ejemplo, usando `chunksize=10**6`) para no sobrecargar la memoria, especialmente si se trabaja con el dataset completo de Kaggle (~20 GB).

2. **Extracción de Características Estadísticas de Curvas de Luz**:
   - Para cada `object_id`, se calculan estadísticas incrementales (Welford) de la señal de flujo en cada filtro (normalmente 6 filtros), incluyendo:
     - Conteo de observaciones (`n_obs`),
     - Rango temporal de observaciones (`t_span`),
     - Media, desviación estándar, skewness y kurtosis de los flujos detectados y de todos los flujos,
     - Valores mínimo y máximo de flujo,
     - Fracción de observaciones detectadas (basado en un umbral de detección de señal),
     - Número de filtros con detecciones (`n_bands`).
   - Se implementa una rutina `init_stats()` para inicializar estadísticas y `update_moments()` para actualizar momentos de orden 1 a 4 de forma incremental por cada nueva observación, permitiendo cómputo online sin retener todos los datos en memoria.

3. **Combinación con Metadatos**:
   - Una vez calculadas las estadísticas de luz para cada `object_id`, se unen con el DataFrame de metadatos mediante `merge` sobre `object_id`.
   - Se manejan valores faltantes rellenándolos con la mediana de cada columna numérica (`fillna`).

4. **Preparación de Datos para Modelado**:
   - Se definen las columnas a eliminar (`drop_cols`), típicamente identificadores y columnas no predictoras: `object_id`, coordenadas (`ra`, `decl`, etc.), y `target` extraída para variable objetivo (`y`).
   - `X` contiene solo características numéricas procesadas; `y` es la columna de clase (`target`).
   - Se aplica codificación de etiquetas (`LabelEncoder`) para `y` si no es numérico, o se usa directamente si ya está en formato entero.
   - Se utiliza `StandardScaler` para escalar características (`X`), vital para redes neuronales para acelerar convergencia y mejorar desempeño.

5. **Entrenamiento y Validación**:
   - División de datos en conjuntos de entrenamiento y validación (`train_test_split`), por ejemplo con `test_size=0.2` y `random_state` fijo para reproducibilidad.
   - Creación de un `MLPClassifier` con una arquitectura definida (capas ocultas, función de activación, algoritmo de optimización, número máximo de iteraciones, etc.). Estos hiperparámetros deben ajustarse según recursos y performance observado.
   - Ajuste del modelo con `clf.fit(X_train, y_train)` y evaluación sobre `X_val` con métricas como `classification_report`, `accuracy_score`, o preferiblemente Log-loss para competiciones de Kaggle.
   - Se sugiere guardar el modelo entrenado con `joblib.dump` a un archivo (`model.joblib`) para futuras predicciones sin reentrenar.

6. **Generación de Submission**:
   - Leer el dataset de prueba (`test_set.csv.zip`) y procesar de forma análoga al set de entrenamiento para extraer las mismas características de curvas de luz y combinar con sus metadatos (`test_set_metadata.csv`).
   - Asegurarse de usar el mismo escalador `StandardScaler` ajustado en entrenamiento para transformar características de test.
   - Predecir probabilidades con `clf.predict_proba(X_test_scaled)`, obteniendo un array de forma [n_samples, n_classes].
   - Construir un DataFrame con columnas: `object_id` y una columna por cada clase posible (por ejemplo, de 0 a 13 o según los labels originales), con las probabilidades correspondientes.
   - Guardar a CSV (`submission.csv`) con formato esperado por Kaggle: sin índice, header correcto.

7. **Visualizaciones y Análisis**:
   - Opcionalmente, generar gráficas de importancia de características (aunque MLP no provee importancia directa; se pueden usar técnicas de permutación u otros modelos auxiliares), curvas de aprendizaje, matriz de confusión en validación, distribuciones de características, etc.
   - Plots con `matplotlib.pyplot` para visualizar desempeño y detectar posibles desequilibrios de clases o anomalías en las características.

8. **Limitaciones y Mejora**:
   - **Recursos Computacionales**: Procesar el dataset completo requiere memoria considerable; se puede limitar al primer subconjunto de objetos (por ejemplo, primeros 1000 IDs) para pruebas rápidas, pero para resultados finales conviene un entorno con más RAM o cluster/procesamiento distribuido.
   - **Desequilibrio de Clases**: Clases raras (como la clase 52 en la exposición) pueden presentar muy pocas muestras, dificultando su clasificación. Se pueden aplicar técnicas de sobremuestreo (SMOTE) o submuestreo, ajustar pesos de clase en el `MLPClassifier` o probar modelos especializados.
   - **Arquitectura de la Red**: Un MLP simple puede no capturar patrones complejos en series temporales. Se pueden explorar redes más profundas, recurrentes (RNN/LSTM), o métodos basados en transformadas (por ejemplo, extracción de features en frecuencia) o incluso modelos de aprendizaje profundo con arquitecturas específicas para series de tiempo.
   - **Preprocesamiento Avanzado**: Detección de outliers, imputación más sofisticada, normalización por objeto o por filtro, incorporación de características temporales avanzadas (periodogramas, autocorrelaciones), etc.
   - **Validación Cruzada**: Implementar K-fold CV estratificado para evaluar robustez y evitar sobreajuste al split simple.
   - **Optimización de Hiperparámetros**: Uso de GridSearchCV o técnicas bayesianas (Optuna) para ajustar hiperparámetros del MLP (número de capas, neuronas, tasa de aprendizaje, regularización, etc.).

## Uso de la Carpeta / Ejecución
1. Colocar los archivos CSV/zipped en la raíz de este repositorio o actualizar rutas en el notebook.
2. Ejecutar el notebook `ultimoproyectodefinitivo_comentado.ipynb` paso a paso o usar un script Python equivalente:
   - Ajustar parámetros de procesamiento de chunks (por ejemplo, `chunksize`) según memoria disponible.
   - Configurar hiperparámetros del `MLPClassifier` en la sección de entrenamiento.
3. Verificar que `model.joblib` y `submission.csv` se generen correctamente.
4. Subir `submission.csv` a la competición PLAsTiCC en Kaggle para evaluar Log-loss final y ranking.

## Referencias
- Reto PLAsTiCC Kaggle: https://www.kaggle.com/competitions/PLAsTiCC-2018
- Welford’s method para cálculo de varianza y momentos superiores.
- Documentación de scikit-learn: `MLPClassifier`, `StandardScaler`, `LabelEncoder`, métricas y model_selection.

## Créditos
- Autor: [Tu Nombre]
- Fecha: Junio 2025
