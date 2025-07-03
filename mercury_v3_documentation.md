# Mercury v3+ Hyper Neural Network

## Descripción General

Mercury v3+ Hyper es una red neuronal profunda diseñada para resolver problemas de clasificación binaria en espacios tridimensionales altamente no lineales. Esta implementación se especializa en la separación de patrones complejos entrelazados, utilizando técnicas avanzadas de optimización y regularización.

## Características Técnicas

### Arquitectura de la Red
- **Tipo**: Red neuronal profunda feedforward
- **Capas**: 3 capas (2 ocultas + 1 de salida)
- **Función de activación**: 
  - Capas ocultas: Tangente hiperbólica (tanh)
  - Capa de salida: Sigmoide
- **Función de pérdida**: Cross-entropy con regularización L2

### Técnicas de Optimización
- **Optimizador**: Adam (Adaptive Moment Estimation)
- **Learning Rate**: Adaptativo con decay exponencial
- **Regularización**: 
  - Dropout (80% retención de neuronas)
  - L2 regularization (λ = 0.01)
- **Inicialización**: Pesos aleatorios con distribución normal (σ = 0.01)

### Configuración por Defecto
```python
# Hiperparámetros optimizados
n_hidden1 = 16        # Neuronas primera capa oculta
n_hidden2 = 512       # Neuronas segunda capa oculta
learning_rate = 0.01  # Tasa de aprendizaje inicial
epochs = 3000         # Épocas de entrenamiento
dropout_rate = 0.8    # Tasa de retención dropout
```

## Instalación

### Requisitos
```bash
pip install numpy matplotlib scikit-learn scipy torch
```

### Dependencias
- Python 3.7+
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- Scikit-learn >= 0.24.0
- SciPy >= 1.6.0
- PyTorch >= 1.7.0

## Uso Básico

### Importación y Configuración
```python
import numpy as np
from Mercury_v3_Hyper import *

# Generar datos de prueba
X, Y = generate_3d_spiral_data(n_points=1000, noise=0.2, rotations=3)

# Entrenar la red
W1, b1, W2, b2, W3, b3, costs, accuracies = train_deep(
    X, Y, 
    n_hidden1=16, 
    n_hidden2=512, 
    learning_rate=0.01, 
    epochs=3000
)

# Realizar predicciones
predictions = predict_deep(X, W1, b1, W2, b2, W3, b3)
```

### Optimización de Hiperparámetros
```python
# Definir rangos de búsqueda
hidden1_neurons = [16, 32, 64]
hidden2_neurons = [256, 512, 1024]
learning_rates = [0.001, 0.01, 0.1]
epochs_list = [2000, 3000, 5000]

# Optimizar
best_params, results = optimize_deep_hyperparameters(
    X, Y, 
    hidden1_neurons, 
    hidden2_neurons, 
    learning_rates,
    epochs_list, 
    decay_rate=0.01
)
```

## Funciones Principales

### `generate_3d_spiral_data(n_points, noise, rotations)`
Genera datos sintéticos de espirales entrelazadas en 3D para entrenamiento y pruebas.

**Parámetros:**
- `n_points`: Número total de puntos a generar
- `noise`: Cantidad de ruido gaussiano (0.0 - 1.0)
- `rotations`: Número de rotaciones de la espiral

**Retorna:**
- `X`: Matriz de características (3, n_points)
- `Y`: Vector de etiquetas (1, n_points)

### `train_deep(X, Y, n_hidden1, n_hidden2, learning_rate, epochs, verbose)`
Entrena la red neuronal con los datos proporcionados.

**Parámetros:**
- `X`: Datos de entrada
- `Y`: Etiquetas verdaderas
- `n_hidden1`: Neuronas en primera capa oculta
- `n_hidden2`: Neuronas en segunda capa oculta
- `learning_rate`: Tasa de aprendizaje inicial
- `epochs`: Número de épocas de entrenamiento
- `verbose`: Mostrar progreso durante entrenamiento

**Retorna:**
- Pesos y sesgos entrenados (W1, b1, W2, b2, W3, b3)
- Historial de costos y precisiones

### `predict_deep(X, W1, b1, W2, b2, W3, b3)`
Realiza predicciones sobre nuevos datos.

**Parámetros:**
- `X`: Datos de entrada
- `W1, b1, W2, b2, W3, b3`: Parámetros entrenados

**Retorna:**
- `predictions`: Predicciones binarias (0 o 1)

## Aplicaciones en la Vida Real

### 🏥 Medicina y Bioinformática
- **Diagnóstico médico**: Clasificación de tumores benignos vs malignos
- **Análisis de proteínas**: Identificación de conformaciones estables
- **Neurociencia**: Detección de patrones de actividad cerebral anómalos

### 🌍 Ciencias Ambientales
- **Calidad del aire**: Clasificación de zonas contaminadas
- **Geología**: Detección de formaciones rocosas con potencial minero
- **Oceanografía**: Clasificación de masas de agua por características químicas

### 🏭 Ingeniería y Manufactura
- **Control de calidad**: Detección de productos defectuosos
- **Mantenimiento predictivo**: Clasificación de equipos que requieren servicio
- **Robótica**: Navegación en espacios complejos con obstáculos

### 💰 Finanzas y Economía
- **Detección de fraude**: Identificación de transacciones fraudulentas
- **Análisis de riesgo crediticio**: Clasificación de clientes por riesgo
- **Trading algorítmico**: Predicción de movimientos del mercado

### 📊 Ciencia de Datos
- **Segmentación de clientes**: Clasificación de comportamientos de compra
- **Análisis de sentimientos**: Clasificación de opiniones
- **Detección de anomalías**: Identificación de patrones inusuales

## Visualización

### Gráficos Disponibles
- **Datos originales**: Visualización 3D de espirales entrelazadas
- **Predicciones**: Comparación entre predicciones y valores reales
- **Curvas de aprendizaje**: Evolución del error y precisión
- **Optimización**: Comparación de hiperparámetros
- **Animación 3D**: Visualización rotativa de la frontera de decisión

### Ejemplos de Visualización
```python
# Visualizar datos originales
plot_3d_spiral_data(X, Y)

# Visualizar predicciones
plot_3d_predictions(X, Y, predictions)

# Crear animación de frontera de decisión
animation = plot_decision_boundary_animation(X, Y, W1, b1, W2, b2, W3, b3)
```

## Métricas de Evaluación

### Métricas Implementadas
- **Precisión (Accuracy)**: Porcentaje de predicciones correctas
- **Matriz de Confusión**: Distribución de verdaderos/falsos positivos y negativos
- **AUC-ROC**: Área bajo la curva ROC
- **Reporte de Clasificación**: Precision, Recall, F1-Score por clase
- **MSE**: Error cuadrático medio durante el entrenamiento

### Ejemplo de Evaluación
```python
# Evaluar modelo
predictions = predict_deep(X_test, W1, b1, W2, b2, W3, b3)
evaluate_model(Y_test, predictions)
```

## Arquitectura Detallada

### Flujo de Datos
```
Input (3D) → Dense(16, tanh) → Dropout(0.8) → Dense(512, tanh) → Dropout(0.8) → Dense(1, sigmoid) → Output
```

### Proceso de Entrenamiento
1. **Forward Propagation**: Cálculo de activaciones capa por capa
2. **Compute Loss**: Cálculo de cross-entropy + regularización L2
3. **Backward Propagation**: Cálculo de gradientes con Adam
4. **Update Parameters**: Actualización de pesos y sesgos
5. **Learning Rate Decay**: Reducción gradual de la tasa de aprendizaje

## Consideraciones de Rendimiento

### Complejidad Computacional
- **Tiempo de entrenamiento**: O(n × m × epochs)
- **Memoria**: O(n × m) donde n = características, m = muestras
- **Predicción**: O(n × m) tiempo lineal

### Optimizaciones Implementadas
- Vectorización con NumPy para operaciones matriciales
- Dropout solo durante entrenamiento
- Adam optimizer para convergencia rápida
- Learning rate decay para estabilidad

## Limitaciones

### Limitaciones Técnicas
- Diseñado específicamente para datos 3D
- Clasificación binaria únicamente
- Requiere ajuste manual de hiperparámetros
- Sensible a la escala de los datos

### Recomendaciones de Uso
- Normalizar datos de entrada
- Usar validación cruzada para hiperparámetros
- Monitorear overfitting con conjunto de validación
- Considerar early stopping para evitar sobreentrenamiento

## Contribuciones

### Cómo Contribuir
1. Fork del repositorio
2. Crear rama de feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

### Áreas de Mejora
- [ ] Soporte para clasificación multiclase
- [ ] Implementación de más optimizadores
- [ ] Soporte para datos de dimensiones variables
- [ ] Interfaz gráfica para configuración
- [ ] Exportación de modelos entrenados

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## Contacto

Para preguntas, sugerencias o reportar bugs, por favor abre un issue en el repositorio de GitHub.

---

**Desarrollado con ❤️ para la comunidad de Machine Learning**