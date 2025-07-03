import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from scipy.special import expit as sigmoid  # Reemplaza la sigmoide manual
import sys

# Función para generar datos en forma de espiral entrelazada en 3D
def generate_3d_spiral_data(n_points=1000, noise=0.3, rotations=2):
    """
    Genera datos en forma de espiral entrelazada en 3D para clasificación binaria
    
    Args:
        n_points: Numero de puntos por clase
        noise: Cantidad de ruido gaussiano a añadir
        rotations: Numero de rotaciones de la espiral
        
    Returns:
        X: Matriz de caracteristicas de forma (3, n_points*2)
        Y: Vector de etiquetas de forma (1, n_points*2)
        
    ¿Qué hace Mercury_v3+?
    Lo que hace es:

    ✅ Genera dos espirales entrelazadas en 3D.
    ✅ Usa una red con dos capas ocultas para encontrar patrones complejos y separar las espirales.
    ✅ Usa funciones de activación tanh y sigmoide para mapear los datos a algo que sí se pueda clasificar.
    ✅ Optimiza los pesos y las tasas de aprendizaje para que el modelo aprenda mejor con cada iteración.
    ✅ Y al final, intenta clasificar los puntos correctamente (es decir, saber a qué espiral pertenece cada punto).

    En pocas palabras: Es una red neuronal diseñada para manejar problemas de clasificación complejos que tienen patrones no lineales bien enredados. 
    Si la entrenas bien, debería poder separar los datos con alta precisión. 

    """
    n = n_points // 2  # Puntos por espiral
    
    # Parámetro t para generar las espirales
    t = np.linspace(0, rotations * 2 * np.pi, n)
    
    # Primera espiral (ascendente)
    x1 = np.cos(t) + np.random.randn(n) * noise
    y1 = np.sin(t) + np.random.randn(n) * noise
    z1 = t / (rotations * 2 * np.pi) * 5 + np.random.randn(n) * noise
    
    # Segunda espiral (ascendente, pero con un offset en la rotación)
    x2 = np.cos(t + np.pi) + np.random.randn(n) * noise
    y2 = np.sin(t + np.pi) + np.random.randn(n) * noise
    z2 = t / (rotations * 2 * np.pi) * 5 + np.random.randn(n) * noise
    
    # Combinar los datos
    X = np.vstack((
        np.concatenate((x1, x2)),
        np.concatenate((y1, y2)),
        np.concatenate((z1, z2))
    ))
    Y = np.zeros((1, n_points))
    Y[0, n:] = 1  # Primera mitad clase 0, segunda mitad clase 1
    
    return X, Y

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
np.random.seed(42)
X, Y = generate_3d_spiral_data(n_points=1000, noise=0.3, rotations=2)

# Transponer X para trabajar con sklearn
X_transposed = X.T
Y_transposed = Y.T

# Primero, separamos el 20% para prueba
X_temp_t, X_test_t, Y_temp_t, Y_test_t = train_test_split(
    X_transposed, Y_transposed, test_size=0.2, random_state=42)

# Luego, del 80% restante, tomamos el 25% para validación (20% del total)
X_train_t, X_val_t, Y_train_t, Y_val_t = train_test_split(
    X_temp_t, Y_temp_t, test_size=0.25, random_state=42)

# Volver a la forma original para trabajar con nuestro modelo
X_train = X_train_t.T
Y_train = Y_train_t.T
X_val = X_val_t.T
Y_val = Y_val_t.T
X_test = X_test_t.T
Y_test = Y_test_t.T

print(f"Datos de entrenamiento: {X_train.shape[1]} muestras")
print(f"Datos de validacion: {X_val.shape[1]} muestras")
print(f"Datos de prueba: {X_test.shape[1]} muestras")

# Inicialización de pesos y sesgos para una red más profunda
def initialize_deep_parameters(n_input, n_hidden1, n_hidden2, n_output):
    np.random.seed(42)
    w1 = np.random.randn(n_hidden1, n_input) * 0.01
    b1 = np.zeros((n_hidden1, 1))
    w2 = np.random.randn(n_hidden2, n_hidden1) * 0.01
    b2 = np.zeros((n_hidden2, 1))
    w3 = np.random.randn(n_output, n_hidden2) * 0.01
    b3 = np.zeros((n_output, 1))
    return w1, b1, w2, b2, w3, b3

# Propagación hacia adelante para una red más profunda
def forward_deep_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    
    # Aplicar dropout a la capa 1
    keep_prob = 0.8  # Retener el 80% de las neuronas activas
    D1 = np.random.rand(*A1.shape) < keep_prob
    A1 = A1 * D1 / keep_prob
    
    Z2 = np.dot(W2, A1) + b2
    A2 = np.tanh(Z2)
    
    # Aplicar dropout a la capa 2
    D2 = np.random.rand(*A2.shape) < keep_prob
    A2 = A2 * D2 / keep_prob
    
    Z3 = np.dot(W3, A2) + b3
    A3 = 1 / (1 + np.exp(-Z3))  # Función sigmoide para la salida
    
    return Z1, A1, Z2, A2, Z3, A3

# Función de error (cross-entropy)
def compute_cost(A3, Y, W1=None, W2=None, W3=None,):
    m = Y.shape[1]
    epsilon = 1e-10  # Para evitar log(0)
    cost = -(1/m) * np.sum(Y * np.log(A3 + epsilon) + (1 - Y) * np.log(1 - A3 + epsilon))
    lambda_L2 = 0.01  # Factor de regularización L2
    if W1 is not None and W2 is not None and W3 is not None:
        L2_cost = (lambda_L2 / (2 * Y.shape[1])) * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
        cost += L2_cost
    return cost

# Retropropagación para una red más profunda
def backward_deep_propagation_adam(X, Y, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, b1, b2, b3, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = X.shape[1]
    
    # Gradientes
    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * (1 - np.power(A2, 2))
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    # Implementación de Adam
    global v_dW1, v_dW2, v_dW3, v_db1, v_db2, v_db3
    global s_dW1, s_dW2, s_dW3, s_db1, s_db2, s_db3
    global t

    t += 1  # Contador de iteraciones

    def adam_update(v, s, d, t):
        v = beta1 * v + (1 - beta1) * d
        s = beta2 * s + (1 - beta2) * (d ** 2)
        v_corr = v / (1 - beta1 ** t)
        s_corr = s / (1 - beta2 ** t)
        return v_corr / (np.sqrt(s_corr) + epsilon), v, s

    update_W1, v_dW1, s_dW1 = adam_update(v_dW1, s_dW1, dW1, t)
    update_b1, v_db1, s_db1 = adam_update(v_db1, s_db1, db1, t)
    update_W2, v_dW2, s_dW2 = adam_update(v_dW2, s_dW2, dW2, t)
    update_b2, v_db2, s_db2 = adam_update(v_db2, s_db2, db2, t)
    update_W3, v_dW3, s_dW3 = adam_update(v_dW3, s_dW3, dW3, t)
    update_b3, v_db3, s_db3 = adam_update(v_db3, s_db3, db3, t)

    # Actualización de parámetros
    W1 -= learning_rate * update_W1
    b1 -= learning_rate * update_b1
    W2 -= learning_rate * update_W2
    b2 -= learning_rate * update_b2
    W3 -= learning_rate * update_W3
    b3 -= learning_rate * update_b3

    return W1, b1, W2, b2, W3, b3

# Entrenamiento de la red más profunda
def train_deep(X, Y, n_hidden1, n_hidden2, learning_rate, epochs, verbose=True):
    n_input = X.shape[0]
    n_output = Y.shape[0]
    
    # Inicializar parámetros
    W1, b1, W2, b2, W3, b3 = initialize_deep_parameters(n_input, n_hidden1, n_hidden2, n_output)
    
    # Inicializar variables para Adam
    global v_dW1, v_dW2, v_dW3, v_db1, v_db2, v_db3
    global s_dW1, s_dW2, s_dW3, s_db1, s_db2, s_db3
    global t
    
    # Inicializar los acumuladores de momento y RMSprop
    v_dW1, v_dW2, v_dW3 = np.zeros(W1.shape), np.zeros(W2.shape), np.zeros(W3.shape)
    v_db1, v_db2, v_db3 = np.zeros(b1.shape), np.zeros(b2.shape), np.zeros(b3.shape)
    s_dW1, s_dW2, s_dW3 = np.zeros(W1.shape), np.zeros(W2.shape), np.zeros(W3.shape)
    s_db1, s_db2, s_db3 = np.zeros(b1.shape), np.zeros(b2.shape), np.zeros(b3.shape)
    t = 0  # Contador de iteraciones
    
    costs = []
    accuracies = []
    mse_history = []
    # Configurar decay de la tasa de aprendizaje
    initial_lr = learning_rate
    decay_rate = 0.001  # Puedes ajustar este valor
    
    for i in range(epochs):
        # Ajustar la tasa de aprendizaje
        current_lr = initial_lr / (1 + decay_rate * i)
        
        Z1, A1, Z2, A2, Z3, A3 = forward_deep_propagation(X, W1, b1, W2, b2, W3, b3)
        cost = compute_cost(A3, Y, W1, W2, W3)
        mse = np.mean((A3 - Y) ** 2)
        mse_history.append(mse)
        
        W1, b1, W2, b2, W3, b3 = backward_deep_propagation_adam(X, Y, Z1, A1, Z2, A2, Z3, A3, 
                                                                W1, W2, W3, b1, b2, b3, current_lr)
        
        if i % 1000 == 0:
            predictions = predict_deep(X, W1, b1, W2, b2, W3, b3)
            accuracy = np.mean(predictions == Y) * 100
            costs.append(cost)
            accuracies.append(accuracy)
            
            if verbose:
                print(f"Iteracion {i}: Error = {cost:.4f}, Precision = {accuracy:.2f}%")
    
    plt.plot(mse_history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Error Cuadrático Medio (MSE)')
    plt.show()
    
    return W1, b1, W2, b2, W3, b3, costs, accuracies

# Predicción para la red más profunda
def predict_deep(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_deep_propagation(X, W1, b1, W2, b2, W3, b3)
    predictions = (A3 > 0.5).astype(int)
    return predictions
# Métricas adicionales de evaluación
def evaluate_model(Y_true, Y_pred):
    cm = confusion_matrix(Y_true.flatten(), Y_pred.flatten())
    report = classification_report(Y_true.flatten(), Y_pred.flatten())
    auc = roc_auc_score(Y_true.flatten(), Y_pred.flatten())

    print("\n Matriz de Confusion:")
    print(cm)
    print("\n Reporte de Clasificacion:")
    print(report)
    print(f"\n AUC-ROC Score: {auc:.4f}")

# Visualización de los datos en 3D
def plot_3d_spiral_data(X, Y):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colores para cada clase
    colors = np.array(['blue', 'red'])
    
    ax.scatter(X[0, :], X[1, :], X[2, :], c=colors[Y[0, :].astype(int)], s=30, alpha=0.8)
    
    ax.set_title('Espirales Entrelazadas en 3D', fontsize=16)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    # Ajustar el ángulo de vista
    ax.view_init(elev=30, azim=45)
    
    plt.show()

# Visualización de predicciones en 3D
def plot_3d_predictions(X, Y, Y_pred):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Identificar puntos correctamente clasificados y puntos erróneos
    correct = (Y_pred == Y)[0]
    incorrect = ~correct
    
    # Graficar puntos correctos e incorrectos
    ax.scatter(X[0, correct], X[1, correct], X[2, correct], c=Y[0, correct], cmap=plt.cm.RdBu, alpha=0.8, marker='o', label='Correctos')
    ax.scatter(X[0, incorrect], X[1, incorrect], X[2, incorrect], color='black', alpha=1, marker='x', s=100, label='Incorrectos')
    
    ax.set_title('Predicciones en Espirales 3D', fontsize=16)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.legend()
    
    # Ajustar el ángulo de vista
    ax.view_init(elev=30, azim=45)
    
    plt.show()

# Optimización de hiperparámetros para la red más profunda
def optimize_deep_hyperparameters(X, Y, hidden1_neurons, hidden2_neurons, learning_rates, epochs_list, decay_rate):
    if not isinstance(learning_rates, (list,tuple)):
        learning_rates = [learning_rates]

    results = []
    best_accuracy = 0
    best_params = None

    for n_h1 in hidden1_neurons:
        for n_h2 in hidden2_neurons:
            for lr in learning_rates:
                for epochs in epochs_list:
                    print(f"Probando: Neuronas capa1={n_h1}, Neuronas capa2={n_h2}, Learning Rate={lr}, Epocas={epochs}")
                    
                    # Entrenamos con los parámetros actuales
                    W1, b1, W2, b2, W3, b3, _, _ = train_deep(X, Y, n_h1, n_h2, lr, epochs, verbose=False)
                    
                    # Evaluamos el rendimiento
                    predictions = predict_deep(X, W1, b1, W2, b2, W3, b3)
                    accuracy = np.mean(predictions == Y) * 100
                    
                    # Guardamos los resultados
                    results.append({
                        'n_hidden1': n_h1,
                        'n_hidden2': n_h2,
                        'learning_rate': lr,
                        'epochs': epochs,
                        'accuracy': accuracy
                    })
                    
                    print(f"Precision: {accuracy:.2f}%")
                    
                    # Actualizamos los mejores parámetros si hay mejora
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            'n_hidden1': n_h1,
                            'n_hidden2': n_h2,
                            'learning_rate': lr,
                            'epochs': epochs
                        }
    
    # Ordenamos los resultados por precisión
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Mostramos los mejores resultados
    print("\n--- Mejores Resultados ---")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. Neuronas capa1: {result['n_hidden1']}, Neuronas capa2: {result['n_hidden2']}, " +
              f"Learning Rate: {result['learning_rate']}, Epocas: {result['epochs']}, " +
              f"Precision: {result['accuracy']:.2f}%")
    
    print("\n--- Mejores Parametros ---")
    print(f"Neuronas en primera capa oculta: {best_params['n_hidden1']}")
    print(f"Neuronas en segunda capa oculta: {best_params['n_hidden2']}")
    print(f"Tasa de aprendizaje: {best_params['learning_rate']}")
    print(f"Epocas: {best_params['epochs']}")
    print(f"Precision: {best_accuracy:.2f}%")
    
    return best_params, results

learning_rates = [0.01, 0.05, 0.1]

# Generar datos de espirales en 3D
np.random.seed(42)
X, Y = generate_3d_spiral_data(n_points=1000, noise=0.2, rotations=3)

# Visualizar datos
plot_3d_spiral_data(X, Y)

# Parámetros a probar para la optimización
hidden1_neurons = [16]
hidden2_neurons = [512]
learning_rates = [0.01]  # Tasa de aprendizaje inicial
decay_rate = 0.01  # Ajusta según pruebas
epochs_list = [3000]

# Optimizar hiperparámetros
print("Iniciando optimizacion de hiperparametros...")
best_params, results = optimize_deep_hyperparameters(X, Y, 
                                                     hidden1_neurons, 
                                                     hidden2_neurons, 
                                                     learning_rates,
                                                     epochs_list, 
                                                     decay_rate)

# Entrenar con los mejores parámetros
print("\nEntrenando con los mejores parametros...")
W1, b1, W2, b2, W3, b3, costs, accuracies = train_deep(X, Y, 
                                                       best_params['n_hidden1'], 
                                                       best_params['n_hidden2'], 
                                                       best_params['learning_rate'], 
                                                       best_params['epochs'])

# Evaluar modelo final
predictions = predict_deep(X, W1, b1, W2, b2, W3, b3)
val_predictions = predict_deep(X_val, W1, b1, W2, b2, W3, b3)
test_predictions = predict_deep(X_test, W1, b1, W2, b2, W3, b3)

print("\n Evaluacion en Conjunto de Validacion:")
evaluate_model(Y_val, val_predictions)

print("\n Evaluacion en Conjunto de Prueba:")
evaluate_model(Y_test, test_predictions)


# Visualizar curva de aprendizaje
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(range(0, best_params['epochs'], 1000), costs)
plt.title('Curva de Error')
plt.xlabel('Epocas (en miles)')
plt.ylabel('Error')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(0, best_params['epochs'], 1000), accuracies)
plt.title('Curva de Precision')
plt.xlabel('Epocas (en miles)')
plt.ylabel('Precision (%)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Visualizar predicciones
plot_3d_predictions(X, Y, predictions)

# Visualizar resultados de hiperparámetros
plt.figure(figsize=(12, 6))
plt.bar(range(len(results[:10])), [r['accuracy'] for r in results[:10]])
plt.xlabel('Combinacion de hiperparametros')
plt.ylabel('Precision (%)')
plt.title('Top 10 combinaciones de hiperparametros')
plt.xticks(range(len(results[:10])), [f"{i+1}" for i in range(len(results[:10]))], rotation=45)
plt.tight_layout()
plt.show()

# Crear visualización interactiva para mostrar la frontera de decisión (aproximación)
from matplotlib.animation import FuncAnimation

def plot_decision_boundary_animation(X, Y, W1, b1, W2, b2, W3, b3):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colores para los puntos
    colors = np.array(['blue', 'red'])
    
    # Graficar puntos originales
    ax.scatter(X[0, :], X[1, :], X[2, :], c=colors[Y[0, :].astype(int)], s=30, alpha=0.8)
    
    ax.set_title('Frontera de Decisión en 3D (Rotación)', fontsize=16)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    # Función para actualizar la vista en cada frame
    def update(frame):
        ax.view_init(elev=30, azim=frame)
        return fig,
    
    # Crear animación
    ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 5), interval=100, blit=True)
    plt.close()
    return ani

# Crear animación de la frontera de decisión
animation = plot_decision_boundary_animation(X, Y, W1, b1, W2, b2, W3, b3)
plt.show()
