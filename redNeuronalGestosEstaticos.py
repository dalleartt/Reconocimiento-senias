import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Paso 1: Cargar los datos desde el archivo CSV
data = pd.read_csv('../Redes Neuronales Reconocimiento senias/hand_landmarks.csv')
print('Pasa carga de datos del csv')

# Separar las características (X) y las etiquetas (y)
X = data.iloc[:, :-1].values.astype('float32')  # Todas las columnas excepto la última
y = data.iloc[:, -1].values  # Solo la última columna (etiqueta)
print ('Divide x e y')


# Paso 2: Codificar las etiquetas (letras) a números
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Codificacion")

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Dividir datos")


# Construimos arquitectura de la red neuronal
model = Sequential()
# Capa de entrada -RELU
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
# Capa oculta - RELU
model.add(Dense(64, activation='relu'))
# Capa de salida - SOFTMAX
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

print ("Construye red neuronal")

# Compilamos el modelo
#Algoritmo de optimizacion: Adam (Adaptive Moment Estimation)
#Funcion d e perdida:  categorical cross entropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])




print ("compilacion")

# Entrenamos la red neuronal
# Epoch: 50 (La cantidad de veces que se va pasar por los valores de entrenamiento)
# Batch: 32 (Es la cantidad de datos que va agarrar hasta llegar
#           la totalidad de los datos de entrenamiento)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

print ("entrenamiento")

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Precisión en el conjunto de prueba: {accuracy * 100:.2f}%')
model.save('gesture_recognition_model.keras')
np.save('../Redes Neuronales Reconocimiento senias/classes.npy', label_encoder.classes_)
print("Modelo guardado")
