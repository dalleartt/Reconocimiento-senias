import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Cargar el modelo entrenado
model = load_model('../Redes Neuronales Reconocimiento senias/gesture_recognition_model.keras')  # Cargar el modelo guardado en formato nativo
label_encoder = LabelEncoder()
# Cargar las clases guardadas
classes = np.load('../Redes Neuronales Reconocimiento senias/classes.npy', allow_pickle=True)
print(classes)  # Verifica qué hay en el archivo
label_encoder.classes_ = classes

# Inicializar Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Iniciar la captura de video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen para detectar manos
    result = hands.process(rgb_frame)

    # Verificar si se detectaron manos
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extraer las coordenadas de los puntos de referencia (landmarks)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Convertir a formato de entrada del modelo
            X_test = np.array(landmarks).reshape(1, -1)  # Convertir a array y remodelar para el modelo

            # Realizar la predicción
            prediction = model.predict(X_test)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

            # Mostrar la predicción en la imagen
            cv2.putText(frame, f'Prediccion: {predicted_label[0]}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Dibujar los puntos de referencia en la mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar la imagen con la predicción
    cv2.imshow("Deteccion de gestos", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
hands.close()