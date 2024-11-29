import cv2
import csv
import mediapipe as mp
import os

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils



filename = '../Redes Neuronales Reconocimiento senias/hand_landmarks.csv'

file_exists = os.path.isfile(filename)

# Abre el archivo CSV para guardar los datos
with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Escribe el encabezado solo si el archivo no existe
    if not file_exists:
    # Escribe el encabezado: X, Y, Z para cada uno de los 21 puntos de referencia + Etiqueta
        header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
        header.append("label")
        writer.writerow(header)

    # Captura de video
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
    else:
        print("Cámara abierta correctamente.")

    print("Presiona 'q' para salir. Presiona la letra en el teclado para etiquetar y guardar.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir la imagen al formato RGB (MediaPipe usa RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Si se detectan manos, captura los puntos de referencia
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibuja las conexiones de la mano
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Captura las coordenadas de los puntos de referencia
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                # Muestra la imagen con la mano detectada
                cv2.imshow("Recolector senas", frame)

                # Espera a que se presione una tecla para etiquetar
                key = cv2.waitKey(1) & 0xFF
                if key == ord('1'):
                    cap.release()
                    break
                elif key == ord('2'):  # Presiona 'd' para borrar la última línea
                    if os.path.isfile(filename):
                        with open(filename, mode='r') as read_file:
                            lines = read_file.readlines()
                        if len(lines) > 1:  # Verifica que haya más de una línea (encabezado)
                            with open(filename, mode='w', newline='') as write_file:
                                write_file.writelines(lines[:-1])  # Escribe todo menos la última línea
                            print("Última línea borrada.")
                        else:
                            print("No hay líneas para borrar.")
                    else:
                        print("El archivo no existe.")
                elif key != 255:
                    # Guarda las coordenadas junto con la etiqueta (letra presionada)
                    label = chr(key).upper()
                    landmarks.append(label)
                    writer.writerow(landmarks)
                    print(f"Datos guardados para la letra '{label}'")

    # Cierra el archivo y libera la cámara
cap.release()
cv2.destroyAllWindows()
