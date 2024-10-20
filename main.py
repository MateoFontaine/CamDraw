import cv2
import mediapipe as mp
import numpy as np

# Inicializa MediaPipe para la detección de manos
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Colores de las pelotas
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0)]  # Rojo, Verde, Azul, Amarillo, Naranja
ball_radius = 20
ball_positions = [(50 + i * 100, 30) for i in range(5)]  # Posiciones de las pelotas
red_ball_position = (50, 30)  # Posición de la pelota roja

# Configura la captura de video
cap = cv2.VideoCapture(0)

# Define el lienzo para dibujar
drawing_canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Variable para el color de dibujo
current_color = (255, 255, 255)  # Color por defecto blanco
brush_size = 10  # Grosor inicial del pincel

# Variables para suavizar el trazo
last_x_index, last_y_index = None, None

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No se puede abrir la cámara.")
            break

        # Convierte la imagen a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Invierte la imagen horizontalmente para el efecto espejo
        image = cv2.flip(image, 1)

        image.flags.writeable = False

        # Realiza la detección de manos
        results = hands.process(image)

        # Convierte la imagen de vuelta a BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Dibuja las pelotas de colores
        for i, (pos, color) in enumerate(zip(ball_positions, colors)):
            cv2.circle(image, pos, ball_radius, color, -1)

        # Dibuja la pelota roja que indica el color actual
        cv2.circle(image, red_ball_position, ball_radius, current_color, -1)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibuja los puntos de la mano
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Obtiene las coordenadas del dedo índice y del pulgar
                index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_finger = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                h, w, _ = image.shape
                x_index, y_index = int(index_finger.x * w), int(index_finger.y * h)
                x_thumb, y_thumb = int(thumb.x * w), int(thumb.y * h)

                # Cambia el color si el dedo índice toca una pelota
                for pos, color in zip(ball_positions, colors):
                    if (x_index - pos[0]) ** 2 + (
                            y_index - pos[1]) ** 2 < ball_radius ** 2:  # Distancia al centro de la pelota
                        current_color = color

                # Verifica si la mano está abierta
                if (index_finger.y < thumb.y and  # Dedo índice arriba del pulgar
                        index_finger.y < middle_finger.y and  # Dedo índice arriba del medio
                        index_finger.y < ring_finger.y and  # Dedo índice arriba del anular
                        index_finger.y < pinky.y):  # Dedo índice arriba del meñique

                    # Suavizado del trazo
                    if last_x_index is not None and last_y_index is not None:
                        # Dibuja líneas entre la última posición y la nueva posición
                        cv2.line(drawing_canvas, (last_x_index, last_y_index), (x_index, y_index), current_color,
                                 brush_size)

                    # Dibuja en el lienzo si el dedo índice está parado
                    cv2.circle(drawing_canvas, (x_index, y_index), brush_size, current_color, -1)

                    # Actualiza la última posición del dedo índice
                    last_x_index, last_y_index = x_index, y_index
                else:
                    # Restablece la última posición si la mano está abierta
                    last_x_index, last_y_index = None, None

        # Muestra la imagen con los dibujos
        combined_image = cv2.addWeighted(image, 0.5, drawing_canvas, 0.5, 0)
        cv2.imshow('Dibujo con la mano (Espejo)', combined_image)

        # Sale si se presiona 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
