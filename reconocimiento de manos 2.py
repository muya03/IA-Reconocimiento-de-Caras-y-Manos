import cv2
import mediapipe as mp

#dibuja las formas
mp_dibujar = mp.solutions.drawing_utils
#detecta la mano
mp_manos = mp.solutions.hands

#Video Capturadora
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0)

#algoritmo dado por condicionales y bucles y detección de errores para reconocimiento de manos
with mp_manos.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

        while True:
            ret, rectangulo = cap.read()
            if ret == False:
                break
            height, width, _ = rectangulo.shape
            rectangulo = cv2.flip(rectangulo, 1)
            rectangulo_rgb = cv2.cvtColor(rectangulo, cv2.COLOR_BGR2RGB)
            resultado = hands.process(rectangulo_rgb)
            
            
            #deteccion de errores
            if resultado.multi_hand_landmarks is not None:

                if resultado.multi_hand_landmarks is not None:
                    for hand_landmarks in resultado.multi_hand_landmarks:
                        mp_dibujar.draw_landmarks(
                            rectangulo, hand_landmarks, mp_manos.HAND_CONNECTIONS,
                            mp_dibujar.DrawingSpec(color=(0,255,255), thickness=3, circle_radius=5),
                            mp_dibujar.DrawingSpec(color=(255,0,255), thickness=4, circle_radius=5))
            cv2.imshow('Rectangulo',rectangulo)
            #usar ESC para salir
            if cv2.waitKey(1) & 0xFF == 27:
                break




