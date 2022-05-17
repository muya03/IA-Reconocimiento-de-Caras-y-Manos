import cv2
import mediapipe as mp

def dibujar_caras_detectadas(detectada, image, color: tuple):

    for(x, y, width, height) in detectada:
        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            color,
            thickness=2
        )

def manos(mp_dibujar,mp_manos,captura_de_video):
    with mp_manos.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

        while True:
            ret, rectangulo = captura_de_video.read()
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
            #cv2.imshow('Rectangulo',rectangulo)
            #usar ESC para salir
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
#dibuja las formas
mp_dibujar = mp.solutions.drawing_utils
#detecta la mano
mp_manos = mp.solutions.hands

#Capturar video

captura_de_video = cv2.VideoCapture(0)

# Creación de objetos cascade

cara_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
ojo_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
sonrisa_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

while True:

    #  rectangulo de la cara
    _, rectangulo = captura_de_video.read()
    # rectangulo detecta escala de grises
    escala_de_grises = cv2.cvtColor(rectangulo, cv2.COLOR_BGR2GRAY)

    #detectar mas de una cara

    caras_detectadas = cara_cascade.detectMultiScale(image = escala_de_grises, scaleFactor=1.3, minNeighbors = 4)
    ojo_detectado = ojo_cascade.detectMultiScale(image = escala_de_grises, scaleFactor=1.3, minNeighbors = 4)
    sonrisa_detectado = sonrisa_cascade.detectMultiScale(image = escala_de_grises, scaleFactor=1.9, minNeighbors = 25)
    dibujar_caras_detectadas(caras_detectadas, rectangulo, (0, 0, 255))
    dibujar_caras_detectadas(ojo_detectado, rectangulo, (0, 255, 0))
    dibujar_caras_detectadas(sonrisa_detectado, rectangulo, (255, 0, 0))
    manos(mp_dibujar,mp_manos,captura_de_video)

    #Hacer la vnetana del directo

    cv2.imshow("Detección de Cara", rectangulo)

    # Usar ESC para salir
    if cv2.waitKey(1) == 27:
        break


#hacer el mainloop
    
video_capture.release()
cv2.desrtoyAllWindows()
    
    
    

    
    



        
