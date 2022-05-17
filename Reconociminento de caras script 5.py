import cv2 as cv
potencia_de_rectangulo = 255

#Capturar video

captura_de_video = cv.VideoCapture(0)

# Cascade. Extraidos del modulo directamente, no como archivo. Los archivos son los preterminados.

sonrisa_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_smile.xml")
cara_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
ojo_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")


#EL RECTANGULO

def dibujar_caras_detectadas(detectada, image, color: tuple):

    for(x, y, width, height) in detectada:

        a = x + width
        b = y + height
        
        cv.rectangle(image,(x, y),(a, b),color,thickness=3)

# Bucle del algoritmo        
while True:

    #leer el rectangulo, y arreglo del modulo str, StackOverFlow

    _, rectangulo = captura_de_video.read()

    # rectangulo detecta escala de grises
    
    escala_de_grises = cv.cvtColor(rectangulo, cv.COLOR_BGR2GRAY)

    #deteccion de vecinos, anulacion de fantasmas. (La boca sigue teniendo falsos positivos ; MIRAR)
    
    sonrisa_detectado = sonrisa_cascade.detectMultiScale(image = escala_de_grises, scaleFactor=1.9, minNeighbors = 25)
    caras_detectadas = cara_cascade.detectMultiScale(image = escala_de_grises, scaleFactor=1.3, minNeighbors = 4)
    ojo_detectado = ojo_cascade.detectMultiScale(image = escala_de_grises, scaleFactor=1.3, minNeighbors = 4)

    #Poner la cara
    dibujar_caras_detectadas(caras_detectadas, rectangulo, (0, 0, potencia_de_rectangulo))
    dibujar_caras_detectadas(ojo_detectado, rectangulo, (0, potencia_de_rectangulo, 0))
    dibujar_caras_detectadas(sonrisa_detectado, rectangulo, (potencia_de_rectangulo, 0, 0))

    #Ventana de la camara

    cv.imshow("Detección de Cara", rectangulo)

    # Usar ESC para salir. Esto es un proceso de salida, necesario para empezar el main loop
    if cv.waitKey(1) == 27:
        break

#seguir procedimieno del mainloop y cerrando la ventana
    
video_capture.release()

cv.desrtoyAllWindows()
    
    
    

    
    



        
