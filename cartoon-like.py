import numpy as np
import cv2


#Función para caricaturizar imagen
def cartoonize(img, sketch_mode=False):

    #Variables
    ksize=5
    num_repetitions=10
    sigma_color=5
    sigma_space=7
    ds_factor = 4 

    # Convertir imagen a esacala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Modificación a filtro 
    img_gray = cv2.medianBlur(img_gray, 7) 

    # Detección de bordes 
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5) 
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV) 
 
    # Mask dependiendo el modo
    if sketch_mode: 
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
 
    #Cambiar tamaño para optimizar
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)
 
    # Aplicar filtro bilateral
    for i in range(num_repetitions): 
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space) 
 
    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR) 
    dst = np.zeros(img_gray.shape) 
    dst = cv2.bitwise_and(img_output, img_output, mask=mask) 

    return dst 

#Función principal
if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    cur_char = -1
    prev_char = -1

    #Ver si la camara funciona correctamente
    if not cap.isOpened():
        raise IOError("Error: No se pudo abrir la camara")

    while True:
        ret, frame = cap.read()

        #Frame
        frame = cv2.resize(frame, None, fx = 0.9, fy = 0.9, interpolation=cv2.INTER_AREA)

        # #Mostrar imagen
        # cv2.imshow("Camara", frame)

        # Evento click
        c = cv2.waitKey(1)
        if c == 27:
            break

        #Condición de control
        if c > -1 and c != prev_char:
            cur_char = c
        prev_char = c

        if cur_char == ord('s'):
            cv2.imshow("Caricaturizacion", cartoonize(frame ,sketch_mode=True))
            cv2.imwrite('img-gris.png', cartoonize(frame ,sketch_mode=True))
            print("Imagen en color gris capturada correctamente")
            break

        elif cur_char == ord('c'):
            cv2.imshow("Caricaturizacion", cartoonize(frame, sketch_mode=False))
            cv2.imwrite('img-color.png', cartoonize(frame, sketch_mode=False))
            print("Imagen a color capturada correctamente")
            break

        else:
            cv2.imshow("Caricaturizacion", frame)
        

    cap.release()
    cv2.destroyAllWindows()

