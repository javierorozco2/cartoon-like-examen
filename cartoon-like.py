import cv2

cap = cv2.VideoCapture(0)


#Ver si la camara funciona correctamente
if not cap.isOpened():
    raise IOError("Error: No se pudo abrir la camara")

while True:
    ret, frame = cap.read()

    #Frame
    frame = cv2.resize(frame, None, fx = 0.9, fy = 0.9, interpolation=cv2.INTER_AREA)

    #Mostrar imagen
    cv2.imshow("Camara", frame)

    # Teclear esc para terminar programa
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()

