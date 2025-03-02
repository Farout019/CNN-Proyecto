import cv2

cap = cv2.VideoCapture("http://192.168.100.144:4747/video")  # Reemplaza con la IP correcta

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo obtener la imagen del celular")
        break
    
    cv2.imshow("DroidCam Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
