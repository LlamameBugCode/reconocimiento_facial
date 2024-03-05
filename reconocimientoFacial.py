import cv2 
import os

dataPath ='C:/Users/bug_code/Documents/PROYECTOS/PYTHON/reconocimiento_facial/data'
imagePaths =os.listdir(dataPath)

face_recognizer = cv2.face.EigenFaceRecognizer_create()
#Leyendo el modelo
face_recognizer.read('modeloEigenFace.xml')

#leer los videos de prueba
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
    ret,frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)
        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

