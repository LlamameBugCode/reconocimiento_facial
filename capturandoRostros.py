import cv2
import imutils
import os



personame = "isaac"
dataPath = "C:/Users/bug_code/Documents/PROYECTOS/PYTHON/reconocimiento_facial/data"
personPath = dataPath + '/'+ personame

if not os.path.exists(personPath):
    print("Carpeta creada: ", personPath)
    os.makedirs(personPath)

#video de donde extraeeremos los rostros
cap = cv2.VideoCapture('C:/Users/bug_code/Documents/PROYECTOS/PYTHON/reconocimiento_facial/imagenesYVideosDePrueba/isaac_video.mp4')

#iniciamos el detector de rostros
faceClassif=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count=0

#Leyendo cada fotrogrtama del video y redimencionar en caso de que el tamaño sea muy grande
while True:
    ret,frame = cap.read()
    if ret == False:break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
        count = count + 1
    cv2.imshow('frame',frame)


    k = cv2.waitKey(1)
    if k == 27 or count>=2000:
        break

cap.release()
cv2.destroyAllWindows()





