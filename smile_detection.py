import cv2

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
webcam = cv2.VideoCapture(0)
while True:
    success, frame = webcam.read()
    if not success:
        break

    #change to Grayscale
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #Detect Faces
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor = 1.035, minNeighbors = 5)
    #Detect Smiles


    #print(faces)
    #Face Detection
    for (x, y, w, h) in faces:
        face = frame[y:y+h,x:x+w];
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,200,0), 4)
        gray_face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        smiles = smile_detector.detectMultiScale(gray_face, scaleFactor = 1.7, minNeighbors = 20)
        #Find all smile within face
        for (x_, y_, w_, h_) in smiles:
            # cv2.rectangle(face, (x_,y_), (x_+w_,y_+h_), (0,0,200), 4)
            if len(smiles) > 0:
                cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale = 3, fontFace = cv2.FONT_HERSHEY_PLAIN, color = (255,255,255))


    cv2.imshow('Smile Detector', frame)
    cv2.waitKey(1)


#CleanUp
webcam.release()
cv2.destroyAllWindows()
