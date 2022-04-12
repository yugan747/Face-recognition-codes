import os
import cv2
import face_recognition
import numpy as np


path = 'Imagesoffamous'
images = []
className = []
mylist = os.listdir(path)
print(mylist)

for c1 in mylist:
    curImg = cv2.imread(f'{path}/{c1}')
    images.append(curImg)
    className.append(os.path.splitext(c1)[0])
print(className)    

def findEncoding(images):
    encodelist=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodeListKnown = findEncoding(images)
print(encodeListKnown)
print('encoding complete')   


TestImages = cv2.imread('TestImage/elonmusk123.jpg')
 

TestImage = cv2.cvtColor(TestImages,cv2.COLOR_BGR2RGB)
facesImagesT = face_recognition.face_locations(TestImage)
encodeImagesT = face_recognition.face_encodings(TestImage )
print(encodeImagesT)
faceDis = face_recognition.face_distance(np.array(encodeListKnown), np.array(encodeImagesT))
print(faceDis)
matches = face_recognition.compare_faces(np.array(encodeListKnown), np.array(encodeImagesT))
print(matches)
matchIndex=np.argmin(faceDis)
print(matchIndex)

if matches[matchIndex]:
    name = className[matchIndex].upper()
    print(name)
    for face in facesImagesT:
        y1,x2,y2,x1 = face
        cv2.rectangle(TestImages,(x1,y1),(x2,y2),(0,255,0),2)
        
        cv2.putText(TestImages,name,(x1,y2+26),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

cv2.imshow('testimage',TestImages)    
cv2.waitKey(0)





