import cv2
import numpy as np
import face_recognition

uzair = face_recognition.load_image_file('1.jpg')
uzairimg = cv2.cvtColor(uzair, cv2.COLOR_BGR2RGB)
uzairt = face_recognition.load_image_file('3.jpeg')
uzairtest = cv2.cvtColor(uzairt, cv2.COLOR_BGR2RGB)

faceloc  = face_recognition.face_locations(uzairimg)[0]
faceencoding = face_recognition.face_encodings(uzairimg)[0]
cv2.rectangle(uzairimg,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest  = face_recognition.face_locations(uzairtest)[0]
facetestencoding = face_recognition.face_encodings(uzairtest)[0]
cv2.rectangle(uzairtest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

result = face_recognition.compare_faces([faceencoding],facetestencoding)
facedis = face_recognition.face_distance([faceencoding],facetestencoding)

print(result,facedis)
cv2.putText(uzairtest,f'{result} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

# cv2.imshow('Simple Uzair',faceloc)
cv2.imshow('UzairTEST',uzairimg)
# cv2.imshow('...',uzairt)
cv2.imshow('uTEST',uzairtest)
cv2.waitKey(0)
