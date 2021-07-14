import cv2
import os
import face_recognition
import numpy
from datetime import  datetime

images = []
classimg = []
path = 'data'
mylist = os.listdir(path)

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classimg.append(os.path.splitext(cl)[0])

print(classimg)

def imeEnconding(images):
    encondingimg = []
    for img in images:
        imgg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        imgenc = face_recognition.face_encodings(imgg)[0]
        encondingimg.append(imgenc)
    return encondingimg

def mattendence(name):
    with open('attendence.csv', 'r+') as f:
        mydataList = []
        nameList=[]
        for line in mydataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'{name},{dtString}')



encodedimg = imeEnconding(images)
cap = cv2.VideoCapture(0)
while True:
    sucess, img = cap.read()
    imgg = cv2.resize(img,(0,0), None,0.25,0.25)
    imgg = cv2.cvtColor(imgg,cv2.COLOR_BGR2RGB)
    curfaceloc = face_recognition.face_locations(imgg)
    CurimgEnc = face_recognition.face_encodings(imgg,curfaceloc)

    for encodeFace, faceLoc in zip(CurimgEnc,curfaceloc):
        match = face_recognition.compare_faces(encodedimg,encodeFace)
        dis = face_recognition.face_distance(encodedimg,encodeFace)
        print(dis)
        matchindex = numpy.argmin(dis)

        if match[matchindex]:
            name = classimg[matchindex]
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
            mattendence(name)
    cv2.imshow('Project',img)
    cv2.waitKey(1)


