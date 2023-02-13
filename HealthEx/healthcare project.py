#actual one for epitomo
import cv2
import face_recognition as fr
import os
import numpy as np
import pyttsx3
import pandas as pd
import keyboard
import time
# Test Images = all faces that this program can recognise 
path=r"D:\HealthEx\Epitome\All imgs train"
imgtests=[]
imgnames=[]
allimgtest=os.listdir(path)
print(allimgtest)
for i in allimgtest:
    j=cv2.imread(f'{path}\{i}')
    imgtests.append(j) #actual test imgs lst
    imgnames.append(i[0:-4])
print(imgnames)

def findenco(imgs):
    encodelst=[]
    for i in imgtests:
        i=cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        encode=fr.face_encodings(i)[0]
        encodelst.append(encode)
    return encodelst

tstencodes=findenco(imgtests) #All encoded list of test images
print("Total imgs Encoded: ",len(tstencodes))

#function to talk
def talk(str1):
    voice=pyttsx3.init()
    voice.say(str1)
    voice.runAndWait()

#All options
print("Press:")
print("1. for Person Detection")
op= int(input())

dum=[]
if(op==1):
    cap=cv2.VideoCapture(1)
    while True:
        _,k=cap.read()  #k = current frrame
        ks=cv2.resize(k,(0,0),None,0.25,0.25)  #ks= k_small
        ks=cv2.cvtColor(ks,cv2.COLOR_BGR2RGB)

        encodeloc=fr.face_locations(ks)
        encodeks=fr.face_encodings(ks,encodeloc)
        
        if keyboard.is_pressed("q"): #press q to stop video recording
            break
       
        voi=0
        for encode1,encodeloc1 in zip(encodeks,encodeloc):
            matches=fr.compare_faces(tstencodes,encode1)
            dis=fr.face_distance(tstencodes,encode1)
            #print(dis)   #dis=all matches with all images min value is closest match
            pos=np.where(dis==min(dis))[0][0]
            #print(imgnames[pos],pos,min(dis)) #(name,pos,matchval)

            dum.append(imgnames[pos])

            for i in dum:
                if dum.count(i)>=3:
                    voi=1
                    print(imgnames[pos],pos,min(dis)) #(name,pos,matchval)
                    print(imgnames[pos],pos,min(dis)) #(name,pos,matchval)
                    talk(str(imgnames[pos]))
                    break
            if voi==1:
                    break
        if voi==1:
                    break



csv = imgnames[pos] +".csv"
path1=f'D:\HealthEx\Epitome\j epitome patient past\{csv}'
xl1=pd.read_csv(path1)
print("\n",csv," Medical History")
print("\n",xl1)
# cv2.imshow("Hello",)
# cv2.waitKey(0)