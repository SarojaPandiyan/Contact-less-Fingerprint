import cv2
import os
import fingerprint_enhancer
import csv
import tkinter as tk
import datetime
from datetime import date
#display the text
def display(message):
    window=tk.Tk()
    window.minsize(480,360)
    window.config(bg="green")
    window.title("Fingerprint Authentication Module")
    label=tk.Label(window,text=message,font=("BankGothic Md BT",50,"bold"),bg="green",fg="white")
    label.pack(padx=150,pady=150)
    window.mainloop()

#retrieve the person name from the image
def person(path):
    with open('F:/Sem_3/Design Thinking/Implementation/db.csv', mode ='r')as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            if(path==lines[1]):
                return lines[0]
    return "Unidentified"

def logwriter(person):
    current_date=date.today().strftime("%d/%m/%Y")
    current_time=datetime.datetime.now().strftime("%H:%M:%S")
    with open("F:/Sem_3/Design Thinking/Implementation/log.csv",mode='+a') as file:
        file.write(",,,")
        if(person!="Unidentified"):
            
            file.write(f"\n{current_date},{current_time},{person},AUTHENTICATION SUCCESSFUL")
        else:
            file.write(f"\n{current_date},{current_time},{person},AUTHENTICATION UNSUCCESSFUL")
        file.write(",,,")
    file.close()

#reading the sample image in question
sample1=cv2.imread('F:/Sem_3/Design Thinking/Implementation/dataset_1/dataset/real_data/mysterychap.jpg',0)

#enhancing the image
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))  
low = cv2.morphologyEx(sample1, cv2.MORPH_OPEN, se)  
high = cv2.morphologyEx(sample1, cv2.MORPH_CLOSE, se) 
sample1 = (sample1 - low) / (high - low + 1e-6)
sample = fingerprint_enhancer.enhance_Fingerprint(sample1)

#if sample image or enhanced image is not found
if sample is None or sample1 is None:
    print(f"Error loading sample")

#initializing score,file details, key points and matching points
best_score=0
filename=None
image=None
kp1,kp2,mp=None,None,None

#starting the loop to match the sample with other images in the database
for file in [file for file in os.listdir("F:/Sem_3/Design Thinking/Implementation/new_dataset/")]:
    fingerprint_image=cv2.imread("F:/Sem_3/Design Thinking/Implementation/new_dataset/"+file,0)	
    if fingerprint_image is None:
        print(f"Error loading image")	
    sift=cv2.SIFT_create()

    fingerprint_image = cv2.normalize(fingerprint_image, None, 0, 255, cv2.NORM_MINMAX)
    sample = cv2.normalize(sample, None, 0, 255, cv2.NORM_MINMAX)

    keypoints_1,descriptors_1=sift.detectAndCompute(sample,None)
    keypoints_2,descriptors_2=sift.detectAndCompute(fingerprint_image,None)

    matches=cv2.FlannBasedMatcher({'algorithm':1,'trees':10},{}).knnMatch(descriptors_1,descriptors_2,k=2)
    match_points=[]

    for p,q in matches:
        if p.distance < 0.1*q.distance:
           match_points.append(p)
        
    keypoints=0
    if len(keypoints_1)<len(keypoints_2):
        keypoints=len(keypoints_1)
    else:
        keypoints=len(keypoints_2)
        
    if len(match_points)/keypoints * 100 > best_score:
        best_score=len(match_points)/keypoints * 100
        filename=file
        image=fingerprint_image
        kp1,kp2,mp=keypoints_1,keypoints_2,match_points
if filename is not None:
    match=person("F:/Sem_3/Design Thinking/Implementation/new_dataset/"+filename)
    if(match!="Unidentified"):
        display(f"WELCOME, {match}!")
    else:
        display("NO MATCH FOUND")
    logwriter(match)
else:
    print("Error with image")


