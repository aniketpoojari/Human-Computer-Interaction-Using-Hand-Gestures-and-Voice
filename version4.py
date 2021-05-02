# """
# @author: Viet Nguyen <nhviet1009@gmail.com>
# """

# IMPORTS==========
import shutil
import pyttsx3
import speech_recognition as sr
import datetime
import wikipedia
import webbrowser
import os
import pyautogui
import numpy as np
import cv2
import math
from tkinter import *
import threading
import tensorflow as tf
from src.utils import load_graph, detect_hands, predict


#VISION SETTINGS
def nothing(x):
    pass

font = cv2.FONT_HERSHEY_SIMPLEX
graph, sess = load_graph("src/pretrained_model.pb")
def keypress(p):
    y = p[1]
    x = p[0]
    if y >= 100 and y <= 150:
        if x >= 330 and x <= 380:
            pyautogui.typewrite('A')
        elif x >= 380 and x <= 430:
            pyautogui.typewrite('B')
        elif x >= 430 and x <= 480:
            pyautogui.typewrite('C')
        elif x >= 480 and x <= 530:
            pyautogui.typewrite('D')
        elif x >= 530 and x <= 580:
            pyautogui.typewrite('E')
        elif x >= 580 and x <= 630:
            pyautogui.typewrite('F')
    elif y >= 150 and y <= 200:
        if x >= 330 and x <= 380:
            pyautogui.typewrite('G')
        elif x >= 380 and x <= 430:
            pyautogui.typewrite('H')
        elif x >= 430 and x <= 480:
            pyautogui.typewrite('I')
        elif x >= 480 and x <= 530:
            pyautogui.typewrite('J')
        elif x >= 530 and x <= 580:
            pyautogui.typewrite('K')
        elif x >= 580 and x <= 630:
            pyautogui.typewrite('L')
    elif y >= 200 and y <= 250:
        if x >= 330 and x <= 380:
            pyautogui.typewrite('M')
        elif x >= 380 and x <= 430:
            pyautogui.typewrite('N')
        elif x >= 430 and x <= 480:
            pyautogui.typewrite('O')
        elif x >= 480 and x <= 530:
            pyautogui.typewrite('P')
        elif x >= 530 and x <= 580:
            pyautogui.typewrite('Q')
        elif x >= 580 and x <= 630:
            pyautogui.typewrite('R')
    elif y >= 250 and y <= 300:
        if x >= 330 and x <= 380:
            pyautogui.typewrite('S')
        elif x >= 380 and x <= 430:
            pyautogui.typewrite('T')
        elif x >= 430 and x <= 480:
            pyautogui.typewrite('U')
        elif x >= 480 and x <= 530:
            pyautogui.typewrite('V')
        elif x >= 530 and x <= 580:
            pyautogui.typewrite('W')
        elif x >= 580 and x <= 630:
            pyautogui.typewrite('X')
    elif y >= 300 and y <= 350:
        if x >= 330 and x <= 380:
            pyautogui.typewrite('Y')
        elif x >= 380 and x <= 430:
            pyautogui.typewrite('Z')
#==========================================

#VOICE SETTINGS
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

text = []

def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    
def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour >= 0 and hour < 12:
        speak("Good Morning!")
        
    elif hour >= 12 and hour < 18:
        speak("Good Afternoon!")

    else:
        speak("Good Evening!")

    speak("I'm Friday, Please tell me how may i help you")

def takeCommand():
    # text.append("Listening...")
    # inp = input().lower()
    # text.append("Recognizing...")
    # text.append("User said: " + inp)
    # text.append("")
    # return inp

    r = sr.Recognizer()
    with sr.Microphone() as source:
        text.append("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        text.append("Recognizing...") 
        query = r.recognize_google(audio, language='en-in')   
        text.append("User said: " + query)
        text.append("")

    except Exception as e:
        text.append("Say that again please....")
        text.append("")
        return "0"

    return query

def permission(t):
    speak("YOU SURE")
    text.append("ARE YOU SURE YOU WANT TO DELETE " + t)
    p = takeCommand().lower()
    if p == "yes":
        return 1
    elif p == "no":
        return 0
    else:
        return 0

def delete(d, name):
    if len(name.split(".")) == 2:
        if(permission("FILE")):
            os.remove(d + name)
            speak("FILE DELETED")
            text.append("FILE REMOVED SUCCESSFULLY: " + name)
            text.append("AT DIRECTORY: " + d )
            text.append("")
        else:
            speak("SKIPPED")
            text.append("DELETE SKIPPED" )
            text.append("")
    else:
        try:
            os.rmdir(d + name)
            os.mkdir(d + name)
            if(permission("EMPTY FOLDER")):
                os.rmdir(d + name)
                speak("FOLDER DELETED")
                text.append("FOLDER REMOVED SUCCESSFULLY: " + name)
                text.append("AT DIRECTORY: " + d )
                text.append("")
            else:
                speak("SKIPPED")
                text.append("DELETE SKIPPED" )
                text.append("")
        except:
            if(permission("FILLED FOLDER")):
                shutil.rmtree(d + name)
                speak("FOLDER DELETED")
                text.append("FOLDER REMOVED SUCCESSFULLY: " + name)
                text.append("AT DIRECTORY: " + d )
                text.append("")
            else:
                speak("SKIPPED")
                text.append("DELETE SKIPPED" )
                text.append("")

class GUI:
    def __init__(self):
        self.index = 0
        self.root = Tk()
        self.root.title("YOUR PERSONAL ASSISSTANT")
        self.frame = Frame(self.root)
        self.frame.pack()
        self.listNodes = Listbox(self.frame, width=60, height=30, font=("Helvetica", 12))
        self.scrollbar = Scrollbar(self.frame, orient="vertical")
        self.scrollbar.config(command=self.listNodes.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.listNodes.config(yscrollcommand=self.scrollbar.set)
        self.listNodes.pack(fill="y")
        self.refresh_text()
        self.root.mainloop()
    def refresh_text(self):
        try:
            self.listNodes.insert(END, text[self.index])            
            self.listNodes.yview(END)
            if(text[self.index] == "User said: hand"):
                self.root.destroy()
            self.index += 1
        except:
            pass
        self.root.after(100, self.refresh_text)
#===========================================================================


mode = 1


while(1):
    if(mode == 1):

        t1 = threading.Thread(target=GUI) 
        t1.start()

        directory = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        l = directory.split("\\")

        wishMe()

        while(1):

            query = takeCommand().lower()
            query = query.split()

            if query[0] == "hand":
                speak("SWITCHING TO VISION PART")
                try:
                    Thread.sleep(1000)
                except:
                    pass
                t1.join()
                mode = 0
                break

            if query[0] == "type":
                query = query[1:]
                d = ""
                for i in query:
                    d = d + " " + i 
                pyautogui.typewrite(d)

            if query[0] == "space":
                 pyautogui.press('space')

            if query[0] == "return":
                pyautogui.press('enter')     



            if query[0] == "wikipedia":
                    speak('Searching Wikipedia...')
                    query = query[1:]
                    d = ""
                    for i in query:
                        d = d + " " + i
                    results = wikipedia.summary(d, sentences=2)
                    speak("According to wikipedia")
                    text.append(results )
                    text.append("")
                    speak(results)

            if query[0] == "browse":
                if query[1] == "youtube":
                    webbrowser.open("youtube.com")
                elif query[1] == "google":
                    webbrowser.open("google.com")
                elif query[1] == "stackoverflow":
                    webbrowser.open("stackoverflow.com")
            
            if query[0] == "open":
                d = ""
                for i in l:
                    d = d + i + "/"
                l1 = os.listdir(d)
                l2 = []
                for j in l1:
                    if query[1] == j.split(".")[0].lower():
                        l2.append(j)
                if len(l2) != 0:
                    if len(l2) == 1:
                        if len(l2[0].split(".")) == 2:
                            speak("FILE OPENED")
                            text.append("OPENING FILE: " + l2[0])
                            text.append("AT DIRECTORY: " + d )
                            text.append("")
                            os.startfile(d + l2[0])
                        else:
                            speak("FOLDER OPENED")    
                            text.append("OPENING FOLDER: " + l2[0])
                            text.append("AT DIRECTORY: " + d )
                            text.append("")
                            l.append(l2[0])
                            d = d + l[-1] + "/"
                            os.startfile(d)

                    else:
                        for k in range(0, len(l2)):
                            text.append(str(k+1) + ". " + l2[k])
                        # m = int(input())
                        m = int(takeCommand().lower())
                        if m > 0 and m <= len(l2):
                            if len(l2[m-1].split(".")) == 2:
                                speak("FILE OPENED")
                                text.append("OPENING FILE: " + l2[m-1])
                                text.append("AT DIRECTORY: " + d )
                                text.append("")
                                os.startfile(d + l2[m-1])
                            else:
                                speak("FOLDER OPENED")    
                                text.append("OPENING FOLDER: " + l2[m-1])
                                text.append("AT DIRECTORY: " + d )
                                text.append("")
                                l.append(l2[m-1])
                                d = d + l[-1] + "/"
                                os.startfile(d)
                        else:
                            speak("SKIPPED")
                            text.append("OPEN SKIPPED" )
                            text.append("")

                else:
                    text.append("NO SUCH FILE OR DIRECTORY" )
                    text.append("")
                    speak("NO SUCH FILE OR DIRECTORY")



            if query[0] == "back":
                del l[-1]
                d = ""
                for i in l:
                    d = d + i + "/"
                speak("GOING BACK")
                text.append("GOING BACK TO DIRECTORY:" + d )
                text.append("")
                os.startfile(d)
                
            if query[0] == "delete":
                d = ""
                for i in l:
                    d = d + i + "/"
                l1 = os.listdir(d)
                l2 = []
                for j in l1:
                    if query[1] == j.split(".")[0].lower():
                        l2.append(j)
                if len(l2) != 0:
                    if len(l2) == 1:
                        delete(d, l2[0])
                    else:
                        for k in range(0, len(l2)):
                            text.append(str(k+1) + ". " + l2[k])
                        # m = int(input())
                        m = int(takeCommand().lower())
                        if m > 0 and m <= len(l2):
                            delete(d, l2[m-1])
                        else:
                            speak("SKIPPED")
                            text.append("DELETE SKIPPED" )
                            text.append("")

                else:
                    speak("NO SUCH FILE OR DIRECTORY")
                    text.append("NO SUCH FILE OR DIRECTORY" )
                    text.append("")
            
            if query[0] == "create":
                d = ""
                for i in l:
                    d = d + i + "/"
                l2 = [".txt", ".docx", ".ppt"]
                text.append("1. folder")
                for j in range (0, len(l2)):
                    text.append(str(j+2) + ". " + l2[j])
               	try:
                    m = int(takeCommand().lower())
                except:
                    m = 0
                if m == 1:
                    os.mkdir(d + query[1])
                    speak("CREATED FOLDER")
                    text.append("FOLDER CREATED SUCCESSFULLY: " + query[1])
                    text.append("AT DIRECTORY: " + d )
                    text.append("")
                elif m > 1 and m <= len(l2) + 1:
                    open(d + query[1] + l2[m-2], 'a').close()
                    speak("CREATED FILE")
                    text.append("FILE CREATED SUCCESSFULLY: " + query[1] + l2[m-2])
                    text.append("AT DIRECTORY: " + d )
                    text.append("")
                else:
                    speak("SKIPPED")
                    text.append("CREATE SKIPPED" )
                    text.append("")

            if query[0] == "current" and query[1] == "directory":
            	d = ""
            	for i in l:
            		d = d + i + "/"
            	text.append(d)
            	text.append("")
            	speak("HERE")

            if "time" in query:
            	now = datetime.datetime.now()
            	current_time = now.strftime("ITS %H HOURS %M MINUTES")
            	text.append(current_time)
            	text.append("")
            	speak(current_time)

    else:
        text = []

        vc = cv2.VideoCapture(0)
        vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        count_last = 0

        cv2.namedWindow("hand",flags=cv2.WINDOW_NORMAL)
        cv2.createTrackbar("upper","hand",0,255,nothing)
        cv2.createTrackbar("lower","hand",0,255,nothing)

        while(1):
            try:
	            #START SEGMENTING SKIN COLOR 
	            ret,frame = vc.read()  
	            frame = cv2.flip(frame, 1)
	            
	            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	            boxes, scores, classes = detect_hands(frame, graph, sess)
	            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	            results = predict(boxes, scores, classes, 0.6, 640, 480)

	            if len(results) == 1:
	                H = frame.shape[0]
	                W = frame.shape[1]
	                black = frame.copy()
	                cv2.rectangle(black, (0, 0), (W, H), (0,0,0), -1)
	                x_min, x_max, y_min, y_max, _ = results[0]
	                crop = frame[y_min:y_max, x_min:x_max]
	                black[y_min:y_max, x_min:x_max] = crop
	                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,0,0), 2)


	                # cv2.rectangle(frame,(450,270),(452,272),(0,255,0),0)
	                # if(cv2.waitKey(20) == 32):
	                	# bgr = frame[271, 451]
	                	# print("hi")

	                xs = int((x_min + x_max)/2)
	                ys = int((y_min + y_max)/2)
	                bgr = frame[ys, xs]

	                ub = cv2.getTrackbarPos("upper","hand")
	                lb = cv2.getTrackbarPos("lower","hand")    

	                high = [ bgr[0] + ub, bgr[1] + ub, bgr[2] + ub]
	                if high[0] > 255:
	                    high[0] = 255 
	                if high[1] > 255:
	                    high[1] = 255
	                if high[2] > 255:
	                    high[2] = 255
	                low = [ bgr[0] - lb, bgr[1] - lb, bgr[2] - lb]

	                hand_lower = np.array(low)                         
	                hand_upper = np.array(high)

	                mask = cv2.inRange(black,hand_lower,hand_upper)
	                kernel = np.ones((7,7),np.uint8)
	                mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)      # Performing Open operation (Increasing the white portion) to remove the noise from image 
	                mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)     # Performing the Close operation (Decreasing the white portion)
	                mask = cv2.bilateralFilter(mask,5,75,75)
	                mask = cv2.bitwise_and(black, black, mask = mask)
	                cv2.imshow("images1", mask)

	                mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
	                ret, mask = cv2.threshold(mask, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	                cv2.imshow("images", mask)
	                #END SEGMENTING SKIN COLOR

	                #START FINDING HAND
	                _,contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	                drawing = np.zeros(frame.shape,np.uint8)
	                comax=0
	                ci = 0
	                for i in range(len(contours)):
	                    cnt = contours[i]
	                    area = cv2.contourArea(cnt)
	                    if area>comax: 
	                        comax = area
	                        ci = i
	                cnt = contours[ci]
	                epsilon = 0.25*cv2.arcLength(cnt,True)                  # Further trying to better approximate the contour by making edges sharper and using lesser number of points to approximate contour cnt.
	                approx = cv2.approxPolyDP(cnt,epsilon,True)
	                #END FINDING HAND

	                #hull = cv2.convexHull(cnt,returnPoints=True) 
	                #cv2.drawContours(frame,[cnt],0,(255,0,0),3)
	                #cv2.drawContours(frame,[hull],0,(0,255,0),3)
	                hull = cv2.convexHull(cnt, returnPoints= False)     
	                defects = cv2.convexityDefects(cnt,hull)
	            
	                #STAR FINDING CENTROID
	                centroid = ()
	                moment = cv2.moments(contours[ci])
	                if moment['m00'] != 0:
	                    cx = int(moment['m10'] / moment['m00'])
	                    cy = int(moment['m01'] / moment['m00'])
	                    centroid += (cx,)
	                    centroid += (cy,)
	                else:
	                    centroid += (0,)
	                    centroid += (0,)
	                cv2.circle(frame,centroid, 5, (0,255,255), -1)
	                #END FINDING CENTROID
	            
	                #STAR FINDING FINGERS
	                count = 0    
	                start, end, far = [], [], []
	                for i in range(defects.shape[0]):
	                    s,e,f,d = defects[i,0]                 
	                    if d > 4000 and d<28000:                            # If normal distance between farthest point(defect) and contour is > 14000 and < 28000, it is the desired defect point.
	                        st = tuple(cnt[s][0])
	                        en = tuple(cnt[e][0])
	                        fa = tuple(cnt[f][0])
	                        m1 = math.degrees(math.atan((fa[1] - st[1]) / (fa[0] - st[0])))
	                        m2 = math.degrees(math.atan((fa[1] - en[1]) / (fa[0] - en[0])))
	                        if(m1 < m2):
	                            an = 180 - (m2 - m1)
	                        else:
	                            an = m1 - m2
	                        if an < 60 :
	                           start.append(st)
	                           end.append(en)
	                           far.append(fa)
	                for i in range(0, len(far)):
	                    cv2.line(frame, start[i], far[i], [0,255,225], 2)
	                    cv2.line(frame,end[i], far[i], [255,255,225], 2)
	                    cv2.circle(frame, far[i], 5, [0,0,255], -1)
	                    count += 1
	                    font = cv2.FONT_HERSHEY_COMPLEX
	                    cv2.putText(frame,str(count+1),(100,100),font,1,(0,0,255),1)
	                #END FINDING FINGERS
	        
	                if(len(far) == 1):
	                    ref = (500, 300)
	                    cv2.circle(frame, ref, 2, (0,0,255), -1)
	                    dist = math.sqrt( math.pow(centroid[0] - ref[0], 2) + math.pow(centroid[1] - ref[1], 2))
	                    if(dist < 30):      
	                        cv2.circle(frame, ref, 30, (0,0,255), 0)  
	                                        
	                        dist_left_top = math.sqrt( math.pow(centroid[0] - end[0][0], 2) + math.pow(centroid[1] - end[0][1], 2))
	                        dist_right_top = math.sqrt( math.pow(centroid[0] - start[0][0], 2) + math.pow(centroid[1] - start[0][1], 2))
	                        dist_bottom = math.sqrt( math.pow(centroid[0] - far[0][0], 2) + math.pow(centroid[1] - far[0][1], 2))
	                        if count_last == 0:                
	                            if dist_bottom > dist_left_top * 0.5 and dist_bottom < dist_left_top * 0.6:
	                                pyautogui.click()
	                                print("left")
	                            elif dist_bottom > dist_right_top * 0.5 and dist_bottom < dist_right_top * 0.6 :
	                                pyautogui.click(button='right')
	                                print("right")
	                        count_last = (count_last + 1) % 2

	                    elif(dist>30 and dist<100):
	                        cv2.circle(frame, ref, 30, (0,0,255), -1)
	                        x, y = 0, 0
	                        if(centroid[0] - ref[0] != 0):
	                            slope = math.degrees(math.atan((centroid[1] - 300) / (centroid[0] - 500)))
	                        if(centroid[1] < ref[1] and centroid[0] > ref[0]):
	                                x, y = 90 + slope, slope 
	                        elif(centroid[1] < ref[1] and centroid[0] < ref[0]):
	                                x, y = -90 + slope, -slope
	                        elif(centroid[1] > ref[1] and centroid[0] < ref[0]):
	                                x, y = -90 - slope, -slope
	                        elif(centroid[1] > ref[1] and centroid[0] > ref[0]):
	                                x, y = 90 - slope, slope
	                        pyautogui.move(x*(dist/200),y*(dist/200))
	                elif(len(far) == 2):
	                    col, row, key = 330, 100, 'A'
	                    for _ in range(4):
	                        for _ in range(6):
	                            cv2.rectangle(frame,(col,row),(col+50,row+50),(0,0,0),2)
	                            cv2.putText(frame,key,(col+15,row+30), font, 1,(0,0,0),2,cv2.LINE_AA)
	                            key = chr(ord(key) + 1)
	                            col += 50
	                        col = 330
	                        row += 50
	                    for _ in range(2):
	                        cv2.rectangle(frame,(col,row),(col+50,row+50),(0,0,0),2)
	                        cv2.putText(frame,key,(col+15,row+30), font, 1,(0,0,0),2,cv2.LINE_AA)
	                        key = chr(ord(key) + 1)
	                        col += 50
	                    cv2.putText(frame,"keyboard",(100,200),font,1,(0,0,255),1)

	                    dist_bottom_1 = math.sqrt( math.pow(centroid[0] - far[1][0], 2) + math.pow(centroid[1] - far[1][1], 2))
	                    dist_bottom_2 = math.sqrt( math.pow(centroid[0] - far[0][0], 2) + math.pow(centroid[1] - far[0][1], 2))

	                    dist_thumb = math.sqrt( math.pow(centroid[0] - end[1][0], 2) + math.pow(centroid[1] - end[1][1], 2))
	                    dist_index = math.sqrt( math.pow(centroid[0] - end[0][0], 2) + math.pow(centroid[1] - end[0][1], 2))
	                    dist_middle = math.sqrt( math.pow(centroid[0] - start[0][0], 2) + math.pow(centroid[1] - start[0][1], 2))
	                    
	                    if dist_bottom_1 > dist_thumb * 0.5 and dist_bottom_1 < dist_thumb * 0.85:
	                        keypress(end[1])
	                    if dist_bottom_2 > dist_index * 0.5 and dist_bottom_2 < dist_index * 0.95 :
	                        keypress(end[0])
	                    if dist_bottom_2 > dist_middle * 0.5 and dist_bottom_2 < dist_middle * 0.95 :
	                        keypress(start[0])
            
            except:
                pass
            cv2.imshow('frame',frame)
            if cv2.waitKey(20) == 27:
                mode = 1
                break
        vc.release()
        cv2.destroyAllWindows()
