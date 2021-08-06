# a simple GUI app for face mask detection on images and in real-time using tkinter

import tkinter as tk
from tkinter import *
from tkinter import messagebox,filedialog
from PIL import ImageTk, Image
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time

from tkinter import ttk
import numpy as np


class Prediction():
    def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	    (h, w) = frame.shape[:2]
	    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	    faceNet.setInput(blob)
	    detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	    faces = []
	    locs = []
	    preds = []

	# loop over the detections
	    for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		    confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		    if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			    (startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			    (startX, startY) = (max(0, startX), max(0, startY))
			    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			    face = frame[startY:endY, startX:endX]
			    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			    face = cv2.resize(face, (224, 224))
			    face = img_to_array(face)
			    face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			    faces.append(face)
			    locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	    if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		    faces = np.array(faces, dtype="float32")
		    preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	    return (locs, preds)

    def load_model():
        face_path="./Models/face_detector"
        model_path="./Models/mask_detector.model"
        prototxtPath = os.path.sep.join([face_path, "deploy.prototxt"])
        weightsPath = os.path.sep.join([face_path,"res10_300x300_ssd_iter_140000.caffemodel"])
        Prediction.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        Prediction.maskNet = load_model(model_path)
    def video_stream():
        while True:

        
            Prediction.cap = cv2.VideoCapture(0)
            if (Prediction.cap.isOpened() == False):
                print("Unable to read camera feed")
                messagebox.showerror("Camera Error", "Unable to start camera! Check for permissions.")
                #MaskApp.btnStateReset()
                break
            

            else:
                MaskApp.btn2["state"]=DISABLED
                MaskApp.btn3["state"]=DISABLED
                MaskApp.button_id=2
                Prediction.mode="RT"
                Prediction.detection()
                break
        

    def resize(img):
        std_shape=(552,875)
        scale_w=std_shape[1]/img.shape[1]
        scale_h=std_shape[0]/img.shape[0]
        if (scale_h<scale_w):
            img_new=cv2.resize(img,(int(img.shape[1]*scale_h),int(img.shape[0]*scale_h)))
        else:
            img_new=cv2.resize(img,(int(img.shape[1]*scale_w),int(img.shape[0]*scale_w)))   

        return img_new

    def detection():
        if Prediction.mode=='RT':
            _, frame = Prediction.cap.read()
            #print(type(frame))
            frame=cv2.resize(frame,(875,552))
        else:
            frame=MaskApp.img

            #print(type(frame))
        
        (locs, preds) = Prediction.detect_and_predict_mask(frame, Prediction.faceNet, Prediction.maskNet)

	# loop over the detected face locations and their corresponding
	# locations
        for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
	        (startX, startY, endX, endY) = box
	        (mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
	        label = "Mask" if mask > withoutMask else "No Mask"
	        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
			
		# include the probability in the label
	        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
	        cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
	        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
	    
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        if Prediction.mode != 'RT':
            cv2image=Prediction.resize(cv2image)
    
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        MaskApp.Lable_frame.imgtk = imgtk
        MaskApp.Lable_frame.configure(image=imgtk)
        if Prediction.mode=='RT':
            MaskApp.Lable_frame.after(1, Prediction.detection)
        

class MaskApp():
    button_id=0
    Prediction.load_model()
    def chg_image(self):
       
        MaskApp.img = self.im
        Prediction.detection()
    def upload_photo(self):
        self.btn3["state"]=DISABLED
        button_id=1   ### 1 for upload photo, 2 for video_stream
        Prediction.mode="Photo"
        filename = filedialog.askopenfilename()
        if filename != "":
            self.im = cv2.imread(filename)
        self.chg_image()


    def btnStateReset(self):
        self.btn2["state"]=NORMAL
        self.btn3["state"]=NORMAL
    
    def home(self):
        
        if MaskApp.button_id==2:
            Prediction.cap.release()
        
        self.btnStateReset()
        self.Lable_frame.configure(image=self.bg)
    
    def refresh(self):
        if self.button_id == 1:
            self.upload_photo
        elif self.button_id==2:
            Prediction.video_stream
            #print("Working")
        else:
            self.button_id=0    


    def about(self):
        top=Toplevel()
        top.resizable(False,False)
        top.title('Info...')
        top.iconbitmap('./logo/logo_mask.ico')
        my_label=Label(top,text="This is a ML trained system to detect masked vs unmasked persons.\n\nCo-developed by Kunwar Yatesh Kumar, Shubham Pratap, Saubhagya Shukla.\n\nStay home, Stay Home",)
        my_label.pack(side=TOP)
        Closebtn=Button(top,text='Close',command=top.destroy,relief=RAISED)
        Closebtn.pack(side=BOTTOM)


    
    def __init__(self):
        
        self.root=tk.Tk()
        self.root.title("Face Mask Detection")
        self.root.iconbitmap('./logo/logo_mask.ico')
        self.root.geometry('990x605')
        self.root.resizable(False,False)

        #left upper part with photo, realtime, refresh, Quitand left lower with about button
        self.left=tk.Frame(self.root,width=150,bg='#2391b8')
        self.left_upper=tk.Frame(self.left,width=150,bg='#5ac3dc')
        self.left_lower=tk.Frame(self.left,width=150,bg='#2391b8')
        self.left_upper.pack(side='top',fill='both',expand=True)
        self.left_lower.pack(side='top',fill='both',expand=True)

        #buttons
        b1img=tk.PhotoImage(file='./icon/home.png')
        btn1=tk.Button(self.left_upper, text='Home',image=b1img,command= self.home,bg='#5ac3dc',relief=FLAT)
        btn1.grid(column=0,row=0,padx=15,pady=10)
        photo=tk.PhotoImage(file='./icon/gallery.png')
        MaskApp.btn2=tk.Button(self.left_upper, text='Photo',image=photo, command=self.upload_photo,bg='#5ac3dc',relief=FLAT,)
        MaskApp.btn2.grid(column=0,row=1,padx=15,pady=5)
        cam=tk.PhotoImage(file='./icon/cam.png')
        MaskApp.btn3=tk.Button(self.left_upper, text='Realtime',image=cam,command=Prediction.video_stream,bg='#5ac3dc', relief= FLAT)
        MaskApp.btn3.grid(column=0,row=2,padx=15,pady=5)
        refresh=tk.PhotoImage(file='./icon/refresh.png')
        btn4=tk.Button(self.left_upper, text='Refresh',image=refresh,command=self.refresh, bg='#5ac3dc',relief= FLAT)
        btn4.grid(column=0,row=3,padx=15,pady=5)
        quit_=tk.PhotoImage(file='./icon/exit.png')
        btn5=tk.Button(self.left_upper, text='Quit',image=quit_,command= self.root.destroy, bg='#5ac3dc',relief= FLAT)
        btn5.grid(column=0,row=4,padx=15,pady=5)
        info=tk.PhotoImage(file='./icon/info.png')
        btn6=tk.Button(self.left_lower, text='About',image=info, bg='#2391b8',command=self.about ,relief= FLAT,compound=LEFT)
        btn6.pack(side='bottom')

        #right side

        self.title_Frame=tk.Frame(self.root,bg="#5ac3dc")

        self.title=tk.Label(self.title_Frame,text='MASK DETECTION',bg="#5ac3dc")
        self.title.pack()
        
        self.bg=PhotoImage(file='./bg/mask_bg.png')
        
        MaskApp.Lable_frame=tk.Label(self.root,bg='gray80')
        
        self.Lable_frame.imgtk = self.bg
        self.Lable_frame.configure(image=self.bg)
        

        #status bar
        self.status_frame=tk.Frame(self.root,bg='#2391b8')
        self.status=tk.Label(self.status_frame,text='STAY HOME STAY SAFE. ALWAYS WEAR MASK!!',bg='#2391b8')
        self.status.pack(fill='both',expand=True)


        #packing with root
        self.left.grid(row=0,column=0,rowspan=2,sticky='nsew')
        self.title_Frame.grid(row=0,column=1,sticky='nsew')
        self.Lable_frame.grid(row=1,column=1,sticky='nsew')
        self.status_frame.grid(row=2,column=0,columnspan=2,sticky='nsew')

        #configure
        self.root.grid_rowconfigure(1,weight=1)
        self.root.grid_columnconfigure(1,weight=1)


        self.root.mainloop()

if __name__== '__main__':
    MaskApp() 