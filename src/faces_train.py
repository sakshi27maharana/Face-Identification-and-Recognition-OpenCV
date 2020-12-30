import os
import cv2
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

#recognizer LBPH
recog = cv2.face.LBPHFaceRecognizer_create()

curr_id = 0
label_id = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			#label of the dir
			#label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
			label = os.path.basename(root).replace(" ","-").lower()
			print(label, path)

			if not label in label_id:
				label_id[label] = curr_id
				curr_id += 1
				
			id_ = label_id[label] 
			print(label_id)

			#y_labels.append(label) # some number
			#x_train.append(path) # verify, turn into numpy array, gray
			pil_img = Image.open(path).convert("L") #grayscale
			
			# resize the image
			size = (550, 550)

			# final image
			final_img = pil_img.resize(size, Image.ANTIALIAS)

			img_arr = np.array(final_img, "uint8")
			print(img_arr)
			faces = face_cascade.detectMultiScale(img_arr, scaleFactor=1.5, minNeighbors=5)

			for (x,y,w,h) in faces:
				print(x,y,w,h)
				# region of interest
				roi = img_arr[y:y+h, x:x+w] 
				x_train.append(roi)
				y_labels.append(id_)

#print(y_labels)
#print(x_train)

# save the labels through pickle
with open("labels.pkl", 'wb') as f :
	pickle.dump(label_id, f)

recog.train(x_train, np.array(y_labels))
recog.save("trainner.yml")