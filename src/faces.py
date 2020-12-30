import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
recog = cv2.face.LBPHFaceRecognizer_create()
recog.read("trainner.yml")

# save the labels through pickle
labels = {"person_name": 1}
with open("labels.pkl", 'rb') as f :
	og_lab = pickle.load(f)
	labels = {v:k for k,v in og_lab.items()}

capture = cv2.VideoCapture(0)

while(True):
	# Frame-by-frame capturing
	ret, frame = capture.read()
	# to convert it in the gray image
	gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# to detect the face in the images
	faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.5, minNeighbors=5)

	# face detector
	for (x,y,w,h) in faces:
		#print(x,y,w,h)
		# region of interest
		roi_gray = gray_img[y:y+h, x:x+w] #gray image [ycord_start, ycord_end] and same for xcord too
		roi_color = frame[y:y+h, x:x+w] #color image

		#recognizer
		id_, confidence = recog.predict(roi_gray)

		#confidence level 
		if confidence>=45: # and confidence<=85:
			print(id_)
			print(labels[id_])
			# OpenCV PutText in the rectangle
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255) # white color
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		# saving it
		img_item = "11.png"
		cv2.imwrite(img_item, roi_color)

		# draw a rectangle
		rec_color = (255, 0, 0) # BGR 0-255
		rec_stroke = 2  # how thick to we want the line of the rectangle
		width = x + w #end coordinate of x
		height = y + h  #end coordinate of y
		cv2.rectangle(frame, (x,y), (width, height), rec_color, rec_stroke)

	# resultant frame
	cv2.imshow('Frame',frame)

	# to stop/wait
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

# releasing the capture
capture.release()
# destroying the windows for cv
cv2.destroyAllWindows()