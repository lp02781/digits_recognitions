import cv2
import time
import numpy as np

def main():
	#camera = cv2.VideoCapture(-1)
	camera = cv2.VideoCapture(0)
	camera.set(3, 640)
	camera.set(4, 480)
  
	while( camera.isOpened() ):
		_, image = camera.read()
		#image = cv2.flip(image,-1)
		
		roi = image[100 : 200, 200 : 400] 
		
		cv2.imshow('roi', roi)
		cv2.imshow('Original', image)
		
		keyValue = cv2.waitKey(10)
        
		if keyValue == ord('q'):
			break
		        
	cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
