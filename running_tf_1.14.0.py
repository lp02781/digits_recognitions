import cv2
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.compat.v1.enable_eager_execution()

model_new = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

model_new.load_weights('./model/hehe')

probability_model = tf.keras.Sequential([
  model_new,
  tf.keras.layers.Softmax()
])

def main():
	#camera = cv2.VideoCapture(-1)
	camera = cv2.VideoCapture(0)
	camera.set(3, 640)
	camera.set(4, 480)
  
	while( camera.isOpened() ):
		_, image = camera.read()
		#image = cv2.flip(image,-1)
		
		save_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		save_image = cv2.resize(save_image, (28,28))
		img_array = np.array([save_image], dtype=float)
		
		predictions = probability_model.predict(img_array)
		predictions = np.array([predictions], dtype=float)
		#print(predictions)
		
		predicted_label = np.argmax(predictions)
		#print(predicted_label)
		
		font = cv2.FONT_HERSHEY_SIMPLEX
		org = (50, 300)
		fontScale = 10
		color = (255, 0, 0)
		thickness = 15
		
		image = cv2.putText(image, str(predicted_label), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
		
		cv2.imshow('Save', save_image)
		cv2.imshow('Original', image)
		
		keyValue = cv2.waitKey(10)
        
		if keyValue == ord('q'):
			break
		elif keyValue == ord('p'):
			print("i got:", predicted_label)
			cv2.imwrite("tf_1_14_out.jpg", image)
		        
	cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
