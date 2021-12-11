import cv2
import time
import tensorflow as tf
#import tensorflow_quantum as tfq

#import cirq
#import sympy
import numpy as np
#import seaborn as sns
import collections

# visualization tools
import matplotlib.pyplot as plt
#from cirq.contrib.svg import SVGCircuit

tf.compat.v1.enable_eager_execution()

def create_classical_model():
    # A simple model based off LeNet from https://keras.io/examples/mnist_cnn/
    model_new = tf.keras.Sequential()
    model_new.add(tf.keras.layers.Conv2D(32, [3, 3], activation='relu', input_shape=(28,28,1)))
    model_new.add(tf.keras.layers.Conv2D(64, [3, 3], activation='relu'))
    model_new.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model_new.add(tf.keras.layers.Dropout(0.25))
    model_new.add(tf.keras.layers.Flatten())
    model_new.add(tf.keras.layers.Dense(128, activation='relu'))
    model_new.add(tf.keras.layers.Dropout(0.5))
    model_new.add(tf.keras.layers.Dense(10))
    return model_new


model_new = create_classical_model()

model_new.load_weights('./model_cnn/fix')

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
		#save_image = np.expand_dims(save_image,axis=0)
		save_image = save_image.reshape(28, 28, 1)
		img_array = np.array([save_image], dtype=float)
		#print(img_array.shape)
		
		predictions = model_new.predict(img_array)
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
