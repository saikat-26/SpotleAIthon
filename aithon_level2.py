import sys
import os  
import pandas as pd  
import numpy as np    
from keras.models import Sequential  
from keras.layers import Dense, Dropout, Activation, Flatten  
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D  
from keras.losses import categorical_crossentropy  
from keras.optimizers import Adam  
from keras.regularizers import l2  
from keras.utils import np_utils 
data=pd.read_csv('/aithon2020_level2_traning.csv') 
def aithon_level2_api(data, test_data)
	train1,train2=[],[]  
	test1,test2=[],[]
	train2=np.array(data['emotion'])
	data=data.drop(columns='emotion')
	train1=np.array(data.values.tolist(),'float32')
	features = 64  
	labels = 3  
	batchsize = 64  
	epochs = 30  
	width, height = 48, 48 
	train2=np_utils.to_categorical(train2, num_classes=labels)
	train1 -= np.mean(train1, axis=0)  
	train2 /= np.std(train2, axis=0) 
	train1 = train1.reshape(train1.shape[0], 48, 48, 1)
	model = Sequential()  
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(train1.shape[1:])))
	model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))  
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))  
	model.add(Dropout(0.5))
	model.add(Conv2D(64, (3, 3), activation='relu'))  
	model.add(Conv2D(64, (3, 3), activation='relu'))  
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))  
	model.add(Dropout(0.5)) 
	model.add(Conv2D(128, (3, 3), activation='relu'))  
	model.add(Conv2D(128, (3, 3), activation='relu'))  
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))    
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))  
	model.add(Dropout(0.2))  
	model.add(Dense(1024, activation='relu'))  
	model.add(Dropout(0.2))  
	model.add(Dense(num_labels, activation='softmax'))  
	model.compile(loss=categorical_crossentropy,  
              optimizer=Adam(),  
              metrics=['accuracy'])  
	model.fit(train1, train2,  
          batch_size=batchsize,  
          epochs=epochs,  
          verbose=1,  
          validation_data=(test1, test2),  
          shuffle=True)  