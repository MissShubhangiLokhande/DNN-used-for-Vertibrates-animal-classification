import keras
from PIL import Image
import numpy as np
import os
import cv2
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tkinter import *
from PIL import ImageTk, Image
import tkinter
import time

data = []
labels = []

Birds = os.listdir("Birds")
for bird in Birds:
    imag = cv2.imread("Birds/" + bird)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((60, 60))
    data.append(np.array(resized_image))
    labels.append(0)

fishes = os.listdir("fishes")
for fish in fishes:
    imag = cv2.imread("fishes/" + fish)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((60, 60))
    data.append(np.array(resized_image))
    labels.append(1)

Reptlies = os.listdir("Reptlies")
for Reptlie in Reptlies:
    imag = cv2.imread("Reptlies/" + Reptlie)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((60, 60))
    data.append(np.array(resized_image))
    labels.append(2)

Amphibias = os.listdir("Amphibias")
for Amphibia in Amphibias:
    imag = cv2.imread("Amphibias/" + Amphibia)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((60, 60))
    data.append(np.array(resized_image))
    labels.append(3)

Mammals = os.listdir("Mammals")
for Mammal in Mammals:
    imag = cv2.imread("Mammals/" + Mammal)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((60, 60))
    data.append(np.array(resized_image))
    labels.append(4)

no = os.listdir("no")
for nn in no:
    imag = cv2.imread("no/" + nn)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((60, 60))
    data.append(np.array(resized_image))
    labels.append(5)

animals=np.array(data)
labels=np.array(labels)
np.save("animals_ii",animals)
np.save("labels_ii",labels)
animals=np.load("animals_ii.npy")
labels=np.load("labels_ii.npy")
s=np.arange(animals.shape[0])
np.random.shuffle(s)
animals=animals[s]
labels=labels[s]
num_classes=len(np.unique(labels))
data_length=len(animals)
(x_train,x_test)=animals[(int)(0.2*data_length):],animals[:(int)(0.2*data_length)] #20% test
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_length=len(x_train)
test_length=len(x_test)
(y_train,y_test)=labels[(int)(0.2*data_length):],labels[:(int)(0.2*data_length)]
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)
time_train_start=time.time()

model=Sequential() #started keras
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(60,60,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(6,activation="softmax"))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=500 ,epochs=1,verbose=1)
score = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test accuracy:', score[1])
time_train_end=time.time()
time_train=time_train_end-time_train_start



model_yaml = model.to_yaml()
with open("ganeral.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("ganeral.h5")
print("Saved model to disk")





def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((60, 60))
    return np.array(image)
def get_animal_name(label):
    if label==0:
        return "bird"
    if label==1:
        return "fish"
    if label ==2:
        return "Reptile"
    if label==3:
        return "Amphibian"
    if label==4:
        return "Mammal"
    if label==5:
        return "No animal"

def predict_animal(file):
    print("Predicting .................................")
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    animal=get_animal_name(label_index)
    print(animal)
    return ("The predicted Animal is a " + animal )


path = "test/1.jpg"
time_pred_strat=time.time()
res = predict_animal(path)
time_pred_end=time.time()
time_pred=time_pred_end-time_pred_strat
res=res
root = Tk()
img = ImageTk.PhotoImage(Image.open(path))
panel = Label(root, image=img)
panel.pack(side="bottom", fill="both", expand="yes")
tkinter.Label(root, text=res).pack()

root.mainloop()