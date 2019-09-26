# DNN-used-for-Vertibrates-animal-classification
A Case Study

------------------------------------------------------------------------------------------------
Tools Used
-----------------------------------------------------------------------------------------------
I used Python syntax for this project. As a framework I used Keras, which is a high-level neural network API written in Python. But Keras can’t work by itself, it needs a backend for low-level operations. Thus I installed a dedicated software library — Google’s TensorFlow.
As a development environment I used the PyCharm. I used Matplotlib for visualization.

In this algorithm we proposed new algorithm to detect and classify five categories of animals: (Mammals, Reptiles, Amphibians, Bird, and Fish). The algorithm based on using model based on DCNNs. 
Three convolutional layer has been used which transforms the image volume into an output volume. 
	Input: RGB image from the dataset consists of (12000 different images).
	Output: detect animals in still images, and classify as one of five categories.

-------------------------------------------------------------------------------------------
Training Algorithm
----------------------------------------------------------------------------------------------
- Create Model of Convolutional Neural Network (CNN) which includes the following:

	Reading image with size 50 x 50.
	Convoluting the image by using filters (16, 32, 64) and 3 max-pooling (pool size=2)
	Extracting feature maps, take the rectified feature as input to produce pooled feature map.
	Converting all the result 2-D array from pooled feature map into a single long continuous linear vector by Flattened process.
	Feeding the Flattened matrix from the pooling layer as input to the fully connected layer to classify the image.
	Coding the outputs, the animal is Mammalwhen label =0, the animal is Reptilewhen label= 1, the animal is Amphibianwhen label =2, the animal is Bird when label =3, the animal is Fish when label =4, and NoAnimalwhen label =5.

-----------------------------------------------------------------------------------------------
Testing Algorithm
-----------------------------------------------------------------------------------------------
Once the model has been trained it is possible to carry out model testing.
Progress is visible on the console when the script runs. At the end it will report the final accuracy of the model.
Once the model has been trained it is possible to carry out model testing. During this phase a second set of data is loaded. This data set has never been seen by the model and therefore it’s true accuracy will be verified.
After the model training is complete, and it is understood that the model shows the right result, it can be saved by: model.save(“name_of_file.h5”).
Finally, the saved model can be used in the real world. The name of this phase is model evaluation. This means that the model can be used to evaluate new data.
