This ML model was developed using Convolutional Neural Netwroks for picture detection. It was developed using two types of pictures:
Dog and Cat. The pictures were also zommed in/out and rotatated when they were feeded to the CNN - it allowed for bigger sample and also allowed to better train the model due to the biger variety of pictures.
Pictures were scaled down to be 64 by 64 pixels to not overload the CPU (default tesnsorflow library - CPU). 

Part where pictures were scaled was copied form the Keras documentation with some changes to the code to match the personal requriements.


* Could not import the data set used for this model as they are too big for GitHub, however it could be possible to feed it with any images and train it and then see the results
Libraries used 
-Tensorflow
-Keras 
-Numpy