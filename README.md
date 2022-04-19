# Project Title
RealTime FaceMask Detection

# Project Description
This contains code to develop the Face Mask Detection Using Computer Vision.

# Folders Description
* annotations: This folder will be used to store all *.csv files and the respective TensorFlow *.record files, which contain the list of annotations for our dataset images. 
* images: This folder contains a copy of all the images in our dataset, as well as the respective *.xml files produced for each one, once labelImg is used to annotate objects.
* images/train: This folder contains a copy of all images, and the respective *.xml files, which will be used to train our model. 
* images/test: This folder contains a copy of all images, and the respective *.xml files, which will be used to test our model. 
* models: This folder will contain a sub-folder for each of training job. Each subfolder will contain the training pipeline configuration file *.config, as well as all files generated during the training and evaluation of our model
* notebooks: This folder contains the jupyter notebook for collecting the sample images. 

# How to Install the Project
* Make sure all the necessary files are uploaded in the appropriate path. 
* Install the required libraries
   1.!Pip Install Pandas
   2.!Pip Install Numpy
   3.!Pip Install TensorFlow
   4.!Pip Install Collections
   5.!Pip Install os
   6.!Pip Install cv2
* Install the required API's 

# How to use the Project
Once all the neccessary modules and libraries are installed correctly we have to train and test the model with certain amount of images. 
Once this step is done, the final step is that we have to run the corresponding python file in the Jupyter notebook or any other python file running environment. 
When this file is finished running it will automatically open the webcam with the square box popping out. This helps to identify the person with and without mask by indication 
'No Mask' and 'Mask'.
    
# Methods Used
* Machine Learning
* Tensor Flow
* Computer Vision

# Technologies 
* Python
* Pandas
* Jupyter
* Numpy
* Object detection


