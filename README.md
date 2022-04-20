# Project Title
RealTime FaceMask Detection

# Project Description
This project contains all necessary code and files to develop the Face Mask Detection using Computer Vision.

# Folders and Files Description
* annotations: This folder will be used to store all *.pbtxt files and the respective TensorFlow *.record files, which contain the list of labels for our dataset images. 
* images: This folder contains a copy of all the images in our dataset, as well as the respective *.xml files produced for each one, once labelImg is used to annotate objects.
* images/train: This folder contains a copy of all images, and the respective *.xml files, which will be used to train our model. 
* images/test: This folder contains a copy of all images, and the respective *.xml files, which will be used to test our model. 
* models: This folder will contain a sub-folder for each of training job. Each subfolder will contain the training pipeline configuration file *.config, as well as all files generated during the training and evaluation of our model
* notebooks: This folder contains the Jupyter notebook for collecting the sample images using OpenCV.
* scripts: In this folder you will find the Python file to load and execute the Real Time Face Mask Detection app (load_model.py) and inside the preprocessing folder, you will find the Python file used to create both train and testing record files.
* model_main_tf2.py: Creates and runs TF2 object detection models. It is the file needed to start training the model with an existing pipeline configuration. 
* pipeline.config: This file contains the setup for training the model, including number of classes, batch size, hyperparameters for training and evaluation metrics.

# How to Install the Project
* Make sure all the necessary files are uploaded in the appropriate path. 
* Install the required libraries
   1. pip install pandas
   2. pip install numpy
   3. pip install tensorflow
   4. pip install labelImg
   5. pip install os
   6. pip install cv2
* Install the required API's 

# How to use the Project
Once all the neccessary modules and libraries are installed correctly we have to train and test the model with certain amount of images. So, the first step will be to add a data set of images. For this step, we recommend to use the collect_imgs.ipynb file which contains the code for collecting and saving sample images.
After collecting sample images, you will need to label the corresponding object which will be used for detection (in this case, the face mask). For this task, we will use the labelImg.py file under scripts directory or you can install it using "pip install labelImg". Next, we will create the corresponding label map to convert the used labels to integer values. This label map is used both by the training and detection processes.

Now, from the images' xml files we will generate the corresponding training and testing record files which is the type of file that Tensorflow needs for training jobs. For this task, we use the next commands: python generate_tfrecord.py -x WORKSPACE_PATH + \training_demo\images\train -l WORKSPACE_PATH + \training_demo\annotations\label_map.pbtxt -o WORKSPACE_PATH + \training_demo\annotations\train.record and python generate_tfrecord.py -x WORKSPACE_PATH + \training_demo\images\test -l WORKSPACE_PATH + \training_demo\annotations\label_map.pbtxt -o WORKSPACE_PATH + \training_demo\annotations\test.record.

Once the training and testing records are ready, we will use the existing pipeline.config for setting up the training job. The command for starting with this job is: python model_main_tf2.py --model_dir=PATH_TO\my_ssd_resnet101_v1_fpn --pipeline_config_path=PATH_TO\pipeline.config. If using a GPU, the training time is normally short depending on the batch size and steps configured. Every 1000 steps a new checkpoint will be generated. This checkpoints have the corresponding "trained" model and you can export and load them in your application.

To run the application, just use the next command: python load_model.py which will load the saved model and start the application. When this file is executed, you should expect to see the webcam open with the bounding box popping out and detecting whether the person is using a face mask or not by indicating 'Mask' or 'No Mask'.

For further reference you can visit: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
    
# AI Fields Used
* Machine Learning
* Computer Vision
* Object Detection

# Technologies 
* Python
* Tensorflow
* OpenCV
* Numpy
* Pandas

# Software
* Jupyter Notebook
* Anaconda 3


