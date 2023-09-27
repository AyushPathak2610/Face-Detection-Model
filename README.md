# Face-Detection-Model
# Made using :- Siamese Neural Network, OpenCV and Kivy app for frontend
I have implemeted a personal face detection model based upon siamese neural network.
Firstly, file paths and directories were setup and open cv was linked to collect individual data and it was written in the anchor and positive files.
The negative images were gathered from the labelled faces from the wild dataset.
A function is made to grab 200 of each folder for model training.
Before being goven input, the the images are preprocessed as per the need of the siamese neural netowrk for the input.
The embedded layer is trained with conv2D layers and followed by the hidden layers, for both, anchor images and positive/negative images, based upon the L1 dustance between them, 
after which verification task is done with sigmoid function.
The model is then trained.
Application folder is setup with input image directory and verification image directory.
Then using openCV, verifcation function is implemented with the verification and detection thresholds set at .7 and .7 resp.
The model is also saved as an h5 file for all the gradients and weigths.
This is then implemented for frontend as a Kivy app.
