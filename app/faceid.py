# Import kivy dependencies
# layout dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
# ux dependecies
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
# to get realtime feed
from kivy.clock import Clock

from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2 as cv
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Build app and layout
class CamApp(App):
    
    def build(self):
        # Main Layout Components
        self.webcam = Image(size_hint=(1,.8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1))
        # Three status for verification label: 1:-"Verification Uninitiated":Initial State, 2:Verified, 3:Unverified
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1,.1))
        
        # Add items to Layout 
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.webcam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        
        # Load tensorflow keras model
        self.model = tf.keras.models.load_model('siamese_model.h5', custom_objects={'L1Dist':L1Dist})
        
        # Setup video capture device
        self.capture = cv.VideoCapture(0)
        # to trigger self.update function at every x number of periods
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout
    
    # Run continously to get webcam feed
    def update(self, *args):
        # Read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[200:200+250,200:200+250,:]
        
        # Flip horizontal and convert the image to texture
        # converting a raw cv image array to a texture for rendering
        # then setting our image equal to that texture
        buf = cv.flip(frame,0).tostring()
        img_texture = Texture.create(size=(frame.shape[1],frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.webcam.texture = img_texture
    
    # load img from file and convert into 100*100px
    def preprocess(self,file_path):
        #read file as byte scale
        byte_img = tf.io.read_file(file_path)
        #load it as img
        img = tf.io.decode_image(byte_img,expand_animations = False)
        #resize the img to 100*100*3 as prescribed in the siamese network
        img = tf.image.resize(img,(100,100))
        #scale between 0 and 1
        img = img/255.0
        return img
    
    # Bring over verification function
    def verify(self,*args):
        # specify thresholds
        detection_thresholds = 0.3
        verification_thresholds = 0.1
        
        # capture input_images from our webcam
        SAVE_PATH = os.path.join('application_data','input_image','input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[200:200+250,200:200+250,:]
        cv.imwrite(SAVE_PATH, frame)
        
        # build results array
        results = []
        for image in os.listdir(os.path.join('application_data','verification_images')):
            input_img = self.preprocess(os.path.join('application_data','input_image','input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data','verification_images',image))
            
            # make predictions
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            # expand dims is to add an extra dimension along axis=1(columns) so that its shape is from (2,...) to (2,1,..), basically that extra 1 is necessary while dealing with batch dimensions 
            results.append(result)
            
        # Detection Threshold: Metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_thresholds)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples
        verification = detection / len(os.listdir(os.path.join('application_data','verification_images')))
        verified = verification > verification_thresholds
        
        # set verification text
        self.verification_label.text = 'Verified' if verification==True else 'Unverified'
        
        # log out details
        Logger.info(results)
        Logger.info(np.sum(np.array(results)>0.5))
        Logger.info(np.sum(np.array(results)>0.2))
        Logger.info(np.sum(np.array(results)>0.7))
        Logger.info(np.sum(np.array(results)>0.4))
        
        return results, verified
        
    
if __name__== '__main__':
    CamApp().run()