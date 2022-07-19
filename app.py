'''
!pip install -U scikit-fuzzy
!pip instal scikit-image

'''

import numpy as np
import gradio as gr
import sklearn
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from gradio.mix import Parallel
import matplotlib.image as img
from sklearn.cluster import KMeans
from FuzzyCMeans import fcm
from FuzzyCMeans import keep_center
from FuzzyCMeans import calc_median

#from frcm import FRCM

def kmeans_gradio(x, clustt):   
    img_arr = x
    clustt = int(clustt)
    ##reshape each of the color channels into a single long array of length W*H.
    
    (h,w,c) = img_arr.shape
    img2D = img_arr.reshape(h*w,c)
    
    kmeans_model = KMeans(n_clusters=clustt)
    #kmeans_model = KMeans(n_clusters=n_clustt) # we shall retain only 7 colors
    cluster_labels = kmeans_model.fit_predict(img2D)
    ##cluster centers thus obtained could as well be floating point numbers. 
    ##So let's convert them to integers so that they are valid RGB values.
    rgb_cols = kmeans_model.cluster_centers_.round(0).astype(int)
    #map all points in a particular cluster to the respective cluster centers.
    img_quant = np.reshape(rgb_cols[cluster_labels],(h,w,c))
        
    return img_quant

def fcm_gradio(x, clustt):
    
   clustt = int(clustt) 
   img = img_as_float(x)
   x = np.reshape(img,(img.shape[0]*img.shape[1],3),order='F')
   
   #Applying FCM to pixels
   cluster_n = clustt
   expo = 20
   min_err = 0.001 
   max_iter = 500 
   verbose = 0
   m,c = fcm(x,cluster_n,expo,min_err,max_iter,verbose)
   m = np.reshape(m,(img.shape[0],img.shape[1]),order='F')
   
   #Replace pixel intensity with centers found by FCM or replace pixel intensity with median for each cluster
   #simg = keep_center(img,m,c,verbose)
   #Optinally replace each pixel intensity with the median intensity of the pixels belonging to same cluster.    
   simg = calc_median(img,m,verbose)
   return simg[:,:,::-1]





image_input = gr.Image()
image_output = gr.Image()
 
km = gr.Interface(kmeans_gradio, inputs=[image_input, "number"], 
                  outputs=image_output, description="Kmeans clustering")
fc = gr.Interface(fcm_gradio, inputs=[image_input,"number"], 
                  outputs=image_output, description="FCM clustering")


Parallel(km,
        fc,
        title = "Comparing clustering algorithms", 
        inputs=[image_input,"number"], 
        outputs=[gr.Image(),gr.Image()],
        examples = [["Sea_mountain.jpg", "5"],
                    ["Sky_mountain.jpg", "5"]],
        description="Choose number of clusters and upload image for K-Means or FCM Clustering").launch()
    
   
      
