import tensorflow.contrib.image
import cv2
import numpy as np
import tensorflow as tf
from embeddings_pre import load_model
import embeddings_pre
import requests
import os
import time

def upload(i):
    path = '../new_pictures/'
            
    files1 = os.listdir(path)
    ntime = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())

    for step in files1:
        if step == "1.txt":
            continue
        split = os.path.splitext(step)
        pic_name = split[0]
        
        scaled_arr=[]
        img = cv2.imread(path + step)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scaled =cv2.resize(gray,(160,160),interpolation=cv2.INTER_LINEAR)
        scaled.astype(float)
        scaled = np.array(scaled).reshape(160, 160, 1)
        scaled = embeddings_pre.prewhiten(scaled)
        scaled_arr.append(scaled)
        
        feed_dict = { images_placeholder: scaled_arr, phase_train_placeholder:False ,keep_probability_placeholder:1.0}
        #---------------------------------上傳資料庫---------------------------------
        x = sess.run(embeddings, feed_dict=feed_dict)
        Data = {
            "user_name" : pic_name + str(i),
            "embedding" : str(x),
            "date"      : str(ntime)
        }
        conn = requests.post("http://140.136.150.100/upload.php", data = Data)
        #----------------------------------------------------------------------------
        print(conn.text)
        #----------------------------------------------
    
with tf.Session() as sess:
    load_model('../model/')
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("Mul:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    keep_probability_placeholder= tf.get_default_graph().get_tensor_by_name('keep_probability:0')
    
    for i in range (1,221):
        upload(i)
        time.sleep(1)