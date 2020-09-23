# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:00:16 2019

@author: UX501
"""
import tensorflow.contrib.image
import tensorflow as tf
import numpy as np
import sys
import os
import cv2
import h5py
import matplotlib.pyplot as plt
import requests
import time

# In[2]:


def main():
    cv2_face()
#    path='../pictures/embedding.h5'
#    if os.path.exists(path):
#        os.remove(path)
#        print('embedding have been removed！！！')

#    embs,class_arr=cv2_face()
   
#    f=h5py.File('../pictures/embedding.h5','w')
#    class_arr=[i.encode() for i in class_arr]
#    embs = [j for j in embs]
#    f.create_dataset('class_name',data=class_arr)
#    f.create_dataset('embeddings',data=embs)
#    f.close()



# In[3]:resize face

def cv2_face():
   
    path = '../new_pictures/'
    
    class_names_arr=[]
    files1 = os.listdir(path)
    #print(files1)
    embs = []

    ntime = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())

    with tf.Session() as sess:
        load_model('../model/')
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("Mul:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        keep_probability_placeholder= tf.get_default_graph().get_tensor_by_name('keep_probability:0')
        
        for step in files1:
            if step == "1.txt":
                continue
            #print(step)
            split = os.path.splitext(step)
            pic_name = split[0]
            #print("pic_name = ",pic_name)
            
            scaled_arr=[]
            img = cv2.imread(path + step)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            scaled =cv2.resize(gray,(160,160),interpolation=cv2.INTER_LINEAR)
            scaled.astype(float)
            scaled = np.array(scaled).reshape(160, 160, 1)
            scaled = prewhiten(scaled)
            #check shape
            #print(scaled.shape)
            scaled_arr.append(scaled)
            class_names_arr.append(step)
            
            feed_dict = { images_placeholder: scaled_arr, phase_train_placeholder:False ,keep_probability_placeholder:1.0}
            #---------------------------------上傳資料庫---------------------------------
            x = sess.run(embeddings, feed_dict=feed_dict)
            Data = {
                "user_name" : pic_name,
                "embedding" : str(x),
                "date"      : str(ntime)
            }
            conn = requests.post("http://140.136.150.100/upload.php", data = Data)
            #----------------------------------------------------------------------------
            # calculate embeddings
            #embs.append(sess.run(embeddings, feed_dict=feed_dict))
            print(conn.text)
            
    #return embs,class_names_arr
# In[]
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std,1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x,mean),1/std_adj)
    return y
# In[4]:


def load_model(model_dir,input_map=None):
    '''reload model'''
    
    ckpt = tf.train.get_checkpoint_state(model_dir)                         
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')   
    saver.restore(tf.get_default_session(), ckpt.model_checkpoint_path)


# In[ ]:


if __name__=='__main__':
    main()
