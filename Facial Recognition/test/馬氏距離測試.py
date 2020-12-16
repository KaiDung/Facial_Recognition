import tensorflow.contrib.image
import tensorflow as tf
from embeddings_pre import load_model
import embeddings_pre
import requests
import numpy as np

def reload():
        global class_arr,emb_arr
        r = requests.get("http://140.136.150.100/download.php")
        
        hold =""
        stop=0
        arr1 = []
        arr2 = []
        for char in r.text:
            hold += char
            if char == '{' or char == '}':
                stop+=1
            if stop == 2:
                data = eval(hold)
                arr1.append(data["user_name"])
                temp = []
                for e in data['embedding'][2:-2].split():
                    temp.append(float(e))
                arr2.append(temp)
                stop = 0
                hold = ""
                
        class_arr = np.array(arr1)
        emb_arr = np.array(arr2)
        
        X = np.reshape(emb_arr[0],(256,1))
        for xm in emb_arr:
            temp = np.reshape(xm,(256,1))
            if (temp == X).all():
                continue
            X = np.hstack((X,temp))
        print(X.shape)
        Var = np.cov(X)
        print(Var.shape)
        #共變異矩陣是對稱矩陣
        if(Var == Var.T).all():
            print("YES")
        #共變異矩陣的反矩陣
        Var_inv =np.linalg.inv(Var)
        
        return X,Var,Var_inv


if __name__ == "__main__":
    import sys
    with tf.Session() as sess:
        load_model('../model/')
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("Mul:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        keep_probability_placeholder= tf.get_default_graph().get_tensor_by_name('keep_probability:0')
        
        X,Var,Var_inv = reload()
            