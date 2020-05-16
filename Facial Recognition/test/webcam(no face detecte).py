# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 14:53:43 2019

@author: UX501
"""
# In[0]: import library & function
import tensorflow.contrib.image
from PyQt5 import QtWidgets, uic, QtGui,QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
import tensorflow as tf
from embeddings_pre import load_model
import embeddings_pre
import h5py
import os
import time
import sys
import requests
import Drive_API
from Drive_API import Drive_upload
#import matplotlib.pyplot as plt
# In[1]:Load target features
#open camera
THRED=0.85
left_w = 200
left_h = 160
face_scale = 228
in_or_out = 0
auto_detect_check = 0
#include target face
'''
if os.path.exists('../pictures/embedding.h5'):
    f=h5py.File('../pictures/embedding.h5','r')
    class_arr=f['class_name'][:]
    #print("class = ",class_arr)
    class_arr=[k.decode() for k in class_arr]
    emb_arr=f['embeddings'][:]
else:
    class_arr = []
    emb_arr = []
#print(emb_arr.type)
'''
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
lower_blue = np.array([100,43,46])
upper_blue = np.array([124,255,255])
# In[2]:Load Qt UI & setting function
class GUI_window(QtWidgets.QMainWindow):
    def __init__(self,parent=None):
        
        #載入資料庫的資料
        self.reload()
        
        QtWidgets.QMainWindow.__init__(self)
        #self.setWindowTitle('ggez')
        self.ui = uic.loadUi("webcam.ui",self)
        #self.ui.setFixedSize(self.size())
        self.ui.tabWidget.setTabText(0,"Main")
        self.ui.tabWidget.setTabText(1,"Search")
        self.ui.tabWidget.setTabText(2,"Setting")
        
        self.ui.Open_Button.clicked.connect(self.open_detect)
        self.ui.pushButton.clicked.connect(self.save_image)
        self.ui.Stop_Button.clicked.connect(self.stop)
        self.ui.lineEdit.returnPressed.connect(self.save_image)
        
        self.ui.Send_Button.clicked.connect(self.search_event)
        self.ui.lineEdit_2.returnPressed.connect(self.search_event)
        #由h5抓特徵資料
        #self.ui.reload_image.clicked.connect(self.reload_paremeter)
        #由資料庫抓取資料
        self.ui.reload_image.clicked.connect(self.reload)
        
        self.ui.auto_detect.clicked.connect(self.auto_detect_event)
        self.auto_detect.setStyleSheet("QPushButton{background:black;border-radius:12;}"
            )
        self.widget.setStyleSheet("QWidget{border-width:2px;}"
                                  "QWidget{border-color:black;}"
                                  "QWidget{border-style:outset;}"
                                  "QWidget{height:100;}"
                                  "QWidget{border-radius:5px;}"
                                  "QWidget{background-color:qlineargradient(x1 : 0, y1 : 0, x2 : 0, y2 : 1, stop :  0.0 #f5f9ff,stop :   0.5 #c7dfff,stop :   0.55 #afd2ff,stop :   1.0 #c0dbff);}")
        #self.label3 = QLabel(self)
        #self.label3.setGeometry(1, 49, 281, 54)
        self.ui.close_button.clicked.connect(self.close_camera)
        self.ui.tabWidget.setTabIcon(0,QtGui.QIcon("home.png"))
        self.ui.tabWidget.setIconSize(QtCore.QSize(30,30))
        self.ui.tabWidget.setTabIcon(1,QtGui.QIcon("search.png"))
        self.ui.tabWidget.setIconSize(QtCore.QSize(30,30))
        self.ui.tabWidget.setTabIcon(2,QtGui.QIcon("setting.png"))
        self.ui.tabWidget.setIconSize(QtCore.QSize(30,30))
        self.setWindowIcon(QtGui.QIcon("window_icon.png"))
        #QTabWidget>QWidget>QWidget{background: gray;}
        tab_shape = QtWidgets.QTabWidget.Triangular
        self.tabWidget.setTabShape(tab_shape)
        self.check = 0
        self.i = 0
        auto_detect_check = 0
        self.label2 = QLabel(self)
        #self.label2.setGeometry(self.label.width()+20,150,45,45)
        #self.label2.setStyleSheet("QLabel{background:black;}"
                   #"QLabel{color:rgb(300,300,300,120);font-size:30px;font-weight:bold;font-family:宋体;}"
                   #"QLabel{border-radius: 22;}"
                   #)
        self.label1 = QLabel(self)
        #self.label1.move(200, 60)
        #$self.label1.AlignCenter()
        #self.label1.setAlignment(QtCore.Qt.AlignCenter)
        self.label1.setStyleSheet("QLabel{background:0;}"
                   "QLabel{color:rgb(300,300,300,120);font-size:30px;font-weight:bold;font-family:宋体;}"
                   )
    # 动态显示时间在label上
        timer = QTimer(self)
        timer.timeout.connect(self.showtime)
        timer.start()
        self.label1.setAlignment(QtCore.Qt.AlignCenter)
        self.label2.setGeometry(self.label.width()+20,150,45,45)
        #self.label1.resize(self.width(),50)
        self.show()
    def search_event(self):
        search_object = self.ui.lineEdit_2.text()
        print(search_object)
        Data = {
            "user_name" : search_object
            }
        conn = requests.post('http://140.136.150.100/search.php', data = Data)
        print(conn.text)
        var = self.ui.lineEdit_2.setText('')
        
        slm=QStringListModel()
        self.qList = []
        #self.listView.setStyleSheet("Text{horizontalAlignment: Text.AlignCenter;}")
        #self.qList=['Item 1','Item 2','Item 3','Item 4']
        temp = ""
        self.ui.listWidget.clear()
        for char in conn.text:
            if char != '}' and char != '{':
                temp = temp + char
            else:
                temp = temp.replace("\\","")
                temp = temp.replace('"',"")
                temp = temp.replace(",","   ")
                temp = temp.replace("e:","e: ")
                self.ui.listWidget.addItem(temp)
                temp = ''
        #self.qList.append('jim99224')
        #slm.setStringList(self.qList)
        #self.ui.listWidget.setModel(slm)
        # Data = {
                #"user_name" : search_object
            #}
    def auto_detect_event(self):
        global auto_detect_check
        if auto_detect_check == 0:
            self.widget.raise_()
            self.auto_detect.raise_()
            self.anim = QPropertyAnimation(self.auto_detect, b"geometry")
            self.anim.setDuration(400)
            self.anim.setStartValue(QRect(11,11,25,25))
            self.anim.setEndValue(QRect(245,11,25,25))
            self.anim.start()
            auto_detect_check = 1
        else:
            self.anim = QPropertyAnimation(self.auto_detect, b"geometry")
            self.anim.setDuration(400)
            self.anim.setStartValue(QRect(245,11,25,25))
            self.anim.setEndValue(QRect(11,11,25,25))
            self.anim.start()
            auto_detect_check = 0
    def showtime(self):
        ntime = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
        #datetime = QDateTime.currentDateTime()
        #text = datetime.toString(Qt.ISODate)
        self.label1.setGeometry(0, 50, self.width(),50)
        self.label2.setGeometry(self.label.width()+20,150,45,45)
        self.label1.setText(ntime)
    
    def keyPressEvent(self,event):
        global f
        print('keypress')
        print(event.key())
        if event.key() == 16777216:# 16777216 是 Esc按鍵的事件編碼
            self.showMinimized()
        else:
            event.ignore()
        if event.key() == 81:# 81 是 q按鍵的事件編碼
            print("Esc pressed")
            self.close()
    def save_image(self):
        global emb_arr,class_arr
        var = self.ui.lineEdit.text()
        if self.check == 0:
            self.cap = cv2.VideoCapture(0)
            #self.cap = cv2.VideoCapture(1+cv2.CAP_DSHOW)
            self.timer_camera = QtCore.QTimer()
            self.timer_camera.start(150)
            self.timer_camera.timeout.connect(self.show_image)
            self.check = 1
        ret,frame = self.cap.read()
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (5,5), 0, 0)
        faces = faceCascade.detectMultiScale(frame, 1.1, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            #roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
        #cv2.rectangle(frame,(left_w,left_h),(left_w+face_scale,left_h+face_scale),(0,255,0),2)
        face = frame[left_h:left_h+face_scale,left_w:left_w+face_scale]   
        for (x,y,w,h) in faces:
            if x+y+w+h!=0:
                face = frame[y:y+h,x:x+w]  
        #cv2.rectangle(frame,(left_w,left_h),(left_w+face_scale,left_h+face_scale),(0,255,0),2)
        #face = frame[left_h:left_h+face_scale,left_w:left_w+face_scale]            
        if var!='':
            face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_CUBIC )
            #save_image = QtWidgets.QFileDialog.getExistingDirectory(self,"choose direction","../pictures")
            #print(save_image)
            
            #c_path=os.path.join(save_image,var+'.jpg')
            #cv2.imwrite(c_path,face)
            pic_path = "../new_pictures/"+var+".jpg"
            
            cv2.imwrite(pic_path,face)
            #cv2.waitKey(100)
            
            #---------------照片上傳雲端---------------------
            Drive_upload(pic_path,var)
            
            #-----------------------------------------------
            frame = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            self.ui.label_2.setPixmap(QtGui.QPixmap.fromImage(showImage))
            #f.close()
            class_arr = []
            emb_arr = []
            
            #-----------重新執行特徵分析並上傳資料庫----------
            embeddings_pre.main() 
            #----------------------------------------------
            self.reload()
            
            #f=h5py.File('../pictures/embedding.h5','r')
            #class_arr=f['class_name'][:]
            #class_arr=[k.decode() for k in class_arr]
            #emb_arr=f['embeddings'][:]
            
            #用完就刪除照片
            if os.path.exists(pic_path):
                os.remove(pic_path)
            
            print('-----Save完成-----')
            var = self.ui.lineEdit.setText('')
    '''
    def reload_paremeter(self):
        global f,class_arr,emb_arr
        f.close()
        class_arr = []
        emb_arr = []
        embeddings_pre.main()
        f=h5py.File('../pictures/embedding.h5','r')
        class_arr=f['class_name'][:]
        class_arr=[k.decode() for k in class_arr]
        emb_arr=f['embeddings'][:]
        print('built complete')
    '''
    
    
    def reload(self):
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
                arr1.append(data['user_name'])
                
                flag =0
                hold2=""
                arr4=[]
                for ch in data['embedding']:
            
                    if ch =='[' or ch==']' or ch==' ':
                        temp = ch
                    else:
                        hold2+=ch
                      
                    try:
                        if ch==' ' or ch==']':
                            if flag ==0 :
                                arr4.append(np.float32(hold2)) 
                                flag=1
                                hold2 = ""
                    except:
                        continue
                    if ch!=' ' and flag ==1:
                        flag=0
                        
                arr2.append(arr4)
                stop=0
                hold=""
        class_arr = np.array(arr1)
        emb_arr = np.array(arr2)
        print("-----Reload完成----")
        
    def show_image(self):
        global emb_arr,class_arr
        global in_or_out
        #偵測是否勾選自動偵測
        #auto_detect_check = self.ui.auto_detect_check.isChecked()
        #print(auto_detect_check)
        #if auto_detect_check == 1:
            #print("shit")
        rat,frame = self.cap.read()
        
        if rat == True:
            frame = cv2.flip(frame,2)
            t1=cv2.getTickCount()
            img,scaled_arr,facexy = cv2_face(frame,auto_detect_check)
            if scaled_arr is not None:
            
                feed_dict = { images_placeholder: scaled_arr, phase_train_placeholder:False ,keep_probability_placeholder:1.0}
                embs = sess.run(embeddings, feed_dict=feed_dict) 
                face_class=['Others']
                diff = []
                
                for emb in emb_arr:
                    diff.append(np.mean(np.square(embs[0] - emb)))
                min_diff=min(diff)
                if min_diff<THRED:
                    index=np.argmin(diff)
                    face_class[0]=class_arr[index]
                t2=cv2.getTickCount()
                t=(t2-t1)/cv2.getTickFrequency()
                
                if auto_detect_check==True:
                    for (x,y,w,h) in facexy: 
                        cv2.putText(img, '{}'.format(face_class[0]), 
                                (x,y), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,(0, 0, 255), 2)
                        break
                else: 
                    cv2.putText(img, '{}'.format(face_class[0]), 
                            (left_w,left_h), 
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,(0, 0, 255), 2)
                    
                
                ntime = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
                
                if in_or_out == 1 and face_class[0]=='Others':
                    in_or_out = 0
                
                if in_or_out == 0 and face_class[0]!='Others':
                    record_data = {
                          "user_name" : face_class[0],
                          "time" : ntime      
                    }
                    conn = requests.post("http://140.136.150.100/record.php",data = record_data)
                    print(face_class[0],ntime)
                    print(conn.text)
                    in_or_out = 1
                
                cv2.putText(img, '{:.4f}'.format(t), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                cv2.putText(img, '{:.4f}'.format(min_diff), (100, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
            else:
                img=frame
                t3 = cv2.getTickCount()
                t = (t3-t1)/cv2.getTickFrequency()
                cv2.putText(img, '{:.4f}'.format(t), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
#                cv2.putText(img, '{:.4f}'.format(min_diff), (100, 20),
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        else:
            img=frame
            t4 = cv2.getTickCount()
            t = (t4-t1)/cv2.getTickFrequency()
            cv2.putText(img, '{:.4f}'.format(t), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #圖片隨視窗改變
        #cv2.resize(frame,(x,y))
        frame = cv2.resize(frame,(int((self.ui.label.height()-14)/3)*4,(int((self.ui.label.height()-14)/3)*3)))
        #呈現圖片

        showImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        if self.ui.tabWidget.currentIndex() != 2:
            self.ui.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            self.ui.label_2.setPixmap(QtGui.QPixmap.fromImage(showImage))
        
    def open_detect(self):
        if self.check == 0:
            #self.cap=cv2.VideoCapture(0+cv2.CAP_DSHOW)
            self.cap=cv2.VideoCapture(0)
            self.timer_camera = QtCore.QTimer()
            self.timer_camera.start(150)
            self.timer_camera.timeout.connect(self.show_image)
            self.check = 1
    def close_camera(self):
        if self.check == 1:
            self.timer_camera.stop()
            self.cap.release()
            self.check = 0
    def changeEvent(self,event):
        print(event.type())
        if event.type() == 99:#99是正常視窗狀態
            if self.i == 1:
                if self.check == 0:
                    #self.cap=cv2.VideoCapture(0+cv2.CAP_DSHOW)
                    self.cap=cv2.VideoCapture(0)
                    self.timer_camera = QtCore.QTimer()
                    self.timer_camera.start(150)
                    self.timer_camera.timeout.connect(self.show_image)
                    print("camera open")
                    self.check = 1
            self.i = 0 
        if event.type() == 105:#105是縮小視窗狀態
            if self.i == 0:
                if self.check == 1:
                    self.timer_camera.stop()
                    self.cap.release()
                    print("camera close")
                    self.check = 0
            self.i = 1
    def closeEvent(self,event):
        #global f
        if self.check == 1:
            self.timer_camera.stop() 
            self.cap.release()
            print("camera open")
        else:
            print("camera close")
        #f.close()
        print("close window")
    def stop(self):
        self.close()
        print("stop pressed")
        

    # In[3]:detect face       
def cv2_face(image,a_d_c):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scaled_arr=[]
    faces = faceCascade.detectMultiScale(gray, 1.1, 5)
    if a_d_c == True:
        
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
            #roi_gray = gray[y:y+h, x:x+w]
            #roi_color = image[y:y+h, x:x+w]
            break
        face = gray[left_h:left_h+face_scale,left_w:left_w+face_scale]
        for (x,y,w,h) in faces:
            if x+y+w+h!=0:
                face = gray[y:y+h,x:x+w]
                break
    else:
        cv2.rectangle(image,(left_w,left_h),(left_w+face_scale,left_h+face_scale),(0,255,0),2)
        face = gray[left_h:left_h+face_scale,left_w:left_w+face_scale]
   
    # for (x,y,w,h) in faces:
       # if x+y+w+h!=0:
           # face = gray[y:y+h,x:x+w]
    scaled =cv2.resize(face,(160,160),interpolation=cv2.INTER_LINEAR)
    scaled.astype(float)
    scaled = np.array(scaled).reshape(160, 160, 1)
    scaled = prewhiten(scaled)
    scaled_arr.append(scaled)
    return image,scaled_arr,faces
   
# In[4]prewhiten(calculate distance)
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std,1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x,mean),1/std_adj)
    return y


stylesheet = '''
        QTabWidget {
            background-color: green;
        }
        QTabWidget::pane {
            border: 1px solid #31363B;
            padding: 2px;
            margin:  0px;
        }
        QTabBar {
            border: 0px solid #31363B;
            color: yellow;
        }
        QTabBar::tab:top:selected {
            color: red;
        }
        QPushButton {
            qproperty-alignment: AlignCenter;
            background-color: qlineargradient(x1 : 0, y1 : 0, 
                                              x2 : 0, y2 : 1, 
                                             stop :  0.0 #f5f9ff,
                                             stop :   0.5 #c7dfff,
                                             stop :   0.55 #afd2ff,
                                             stop :   1.0 #c0dbff);
            border-style: outset;
            border-width: 2px;
            border-radius: 10px;
            width: 280px;
            font-size: 50px;
            font: bold;
            color: #006aff;
            font: bold large "Arial";
            height: 50px;
            border-color: beige;
        }
        QLabel {
            font: bold 50px;
        }
        QLineEdit {
            font-size: 50px;
            qproperty-alignment: AlignCenter;
            min-width: 280px;
            min-height: 50px;
        }
        QCheckBox::indicator{
            border-style: outset;
            border-width: 2px;
            border-radius: 20px;
            border-color: black;
            width: 277;
            height: 50;
        }
       
'''
# In[4]:Run program        
if __name__ == "__main__":
    import sys
    with tf.Session() as sess:
        load_model('../model/')
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("Mul:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        keep_probability_placeholder= tf.get_default_graph().get_tensor_by_name('keep_probability:0')
        if not QtWidgets.QApplication.instance():
            app = QtWidgets.QApplication(sys.argv)
            app.setStyleSheet(stylesheet)
        else:
            app = QtWidgets.QApplication.instance()
            app.setStyleSheet(stylesheet)
        myApp = GUI_window()
        sys.exit(app.exec_())
       
