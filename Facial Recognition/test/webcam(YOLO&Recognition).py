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
#路徑記得改
sys.path.append(r'C:\Users\A00\Desktop\Git_Data\Facial Recognition\test\darknet-master\build\darknet\x64')
import requests
import Drive_API
from Drive_API import Drive_upload
#import matplotlib.pyplot as plt
# In[] 口罩辨識需要的函式庫
from ctypes import *
import random
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
# In[1]:Load target features
#open camera
THRED=0.95
left_w = 200
left_h = 160
face_scale = 228

# In[]
#-----------------------口罩辨識初始化definiton--------------------------------
def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    #weights的路徑要改
    parser.add_argument("--weights", default="./yolo_training/cfg/weights/yolov3_400000.weights",
                        help="yolo weights path")
    
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    
    parser.add_argument("--ext_output", action='store_false',
                        help="display bbox coordinates of detected objects")
    #config設定檔的路徑要改
    parser.add_argument("--config_file", default="./yolo_training/cfg/yolov3.cfg",
                        help="path to config file")
    #data的路徑要改
    parser.add_argument("--data_file", default="./yolo_training/cfg/obj.data",
                        help="path to data file")
    
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    #if str2int(args.input) == str and not os.path.exists(args.input):
        #raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))



# In[2]:Load Qt UI & setting function
class GUI_window(QtWidgets.QMainWindow):
    def __init__(self,parent=None):
        
        #載入資料庫的資料
        self.reload()
        
        QtWidgets.QMainWindow.__init__(self)
        #self.setWindowTitle("ggez")
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

        #由資料庫抓取資料
        self.ui.reload_image.clicked.connect(self.reload)
        
        self.ui.Recognition_checkbox.clicked.connect(self.Recognition_check_event)
        
        self.Recognition_checkbox.setStyleSheet("QPushButton{background:black;border-radius:20;}"
            )
        
        self.widget.setStyleSheet("QWidget{border-width:2px;}"
                                  "QWidget{border-color:black;}"
                                  "QWidget{border-style:outset;}"
                                  "QWidget{height:100;}"
                                  "QWidget{border-radius:5px;}"
                                  "QWidget{background-color:qlineargradient(x1 : 0, y1 : 0, x2 : 0, y2 : 1, stop :  0.0 #f5f9ff,stop :   0.5 #c7dfff,stop :   0.55 #afd2ff,stop :   1.0 #c0dbff);}")

        self.ui.close_button.clicked.connect(self.close_camera)
        self.ui.tabWidget.setTabIcon(0,QtGui.QIcon("home.png"))
        self.ui.tabWidget.setIconSize(QtCore.QSize(30,30))
        self.ui.tabWidget.setTabIcon(1,QtGui.QIcon("search.png"))
        self.ui.tabWidget.setIconSize(QtCore.QSize(30,30))
        self.ui.tabWidget.setTabIcon(2,QtGui.QIcon("setting.png"))
        self.ui.tabWidget.setIconSize(QtCore.QSize(30,30))
        self.setWindowIcon(QtGui.QIcon("window_icon.png"))

        tab_shape = QtWidgets.QTabWidget.Triangular
        self.tabWidget.setTabShape(tab_shape)
        self.check = 0
        self.i = 0
        self.in_or_out = 0
        
        self.Recognition_check = False
        self.label2 = QLabel(self)

        self.label1 = QLabel(self)

        self.label1.setStyleSheet("QLabel{background:0;}"
                   "QLabel{color:rgb(300,300,300,120);font-size:30px;font-weight:bold;font-family:宋体;}"
                   )
        # 動態顯示時間在label上
        timer = QTimer(self)
        timer.timeout.connect(self.showtime)
        timer.start()
        self.label1.setAlignment(QtCore.Qt.AlignCenter)
        self.label2.setGeometry(self.label.width()+20,150,45,45)
        #self.label1.resize(self.width(),50)
        
        
        # In[]
        #----------------口罩辨識初始化-----------------
        self.frame_queue = Queue()
        self.darknet_image_queue = Queue(maxsize=1)
        self.detections_queue = Queue(maxsize=1)
        self.fps_queue = Queue(maxsize=1)
        #宣告一個放照片的Queue
        self.YOLO_image_queue = Queue(maxsize=1)
        #宣告一個放人臉的Queue
        self.YOLO_face_queue = Queue(maxsize=1)
        
        self.face = 0

        self.recorded_people=[]
        self.clock = 0

        self.args =parser()
        
        check_arguments_errors(self.args)
        self.network, self.class_names, self.class_colors = darknet.load_network(
                self.args.config_file,
                self.args.data_file,
                self.args.weights,
                batch_size=1
            )
        
        # Darknet doesn't accept numpy images.
        # Create one with image we reuse for each detect
        self.d_width = darknet.network_width(self.network)
        self.d_height = darknet.network_height(self.network)
        self.darknet_image = darknet.make_image(self.d_width, self.d_height, 3)
        
        self.show()
    # In[]
    def video_capture(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.d_width, self.d_height),
                                       interpolation=cv2.INTER_LINEAR)
            frame_resized = cv2.flip(frame_resized,2)
            self.frame_queue.put(frame_resized)
            darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
            self.darknet_image_queue.put(self.darknet_image)
        self.cap.release()
        print("Thread 1 stop")
    
    def inference(self):
        while self.cap.isOpened():
            darknet_image = self.darknet_image_queue.get()
            prev_time = time.time()
            detections = darknet.detect_image(self.network, self.class_names, self.darknet_image, thresh=self.args.thresh)
            self.detections_queue.put(detections)
            fps = int(1/(time.time() - prev_time))
            self.fps_queue.put(fps)
            #print("FPS: {}".format(fps))
            #印出good,bad,none
            darknet.print_detections(detections, self.args.ext_output)
        self.cap.release()
        print("Thread 2 stop")
    
    
    def drawing(self):
        
        name_color = (138,43,226)
        random.seed(3)  # deterministic bbox colors
        #video = set_saved_video(cap, args.out_filename, (width, height))
        while self.cap.isOpened():
            frame_resized = self.frame_queue.get()     
            detections = self.detections_queue.get()
            fps = self.fps_queue.get()
            
            
            if frame_resized is not None:
                
                #畫框 + 畫標籤
                image = darknet.draw_boxes(detections, frame_resized, self.class_colors)
                
                if self.Recognition_check == True:
                    if detections:
                        for label, confidence, bbox in detections:
                            
                            left,top,right,bottom = darknet.bbox2points(bbox)
                        
                            face = image[top:bottom,left:right]
                            
                            self.face = face
                            
                            #進行人臉辨識
                            t1=cv2.getTickCount()
                            
                            scaled_arr = None
                            try:
                                scaled_arr = cv2_face(face)
                            except:
                                scaled_arr = None
                                
                            if scaled_arr is not None:
                                feed_dict = { images_placeholder: scaled_arr, phase_train_placeholder:False ,keep_probability_placeholder:1.0}
                                embs = sess.run(embeddings, feed_dict=feed_dict)
                                face_class=['Others']
                                diff = []
                                
                                #尋找最相近的人臉特徵
                                for emb in emb_arr:
                                    diff.append(np.mean(np.square(embs[0] - emb)))
                                min_diff=min(diff)                     
                                
                                index=np.argmin(diff)
                                      
                                if min_diff<THRED: 
                                    face_class[0]=class_arr[index]
                                    
                                
                                t2=cv2.getTickCount()
                                t=(t2-t1)/cv2.getTickFrequency()
                                
                                #把人名印在圖片上
                                cv2.putText(image, '{}'.format(face_class[0]), 
                                        (left,top - 35), 
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,name_color, 2)
                                
                                ntime = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
                                
                                if self.clock % 100 == 0:
                                    self.recorded_people.clear()
                                
                                print("recorded_people = ",self.recorded_people)
                                
                                if face_class[0]!='Others' and face_class[0] not in self.recorded_people:
                                    
                                    if self.clock % 30 == 0:
                                        #人名記錄起來
                                        self.recorded_people.append(face_class[0])
                                        
                                        #上傳資料庫search表
                                        record_data = {
                                              "user_name" : face_class[0],
                                              "time" : ntime, 
                                              "mask" : label
                                        }
                                        
                                        conn = requests.post("http://140.136.150.100/record.php",data = record_data)
                                        #print(face_class[0],ntime)
                                        print(conn.text)
                                        
                                    
                                
                                #印上辨識時間 & 誤差
                                cv2.putText(image, '{:.4f}'.format(t), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
                                cv2.putText(image, '{:.4f}'.format(min_diff), (100, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        
                
                self.YOLO_image_queue.put(image) #把RGB圖片存起來
                self.clock += 1
                print("cloock=",self.clock)
                
                if cv2.waitKey(fps) == 27:
                    break
                
        self.cap.release()
        print("Thread 3 stop")
        cv2.destroyAllWindows()
    # In[]
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
    
    def Recognition_check_event(self):
        
        if self.Recognition_check == False:
            self.widget.raise_()
            self.Recognition_checkbox.raise_()
            self.anim = QPropertyAnimation(self.Recognition_checkbox, b"geometry")
            self.anim.setDuration(400)
            self.anim.setStartValue(QRect(11,11,25,25))
            self.anim.setEndValue(QRect(245,11,25,25))
            self.anim.start()
            self.Recognition_check = not self.Recognition_check
        else:
            self.anim = QPropertyAnimation(self.Recognition_checkbox, b"geometry")
            self.anim.setDuration(400)
            self.anim.setStartValue(QRect(245,11,25,25))
            self.anim.setEndValue(QRect(11,11,25,25))
            self.anim.start()
            self.Recognition_check = not self.Recognition_check
    
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
            
        
        if var!='':
            face = self.face
            face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_CUBIC)     
            
            pic_path = "../new_pictures/"+var+".jpg"
            
            cv2.imwrite(pic_path,face)
            
            #---------------照片上傳雲端---------------------
            
            Drive_upload(pic_path,var)
            
            #-----------------------------------------------
            frame = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            self.ui.label_2.setPixmap(QtGui.QPixmap.fromImage(showImage))
            
            #-----------重新執行特徵分析並上傳資料庫----------
            embeddings_pre.main() 
            #----------------------------------------------
            self.reload()
            
            #用完就刪除照片
            if os.path.exists(pic_path):
                os.remove(pic_path)
            
            
            print('-----Save完成-----')
            var = self.ui.lineEdit.setText('')
            
    
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
                arr1.append(data["user_name"])
                temp = []
                for e in data['embedding'][2:-2].split():
                    temp.append(float(e))
                arr2.append(temp)
                stop = 0
                hold = ""
                
        class_arr = np.array(arr1)
        emb_arr = np.array(arr2)
        print("-----Reload完成----")
        
    def show_image(self):
        global emb_arr,class_arr
        global in_or_out

    
        if self.YOLO_image_queue:
            frame = self.YOLO_image_queue.get()#把BGR圖片拿出來用
        
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
            #-------------口罩辨識 多執行續啟動----------
            self.T1 = Thread(target=self.video_capture, args=()).start()
            self.T2 = Thread(target=self.inference, args=()).start()
            self.T3 = Thread(target=self.drawing, args=()).start()
            #-------------------------------------------
            self.timer_camera = QtCore.QTimer()
            self.timer_camera.start(70)
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
                    self.timer_camera.start(70)
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
        print("stop pressed")
        cv2.destroyAllWindows()
        time.sleep(1)
        self.close()
        
        
        

    # In[3]:detect face   
def cv2_face(face):
    face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
    #轉灰階
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    scaled_arr=[]
    #cv2.imshow("face",face)
    
    
    scaled =cv2.resize(face,(160,160),interpolation=cv2.INTER_LINEAR)
    scaled.astype(float)
    scaled = np.array(scaled).reshape(160, 160, 1)
    scaled = prewhiten(scaled)
    scaled_arr.append(scaled)
    
    
    return scaled_arr

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
       
