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
import json
import matplotlib.pyplot as plt
#路徑記得改
sys.path.append(r'C:\Users\allen\Desktop\Git_Data\Facial Recognition\test\darknet-master\build\darknet\x64')
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
THRED=0.75
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
    parser.add_argument("--weights", default="./yolo_training/cfg/weights/yolov3-tiny_16000.weights",
                        help="yolo weights path")
    
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    
    parser.add_argument("--ext_output", action='store_false',
                        help="display bbox coordinates of detected objects")
    #config設定檔的路徑要改
    parser.add_argument("--config_file", default="./yolo_training/cfg/yolov3-tiny.cfg",
                        help="path to config file")
    #data的路徑要改
    parser.add_argument("--data_file", default="./yolo_training/cfg/obj.data",
                        help="path to data file")
    
    parser.add_argument("--thresh", type=float, default=.8,
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

# In[]
class Dialog_window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)
        self.ui = uic.loadUi("Dialog.ui",self)
        self.setWindowIcon(QtGui.QIcon("window_icon.png"))
        self.ui.label.setStyleSheet("QLabel{font: bold 28px;}")
        self.OK_button.clicked.connect(self.close_even)
        self.show()
        
    def close_even(self):
        self.close()
        
# In[]
class Dialog2_window(QtWidgets.QDialog):
    def __init__(self,label_flag, parent=None):
        QtWidgets.QDialog.__init__(self)
        self.ui = uic.loadUi("Dialog2.ui",self)
        self.setWindowIcon(QtGui.QIcon("window_icon.png"))
        self.ui.label.setStyleSheet("QLabel{font: bold 24px;}")
        self.ui.label_2.setStyleSheet("QLabel{font: bold 11px;}")
        
        if label_flag == 'good':
            self.ui.label.setText("辨識成功!允許通行!")
        elif label_flag == 'bad':
            self.ui.label.setText("請把口罩戴好!並重新辨識!")
        elif label_flag == 'none':
            self.ui.label.setText("請把口罩戴起來!並重新辨識!")
        
        self.show()
        
        self.timer = QtCore.QTimer()
        self.timer.start(3000) #3000毫秒 = 3秒
        self.timer.timeout.connect(self.close_even)
        
    def close_even(self):
        self.close()
        
# In[]
class Dialog3_window(QtWidgets.QDialog):
    def __init__(self,counter,parent=None):
        QtWidgets.QDialog.__init__(self)
        self.ui = uic.loadUi("Dialog3.ui",self)
        self.setWindowIcon(QtGui.QIcon("window_icon.png"))
        self.ui.label.setStyleSheet("QLabel{font: bold 24px;}")
        self.ui.label_2.setStyleSheet("QLabel{font: bold 11px;}")
        if counter == 3:
            self.ui.label.setText("辨識失敗次數過多!\n請至Setting介面註冊!")
        self.show()
        
        self.timer = QtCore.QTimer()
        if counter == 3:
            self.timer.start(3000) #3000毫秒 = 3秒
        else:
            self.timer.start(2000) #2000毫秒 = 2秒
        self.timer.timeout.connect(self.close_even)
        
    def close_even(self):
        self.close()       
# In[]
class Dialog4_window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)
        self.ui = uic.loadUi("Dialog4.ui",self)
        self.setWindowIcon(QtGui.QIcon("window_icon.png"))
        self.ui.label.setStyleSheet("QLabel{font: bold 20px;}")
        self.ui.label_2.setStyleSheet("QLabel{font: bold 20px;}")
        self.ui.setStyleSheet('''QPushButton {
                                width: 150px;
                                font-size: 30px;
                                height: 50px;}''')
        self.OK_button.clicked.connect(self.close_even)
        
        #放nonde照片
        frame = cv2.imread('none.png')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        self.ui.label_3.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.ui.label_3.setScaledContents (True) #自適應縮放大小
        #放good照片
        frame = cv2.imread('good.png')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        self.ui.label_4.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.ui.label_4.setScaledContents (True) #自適應縮放大小
        
    def close_even(self):
        self.close()  
        
# In[]
class Register_Dialog(QtWidgets.QDialog):
    def __init__(self,var, parent=None):
        QtWidgets.QDialog.__init__(self)
        self.ui = uic.loadUi("Register_Dialog.ui",self)
        self.setWindowIcon(QtGui.QIcon("window_icon.png"))
        self.ui.label.setStyleSheet("QLabel{font: bold 22px;}")
        self.ui.buttonBox.setStyleSheet('''QPushButton {
                                        width: 150px;
                                        font-size: 30px;
                                        height: 50px;}''')
        
        #放照片
        frame = cv2.imread('../new_pictures/'+var+'.jpg')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        self.ui.img_label.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.ui.img_label.setScaledContents (True) #自適應縮放大小
        
        #判斷按OK還是按Cancel
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        

# In[2]:Load Qt UI & setting function
class GUI_window(QtWidgets.QMainWindow):
    def __init__(self,cam_num,parent=None):
        self.cam_num=cam_num
        #載入資料庫的資料
        self.reload()
        
        QtWidgets.QMainWindow.__init__(self)
        #self.setWindowTitle("ggez")
        self.ui = uic.loadUi("webcamV2.ui",self)
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
        self.ui.label_4.setStyleSheet("QLabel{font: bold 28px;}")
        self.ui.setting_checkBox.setStyleSheet('''QCheckBox{font: bold 35px;}
                                                  QCheckBox::indicator{width: 35px;height: 35px}''')
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
        timer.start(10)
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
        self.clock = 1

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
        # In[] 雜七雜八flag
        #紀錄yolo辨識所花時間
        self.yolo_t=0
        #每五分鐘reload記錄一次的flag
        self.r = 0
        #辨識成功視窗flag
        self.show_dialog2 = 0
        #辨識失敗視窗flag
        self.show_dialog3 = 0
        #紀錄辨識失敗次數
        self.dialog3_counter = 0
        #記錄口罩配戴狀況
        self.label_flag =''
        #進入setting畫面時的flag
        self.first_time_enter_setting = 0
        #拍照時用到的flag 要求good跟none各一張
        self.good_seeting=0
        self.none_setting=0
        
        self.show()
        
    # In[]
    def video_capture(self):
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if ret :
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (self.d_width, self.d_height),
                                           interpolation=cv2.INTER_LINEAR)
                frame_resized = cv2.flip(frame_resized,2)
                self.frame_queue.put(frame_resized)
                darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())
                self.darknet_image_queue.put(self.darknet_image)
                #print("fps_clock == ",self.fps_clock)
        print("Thread 1 stop")
        self.cap.release()
    
    def inference(self):
        while self.cap.isOpened():
            darknet_image = self.darknet_image_queue.get()
            prev_time = time.time()
            detections = darknet.detect_image(self.network, self.class_names, self.darknet_image, thresh=self.args.thresh)
            self.detections_queue.put(detections)
            fps = int(1/(time.time() - prev_time))
            #yolo辨識所花時間
            self.yolo_t = time.time() - prev_time
            self.fps_queue.put(fps)
            #print("FPS: {}".format(fps))
            #印出good,bad,none
            darknet.print_detections(detections, self.args.ext_output)
        print("Thread 2 stop")
        self.cap.release()
    
    
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
                
                #進行人臉辨識
                t1=time.time()
                
                for label, confidence, bbox in detections:
                            
                    left,top,right,bottom = darknet.bbox2points(bbox)
                
                    face = image[top:bottom,left:right]
                    
                    self.face = face
                    self.label_flag = label
                    
                    if self.Recognition_check == True:
                        if detections:
                            
                            face = self.face
                                                             
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
                                                                                                     
                                #把人名印在圖片上
                                cv2.putText(image, '{}'.format(face_class[0]), 
                                        (left,top - 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,name_color, 2)
                                
                                #把loss印在人臉附近
                                cv2.putText(image, 'loss:{:.4f}'.format(min_diff),(left,top - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,204,0), 2)
                                
                                
                                #ntime = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
                                                         
                                if self.clock % 120 == 0:
                                    self.recorded_people.clear()
                                
                                print("recorded_people = ",self.recorded_people)
                                
                                if self.ui.tabWidget.currentIndex() == 0:
                                    if self.clock % 120 == 0:
                                        if face_class[0]!='Others' and face_class[0] not in self.recorded_people :
                                        
                                            #人名記錄起來
                                            self.recorded_people.append(face_class[0])
                                            
                                            #上傳資料庫search表
                                            T = time.localtime()
    
                                            record_data = {
                                                  "user_name" : face_class[0],
                                                  "year" : T.tm_year, 
                                                  "month": T.tm_mon,
                                                  "day"  : T.tm_mday,
                                                  "hour" : T.tm_hour,
                                                  "min"  : T.tm_min,
                                                  "sec"  : T.tm_sec,
                                                  "mask" : label,
                                                  "table":'search'
                                            }
                                            conn = requests.post("http://140.136.150.100/record.php",data = record_data)
                                            #print(face_class[0],ntime)
                                            #print(conn.text)       
                                            self.show_dialog2 = 1
                                        
                                        if face_class[0] == 'Others':
                                            self.show_dialog3 = 1
                                            self.dialog3_counter += 1
                                        
                                        
                            
                    if self.Recognition_check == False:
                        
                        if self.ui.tabWidget.currentIndex() == 0:
                            if self.clock % 120 == 0:
                                #上傳資料庫search2表
                                T = time.localtime()
            
                                record_data = {
                                      "user_name" : 'Unknow',
                                      "year" : T.tm_year, 
                                      "month": T.tm_mon,
                                      "day"  : T.tm_mday,
                                      "hour" : T.tm_hour,
                                      "min"  : T.tm_min,
                                      "sec"  : T.tm_sec,
                                      "mask" : self.label_flag,
                                      "table":'search2'
                                }
                                conn = requests.post("http://140.136.150.100/record.php",data = record_data)
                                print(conn.text) 
                                self.show_dialog2 = 1    
                #全部處理完的時間點
                t2=time.time()
                t = int(1/(t2-t1+self.yolo_t))
                #印上辨識時間
                cv2.putText(image, 'FPS:{}'.format(t), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,204,0), 1)
                
                self.YOLO_image_queue.put(image) #把RGB圖片存起來
                self.clock += 1
                #print("cloock=",self.clock)
                
                #if cv2.waitKey(fps) == 27:
                   # break
    
        print("Thread 3 stop")
        self.cap.release()
      
        
    # In[]
    def search_event(self):
        self.ui.listWidget.clear()
        
        #把comboBox的選項一起post出去
        cb = self.ui.comboBox.currentText()
        cb2 = self.ui.comboBox_2.currentText()
        cb3 = self.ui.comboBox_3.currentText()
        
        search_object = self.ui.lineEdit_2.text()
        Data = {
            "user_name" : search_object,
            "cb"        : cb,
            "cb2"       : cb2,
            "cb3"       : cb3
            }
        conn = requests.post('http://140.136.150.100/search.php', data = Data)
        
        #統計人數
        good = 0
        bad  = 0
        none = 0
        for i in json.loads(conn.text):
            if i["mask"] == "none":
                none +=1
            if i["mask"] == "bad":
                bad +=1
            if i["mask"] == "good":
                good +=1
        
        
        labels = []
        size = []
        if(good + bad + none !=0):
            if good !=0:
                labels.append('good')
                size.append(good)
            if bad !=0:
                labels.append('bad')
                size.append(bad)
            if none !=0:
                labels.append('none')
                size.append(none)
            #畫圓餅圖
            plt.pie(size , labels = labels,autopct='%1.1f%%')
            plt.axis('equal')
            plt.savefig("pie.png")
            Pie = QtGui.QPixmap("pie.png")
            Pie.scaled(self.ui.pie_label.size())
            self.ui.pie_label.setScaledContents(True)
            self.ui.pie_label.setPixmap(Pie)
            
        else:
            self.ui.pie_label.clear()
            
        plt.clf()
        var = self.ui.lineEdit_2.setText('')
        
        
        #先加入第一行
        self.ui.listWidget.addItem("User Name\tTime\t\t\t\tMask")
        #把所有結果列出來
        for i in json.loads(conn.text):
            temp = "{U:<12s}\t{Y}/{M}/{D} {H:0>2s}:{Min:0>2s}:{S:0>2s}\t\t{Mask}".format(
                U = i["user_name"],
                Y = i["year"],
                M = i["month"],
                D = i["day"],
                H = i["hour"],
                Min=i["min"],
                S = i["sec"],
                Mask = i["mask"],
                )
            self.ui.listWidget.addItem(temp)
        
        #加入總統計資料
        self.ui.listWidget.addItem("")
        self.ui.listWidget.addItem("good:{g}人\tbad:{b}人\t\tnone:{n}人".format(g=good,b=bad,n=none))
        
        
    
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
        var = self.ui.lineEdit.text()
        if self.check == 1:
                
            if var!='':
                #save_image前置，先儲存照片在本地
                face = self.face
                face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_CUBIC)  
                pic_path = "../new_pictures/"+var+".jpg"
                cv2.imwrite(pic_path,face)
                
                #跳出註冊提醒視窗
                register_dialog = Register_Dialog(var)
                #判斷是否第一次註冊
                if self.ui.setting_checkBox.isChecked():
                    #確認是否拍了good跟none兩張照片                    
                    if self.label_flag == 'good' and self.good_seeting == 0 or self.label_flag == 'none' and self.none_seeting==0:
                        btn = register_dialog.buttonBox.button(QDialogButtonBox.Ok)
                        btn.setEnabled(True)
                    else:
                        btn = register_dialog.buttonBox.button(QDialogButtonBox.Ok)
                        btn.setEnabled(False)
                        btn.setStyleSheet('''background-color: #A0A0A0;
                                             color: #C0C0C0''')
                
                result = register_dialog.exec_()
                if result == 1:
                    if self.label_flag == 'good':
                        self.good_seeting = 1
                    if self.label_flag == "none":
                        self.none_setting = 1
                    #---------------照片上傳雲端--------------------
                    #Drive_upload(pic_path,var)
                    #----------------------------------------------
                    
                    #-----------重新執行特徵分析並上傳資料庫---------
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
                            "user_name" : pic_name,
                            "embedding" : str(x),
                            "date"      : str(ntime)
                        }
                        conn = requests.post("http://140.136.150.100/upload.php", data = Data)
                        #----------------------------------------------------------------------------
                        print(conn.text)
                    #----------------------------------------------
                    
                    #跳出上傳完成視窗
                    dialog = Dialog_window()
                    self.reload()
                    var = self.ui.lineEdit.setText('')
                
                
                #用完就刪除照片
                if os.path.exists(pic_path):
                    os.remove(pic_path)

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
    
        if self.YOLO_image_queue:
            frame = self.YOLO_image_queue.get()#把BGR圖片拿出來用
            
        #隨視窗縮放
        frame = cv2.resize(frame,(int((self.ui.label.height()-16)/3)*4,(int((self.ui.label.height()-16)/3)*3)))
        
        #呈現圖片
        showImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        
        
        if self.ui.tabWidget.currentIndex() == 0:
            self.ui.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
        elif self.ui.tabWidget.currentIndex() == 2:
            self.ui.label_2.setPixmap(QtGui.QPixmap.fromImage(showImage))
        
        #每5分鐘reload一次
        T = time.localtime(time.time())
        if int(T.tm_sec) == 0 and int(T.tm_min) % 5 == 0 and self.r == 0:
            self.reload()
            self.r = 1
        elif int(T.tm_sec) == 1 and self.r == 1:
            self.r = 0
            
        if self.show_dialog2 == 1 and self.ui.tabWidget.currentIndex() == 0:
            dialog2 = Dialog2_window(self.label_flag)
            self.show_dialog2 = 0
            
        if self.show_dialog3 == 1 and self.ui.tabWidget.currentIndex() == 0:
            dialog3 = Dialog3_window(self.dialog3_counter)
            self.show_dialog3 = 0
            
        if self.first_time_enter_setting == 0 and self.ui.tabWidget.currentIndex() == 2:
            dialog4 = Dialog4_window().exec_()
            self.first_time_enter_setting = 1
            self.good_seeting = 0
            self.none_seeting = 0
            
        if self.ui.tabWidget.currentIndex() != 2:
            self.first_time_enter_setting = 0
        
    def open_detect(self):
        if self.check == 0:
            #self.cap=cv2.VideoCapture(0+cv2.CAP_DSHOW)
            self.cap=cv2.VideoCapture(self.cam_num)
            #-------------口罩辨識 多執行續啟動----------
            self.T1 = Thread(target=self.video_capture, args=()).start()
            self.T2 = Thread(target=self.inference, args=()).start()
            self.T3 = Thread(target=self.drawing, args=()).start()
            #-------------------------------------------
            self.timer_camera = QtCore.QTimer()
            self.timer_camera.start(5)
            self.timer_camera.timeout.connect(self.show_image)
            self.check = 1
            
            
    def close_camera(self):
        if self.check == 1:
            self.timer_camera.stop()
            self.cap.release()
            self.check = 0
            self.ui.label.clear()
            self.ui.label.setText("No Signal")
            self.ui.label_2.clear()
            self.ui.label_2.setText("No Signal")
            
     
    def changeEvent(self,event): 
        #print(event.type())
        '''
        if event.type() == 99:#99是正常視窗狀態
            if self.i == 1:
                if self.check == 0:
                    #self.cap=cv2.VideoCapture(0+cv2.CAP_DSHOW)
                    self.cap=cv2.VideoCapture(self.cam_num)
                    self.timer_camera = QtCore.QTimer()
                    self.timer_camera.start(5)
                    self.timer_camera.timeout.connect(self.show_image)
                    print("camera open")
                    self.check = 1
            self.i = 0 
        '''
        if event.type() == 105:#105是縮小視窗狀態
            self.close_camera()
    '''
    def closeEvent(self,event):
        #global f
        if self.check == 1:
            self.timer_camera.stop() 
            #self.cap.release()
            print("camera open")
        else:
            print("camera close")
        #f.close()
        print("close window")
    '''
    def stop(self):
        print("stop pressed")
        self.close_camera()
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
            height: 65px;
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
        myApp = GUI_window(0)
        sys.exit(app.exec_())